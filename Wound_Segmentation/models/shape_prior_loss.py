import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalShapePriorLoss(nn.Module):
    def __init__(self, target_aspect_ratio_max=3.0, target_depth_variance_max=0.1):
        super(ClinicalShapePriorLoss, self).__init__()
        # Clinical hyperparameters (tune these based on your actual dataset measurements)
        self.max_aspect = target_aspect_ratio_max
        self.max_depth_var = target_depth_variance_max

    def forward(self, logits, points):
        """
        logits: [B, num_classes, N] (from the GNN)
        points: [B, 3, N] (the raw XYZ coordinates)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Get the probability of the "wound" class (assuming index 1 is wound)
        # Shape: [B, N]
        wound_prob = probs[:, 1, :]

        # Prevent division by zero if the network predicts absolutely no wound
        total_prob = torch.clamp(wound_prob.sum(dim=1, keepdim=True), min=1e-6)

        # 1. Calculate Weighted Centroid (Center of Mass)
        # points is [B, 3, N]. We multiply by wound_prob [B, 1, N]
        weighted_points = points * wound_prob.unsqueeze(1)
        centroid = weighted_points.sum(dim=2, keepdim=True) / total_prob.unsqueeze(1)  # [B, 3, 1]

        # Calculate spatial variance (spread of the wound)
        # (points - centroid)^2 weighted by probability
        variance = (wound_prob.unsqueeze(1) * (points - centroid) ** 2).sum(dim=2) / total_prob  # [B, 3]

        # variance[:, 0] is X spread, [:, 1] is Y spread, [:, 2] is Z (depth) spread
        var_x = variance[:, 0]
        var_y = variance[:, 1]
        var_z = variance[:, 2]  # Depth profile

        # 2. Aspect Ratio Penalty
        # Wounds are rarely perfect lines. We penalize extreme aspect ratios (e.g., VarX >> VarY)
        # Add epsilon to prevent divide by zero
        aspect_ratio_1 = (var_x + 1e-6) / (var_y + 1e-6)
        aspect_ratio_2 = (var_y + 1e-6) / (var_x + 1e-6)
        max_aspect_ratio = torch.max(aspect_ratio_1, aspect_ratio_2)

        aspect_penalty = F.relu(max_aspect_ratio - self.max_aspect).mean()

        # 3. Depth Profile Penalty
        # Wounds should be relatively localized in depth compared to the full 3D scan
        depth_penalty = F.relu(var_z - self.max_depth_var).mean()

        # 4. Convexity Proxy (Spatial Spread vs Total Probability Mass)
        # If the wound is fragmented, the variance will be huge compared to the area.
        # We want to encourage a contiguous blob.
        area_proxy = total_prob.squeeze()
        spread = var_x + var_y
        convexity_penalty = F.relu((spread / (area_proxy + 1e-6)) - 0.5).mean()

        # Combine penalties
        total_prior_loss = aspect_penalty + depth_penalty + convexity_penalty
        return total_prior_loss