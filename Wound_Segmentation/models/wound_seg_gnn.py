import torch
import torch.nn as nn
import torch.nn.functional as F
from models.geometry_utils import compute_differential_geometry, knn


def get_graph_feature(x, k=32, idx=None):
    # [Unchanged from your original code]
    B, C, N = x.size()
    if idx is None:
        idx = knn(x[:, :3, :], k=k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x_flat = x.transpose(2, 1).contiguous().view(B * N, C)
    neighbors = x_flat[idx, :].view(B, N, k, C)

    x_center = x.transpose(2, 1).unsqueeze(2).repeat(1, 1, k, 1)
    features = torch.cat((neighbors - x_center, x_center), dim=3).permute(0, 3, 1, 2).contiguous()
    return features


class WoundSegmentationGNN(nn.Module):
    def __init__(self, k=32, num_classes=2):
        super(WoundSegmentationGNN, self).__init__()
        self.k = k

        # Saliency Seeder: Now takes 5 geometry features
        self.saliency_seeder = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Input features: 3 (XYZ) + 5 (Geom) + 1 (Saliency) = 9
        in_channels = 9

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, kernel_size=1)
        )

    def forward(self, points):
        # 1. Extract Differential Geometry Features
        geom_feats = compute_differential_geometry(points, k=self.k)  # [B, 2, N]

        # 2. Saliency Seeding (Outlier Detection)
        saliency_map = self.saliency_seeder(geom_feats)  # [B, 1, N]

        # Combine XYZ, geometry features, and the new saliency map: [B, 6, N]
        x = torch.cat([points, geom_feats, saliency_map], dim=1)

        # 3. Graph Propagation (Layer 1)
        graph_feat1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(graph_feat1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # 4. Graph Propagation (Layer 2)
        graph_feat2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(graph_feat2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        # Concatenate features from all layers
        global_feat = torch.cat([x1, x2], dim=1)  # [B, 128, N]

        # 5. Generate per-point segmentation mask
        logits = self.classifier(global_feat)  # [B, 2, N]

        return logits

def get_model(num_classes=2):
    return WoundSegmentationGNN(num_classes=num_classes)