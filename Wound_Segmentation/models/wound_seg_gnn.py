import torch
import torch.nn as nn
import torch.nn.functional as F
from models.geometry_utils import compute_differential_geometry, knn

def get_graph_feature(x, k=16, idx=None):
    """Constructs edge features for the GNN based on nearest neighbors."""
    B, C, N = x.size()
    if idx is None:
        idx = knn(x[:, :3, :], k=k) # Use only XYZ for neighbor search
        
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x_flat = x.transpose(2, 1).contiguous().view(B*N, C)
    neighbors = x_flat[idx, :].view(B, N, k, C)
    
    x_center = x.transpose(2, 1).unsqueeze(2).repeat(1, 1, k, 1)
    # Concatenate center point features with the difference to neighbors
    features = torch.cat((neighbors - x_center, x_center), dim=3).permute(0, 3, 1, 2).contiguous()
    return features

class WoundSegmentationGNN(nn.Module):
    """
    Lightweight GNN for whole-wound segmentation using differential geometry.
    """
    def __init__(self, k=16, num_classes=2):
        super(WoundSegmentationGNN, self).__init__()
        self.k = k
        
        # Input features: 3 (XYZ) + 2 (Shape Index, Curvedness) = 5
        in_channels = 5
        
        # EdgeConv Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # EdgeConv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Point-wise classification head
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, kernel_size=1)
        )

    def forward(self, points):
        """
        Input: points [B, 3, N]
        Output: segmentation_logits [B, num_classes, N]
        """
        # 1. Extract Differential Geometry Features (C.3.1 Saliency Seeding)
        geom_feats = compute_differential_geometry(points, k=self.k)
        
        # Combine XYZ with geometry features: [B, 5, N]
        x = torch.cat([points, geom_feats], dim=1)
        
        # 2. Graph Propagation (Layer 1)
        graph_feat1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(graph_feat1)
        x1 = x1.max(dim=-1, keepdim=False)[0] # Max pooling over neighbors
        
        # 3. Graph Propagation (Layer 2)
        graph_feat2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(graph_feat2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        # Concatenate features from all layers
        global_feat = torch.cat([x1, x2], dim=1) # [B, 128, N]
        
        # 4. Generate per-point segmentation mask
        logits = self.classifier(global_feat) # [B, 2, N]
        
        return logits
        
def get_model(num_classes=2):
    return WoundSegmentationGNN(num_classes=num_classes)