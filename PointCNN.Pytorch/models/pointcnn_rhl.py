import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def knn(x, k):
    """
    K nearest neighbors
    Args:
        x: (batch_size, num_dims, num_points)
        k: number of nearest neighbors
    Returns:
        idx: (batch_size, num_points, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Extract graph features
    Args:
        x: (batch_size, num_dims, num_points)
        k: number of nearest neighbors
        idx: (batch_size, num_points, k) indices
    Returns:
        feature: (batch_size, num_dims*2, num_points, k)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature  # (batch_size, num_dims*2, num_points, k)


class XConv(nn.Module):
    """
    X-Convolution layer
    """
    def __init__(self, in_channels, out_channels, k=16, dilation=1):
        super(XConv, self).__init__()
        self.k = k
        self.dilation = dilation
        
        # X-transformation network
        self.x_trans = nn.Sequential(
            nn.Conv2d(3, k, 1, bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, k * k, 1, bias=False),
        )
        
        # Feature transformation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, num_points)
        Returns:
            (batch_size, out_channels, num_points)
        """
        batch_size, _, num_points = x.size()
        
        # Get local features
        x_features = get_graph_feature(x, k=self.k)  # (B, C*2, N, k)
        
        # Get coordinates for X-transformation
        xyz = x[:, :3, :]  # (B, 3, N)
        xyz_features = get_graph_feature(xyz, k=self.k)[:, :3, :, :]  # (B, 3, N, k)
        
        # Compute X-transformation matrix
        X = self.x_trans(xyz_features)  # (B, k*k, N, k)
        X = X.permute(0, 2, 3, 1).contiguous()  # (B, N, k, k*k)
        X = X.view(batch_size, num_points, self.k, self.k, self.k)
        X = X.mean(dim=-1)  # (B, N, k, k) - average over the last dimension
        
        # Apply X-transformation to features
        x_features = x_features.permute(0, 2, 3, 1).contiguous()  # (B, N, k, C*2)
        x_trans = torch.matmul(X, x_features)  # (B, N, k, C*2)
        x_trans = x_trans.permute(0, 3, 1, 2).contiguous()  # (B, C*2, N, k)
        
        # Feature extraction
        x_conv = self.conv1(x_trans)  # (B, out_channels, N, k)
        x_conv = self.conv2(x_conv)  # (B, out_channels, N, k)
        
        # Max pooling
        x_conv = x_conv.max(dim=-1, keepdim=False)[0]  # (B, out_channels, N)
        
        return x_conv


class PointCNN(nn.Module):
    """
    PointCNN for classification
    """
    def __init__(self, num_classes=40, input_channels=3, k=16):
        super(PointCNN, self).__init__()
        self.k = k
        
        # X-Conv layers
        self.xconv1 = XConv(input_channels, 64, k=k)
        self.xconv2 = XConv(64, 128, k=k)
        self.xconv3 = XConv(128, 256, k=k)
        self.xconv4 = XConv(256, 512, k=k)
        
        # Global feature
        self.conv_global = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, num_points) or (batch_size, num_points, 3)
        Returns:
            (batch_size, num_classes)
        """
        # Handle both input formats
        if x.size(1) != 3 and x.size(2) == 3:
            x = x.transpose(2, 1).contiguous()
        
        batch_size = x.size(0)
        
        # X-Conv layers
        x1 = self.xconv1(x)      # (B, 64, N)
        x2 = self.xconv2(x1)     # (B, 128, N)
        x3 = self.xconv3(x2)     # (B, 256, N)
        x4 = self.xconv4(x3)     # (B, 512, N)
        
        # Global features
        x_global = self.conv_global(x4)  # (B, 1024, N)
        
        # Global max pooling
        x_global = torch.max(x_global, 2, keepdim=False)[0]  # (B, 1024)
        
        # Classification head
        x = F.relu(self.bn1(self.fc1(x_global)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dp2(x)
        x = self.fc3(x)
        
        return x


def get_model(num_classes=40, input_channels=3, k=16):
    """
    Get PointCNN model
    Args:
        num_classes: number of classification classes
        input_channels: number of input channels (3 for xyz, 6 for xyz+normals, etc.)
        k: number of nearest neighbors
    Returns:
        model
    """
    model = PointCNN(num_classes=num_classes, input_channels=input_channels, k=k)
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing PointCNN model...")
    model = get_model(num_classes=40)
    x = torch.randn(2, 1024, 3)  # (batch, points, channels)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
