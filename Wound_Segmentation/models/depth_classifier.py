import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PointNetEncoder(nn.Module):
    """Lightweight 3D geometry branch."""
    def __init__(self, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x is [B, C, N]
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # Max pooling over points
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # Dummy trans_feat for compatibility with your train.py
        trans_feat = None 
        return x, trans_feat

class SEFusionBlock(nn.Module):
    """Squeeze-and-Excitation channel attention mechanism to fuse 2D and 3D streams."""
    def __init__(self, channel, reduction=16):
        super(SEFusionBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is [B, C]
        b, c = x.size()
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c)
        return x * y

class MCDropout(nn.Dropout):
    """Monte Carlo Dropout that remains active during inference for uncertainty scores."""
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)

class DEPTHClassifier(nn.Module):
    """Dual-stream Encoder with Prototypical Training and Hierarchical curriculum."""
    def __init__(self, num_classes=5, channel=3):
        super(DEPTHClassifier, self).__init__()
        
        # 1. 3D Stream: Lightweight PointNet
        self.stream_3d = PointNetEncoder(channel=channel)
        
        # 2. 2D Stream: MobileNetV3-Large (Feature Extractor)
        mobilenet = models.mobilenet_v3_large(weights=None)
        # Remove the classification head, keep the features and pooling
        self.stream_2d = nn.Sequential(
            mobilenet.features,
            mobilenet.avgpool,
            nn.Flatten()
        )
        # MobileNetV3-Large outputs 960 features
        
        # 3. Squeeze-and-Excitation Fusion
        fused_dim = 1024 + 960 # PointNet(1024) + MobileNet(960)
        self.se_fusion = SEFusionBlock(channel=fused_dim)
        
        # 4. Monte Carlo Dropout Head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            MCDropout(p=0.5), # MC Dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            MCDropout(p=0.5), # MC Dropout
            nn.Linear(256, num_classes)
        )

    def generate_dummy_2d_projection(self, points):
        """
        Temporary utility to generate a dummy 2D projection [B, 3, 224, 224] 
        from the 3D points [B, C, N] to feed the MobileNet stream.
        """
        B = points.size(0)
        device = points.device
        # Create a dummy image tensor matching MobileNetV3 expected input
        dummy_img = torch.zeros((B, 3, 224, 224), device=device)
        return dummy_img

    def forward(self, points):
        """
        Input: points [B, C, N]
        Returns: logits [B, num_classes], trans_feat
        """
        # 3D Stream extraction
        feat_3d, trans_feat = self.stream_3d(points)
        
        # 2D Stream extraction (using dummy projection for now)
        img_2d = self.generate_dummy_2d_projection(points)
        feat_2d = self.stream_2d(img_2d)
        
        # Concatenate features
        fused_feat = torch.cat([feat_3d, feat_2d], dim=1)
        
        # Squeeze-and-Excitation Fusion
        attended_feat = self.se_fusion(fused_feat)
        
        # Classification Head
        logits = self.classifier(attended_feat)
        
        return logits, trans_feat

def get_model(num_classes, normal_channel=False):
    """Wrapper function to match your train.py initialization standard."""
    channel = 6 if normal_channel else 3
    return DEPTHClassifier(num_classes=num_classes, channel=channel)