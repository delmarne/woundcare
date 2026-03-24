import torch

def knn(x, k):
    """
    Computes K-Nearest Neighbors.
    Input: x [B, C, N]
    Output: idx [B, N, K]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # Get indices of the top k nearest points
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx

def compute_differential_geometry(points, k=16):
    """
    Computes principal curvatures, shape index, and curvedness.
    Input: points [B, 3, N]
    """
    B, C, N = points.size()
    
    # 1. Get K-Nearest Neighbors
    idx = knn(points, k=k) # [B, N, K]
    
    # Gather neighbors
    idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    idx_expanded = idx + idx_base
    points_flat = points.transpose(2, 1).contiguous().view(B*N, C)
    neighbors = points_flat[idx_expanded.view(-1)].view(B, N, k, C)
    
    # 2. Compute local covariance matrix
    center = points.transpose(2, 1).unsqueeze(2) # [B, N, 1, C]
    diff = neighbors - center # [B, N, k, C]
    cov = torch.matmul(diff.transpose(3, 2), diff) / k # [B, N, C, C]
    
    # 3. Eigen decomposition for principal curvatures
    # eigenvalues sorted in ascending order. 
    # The smallest eigenvalue corresponds to the normal vector.
    # The other two are related to principal curvatures k1 and k2.
    eigenvalues = torch.linalg.eigvalsh(cov) # [B, N, 3]
    
    # Approximate principal curvatures from eigenvalues (scaled)
    k1 = eigenvalues[:, :, 2] # Max curvature
    k2 = eigenvalues[:, :, 1] # Min curvature
    
    # Add small epsilon to prevent division by zero
    eps = 1e-6
    
    # 4. Shape Index (S) ranges from [-1, 1]
    shape_index = (2.0 / torch.pi) * torch.atan((k1 + k2) / (k1 - k2 + eps))
    
    # 5. Curvedness (C)
    curvedness = torch.sqrt((k1**2 + k2**2) / 2.0)
    
    # Combine features: [B, 2, N]
    geom_features = torch.stack([shape_index, curvedness], dim=1)
    
    return geom_features