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


def compute_differential_geometry(points, k=32):
    """
    Computes principal curvatures, shape index, curvedness, and normal deviation.
    Input: points [B, 3, N]
    Output: geom_features [B, 5, N]
    """
    B, C, N = points.size()

    idx = knn(points, k=k)
    idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    idx_expanded = idx + idx_base
    points_flat = points.transpose(2, 1).contiguous().view(B * N, C)
    neighbors = points_flat[idx_expanded.view(-1)].view(B, N, k, C)

    center = points.transpose(2, 1).unsqueeze(2)
    diff = neighbors - center
    cov = torch.matmul(diff.transpose(3, 2), diff) / k

    eigenvalues = torch.linalg.eigvalsh(cov)

    k1 = eigenvalues[:, :, 2]  # Max curvature
    k2 = eigenvalues[:, :, 1]  # Min curvature
    e_min = eigenvalues[:, :, 0]  # Variance along the normal vector

    eps = 1e-6

    shape_index = (2.0 / torch.pi) * torch.atan((k1 + k2) / (k1 - k2 + eps))
    curvedness = torch.sqrt((k1 ** 2 + k2 ** 2) / 2.0)

    # Surface normal deviation proxy (change in local surface orientation)
    normal_deviation = e_min / (eigenvalues.sum(dim=-1) + eps)

    # Combine features: [B, 5, N]
    geom_features = torch.stack([k1, k2, shape_index, curvedness, normal_deviation], dim=1)

    return geom_features