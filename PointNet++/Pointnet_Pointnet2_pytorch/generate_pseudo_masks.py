import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

def compute_curvedness_cpu(xyz, k=16):
    """Memory-efficient CPU calculation of differential geometry."""
    # 1. Fast nearest neighbors using cKDTree
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k)
    
    # 2. Gather neighbors and compute local covariance
    neighbors = xyz[indices] # [N, k, 3]
    center = xyz[:, np.newaxis, :] # [N, 1, 3]
    diff = neighbors - center # [N, k, 3]
    
    # Batched matrix multiplication for covariance
    cov = np.matmul(diff.transpose(0, 2, 1), diff) / k # [N, 3, 3]
    
    # 3. Eigen decomposition
    eigenvalues = np.linalg.eigvalsh(cov) # [N, 3] (Sorted ascending)
    k1 = eigenvalues[:, 2] # Max curvature
    k2 = eigenvalues[:, 1] # Min curvature
    
    # 4. Curvedness
    curvedness = np.sqrt((k1**2 + k2**2) / 2.0)
    return curvedness

def apply_saliency_seeding(data_dir):
    splits = ['train', 'test']
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        print(f"\nProcessing {split} directory...")
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in tqdm(os.listdir(class_dir), desc=f"Class: {class_name}"):
                if not filename.endswith('.h5'):
                    continue
                    
                filepath = os.path.join(class_dir, filename)
                df = pd.read_hdf(filepath, 'df')
                
                # If mask already exists, skip
                if 'mask' in df.columns:
                    continue
                
                # Extract XYZ coordinates
                xyz = df[['x', 'y', 'z']].values.astype(np.float32)
                
                # Compute curvedness (No OOM errors here!)
                curvedness = compute_curvedness_cpu(xyz, k=16)
                
                # Saliency threshold: top 15% most curved/irregular points become pseudo-wound
                threshold = np.percentile(curvedness, 85)
                mask = np.zeros(xyz.shape[0], dtype=np.int64)
                mask[curvedness > threshold] = 1 
                
                # Save the new mask column back to the file
                df['mask'] = mask
                df.to_hdf(filepath, key='df', mode='w')

if __name__ == "__main__":
    DATA_DIR = "D:/MD_Implementations/PointNet++/sorted_data"
    apply_saliency_seeding(DATA_DIR)
    print("\nPseudo-masks generated successfully!")