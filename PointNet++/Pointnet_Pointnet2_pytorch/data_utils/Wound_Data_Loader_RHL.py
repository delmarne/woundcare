"""
Custom Dataset Loader for Wound Classification H5 Data
Save as: data_utils/WoundDataLoader.py

H5 files contain pandas DataFrames with columns: x, y, z, Nx, Ny, Nz, r, g, b, etc.
"""

import numpy as np
import json
import pandas as pd
import os
from torch.utils.data import Dataset
import torch


class WoundDataset(Dataset):
    """
    Dataset for wound classification with pandas HDF5 format
    """
    
    def __init__(self, json_path, data_dir, split='train', num_points=1024, normalize=True, use_normals=False, use_colors=False):
        """
        Args:
            json_path: Path to JSON file
            data_dir: Root directory (contains train/ and test/ subdirs)
            split: 'train' or 'test'
            num_points: Number of points to sample
            normalize: Whether to normalize to unit sphere
            use_normals: Include normal vectors (Nx, Ny, Nz)
            use_colors: Include RGB colors
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.normalize = normalize
        self.split = split
        self.use_normals = use_normals
        self.use_colors = use_colors
        
        # Load JSON
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        
        if split not in data_dict:
            raise ValueError(f"Split '{split}' not found in JSON")
        
        split_data = data_dict[split]
        
        # Parse nested structure
        self.files = []
        self.labels = []
        self.class_names = sorted(list(split_data.keys()))
        self.num_classes = len(self.class_names)
        
        # Class name to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Collect all files and labels
        for class_name, file_list in split_data.items():
            class_idx = self.class_to_idx[class_name]
            for filename in file_list:
                # Check if file exists
                filepath = os.path.join(self.data_dir, split, class_name, filename)
                if os.path.exists(filepath):
                    self.files.append(filename)
                    self.labels.append(class_idx)
        
        print(f"\n{split.upper()} Dataset Info:")
        print(f"  Total samples: {len(self.files)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Classes: {self.class_names}")
        
        # Show class distribution
        print(f"\n  Class distribution:")
        for class_name in self.class_names:
            count = sum(1 for label in self.labels if label == self.class_to_idx[class_name])
            print(f"    {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Get filename and label
        filename = self.files[idx]
        label = self.labels[idx]
        class_name = self.get_class_name(label)
        
        # Build filepath with subdirectories: data_dir/split/class_name/filename
        filepath = os.path.join(self.data_dir, self.split, class_name, filename)
        
        try:
            point_cloud = self.load_h5_pandas(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return dummy data
            num_features = 3
            if self.use_normals:
                num_features += 3
            if self.use_colors:
                num_features += 3
            point_cloud = np.random.rand(self.num_points, num_features).astype(np.float32)
        
        # Sample or pad to fixed number of points
        point_cloud = self.resample_points(point_cloud, self.num_points)
        
        # Normalize XYZ coordinates if requested
        if self.normalize:
            point_cloud[:, :3] = self.normalize_point_cloud(point_cloud[:, :3])
        
        # Convert to torch tensor
        point_cloud = torch.from_numpy(point_cloud).float()
        label = torch.tensor(label).long()
        
        return point_cloud, label
    
    def load_h5_pandas(self, filepath):
        """Load point cloud from pandas HDF5 file"""
        # Read the dataframe
        df = pd.read_hdf(filepath, 'df')
        
        # Extract XYZ coordinates (required)
        xyz = df[['x', 'y', 'z']].values.astype(np.float32)
        
        # Optionally add normals
        if self.use_normals and all(col in df.columns for col in ['Nx', 'Ny', 'Nz']):
            normals = df[['Nx', 'Ny', 'Nz']].values.astype(np.float32)
            xyz = np.concatenate([xyz, normals], axis=1)
        
        # Optionally add colors (normalized to 0-1)
        if self.use_colors and all(col in df.columns for col in ['r', 'g', 'b']):
            colors = df[['r', 'g', 'b']].values.astype(np.float32) / 255.0
            xyz = np.concatenate([xyz, colors], axis=1)
        
        return xyz
    
    def resample_points(self, points, num_points):
        """Resample point cloud to fixed number of points"""
        N = points.shape[0]
        
        if N >= num_points:
            # Random sampling without replacement
            choice = np.random.choice(N, num_points, replace=False)
        else:
            # Random sampling with replacement (pad)
            choice = np.random.choice(N, num_points, replace=True)
        
        return points[choice, :]
    
    def normalize_point_cloud(self, points):
        """Normalize XYZ coordinates to unit sphere centered at origin"""
        # Center at origin
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        return points
    
    def get_class_name(self, idx):
        """Get class name from index"""
        return self.class_names[idx]


def test_dataset(json_path, data_dir):
    """Test function to verify dataset loading"""
    print("="*60)
    print("TESTING DATASET")
    print("="*60)
    
    # Test train dataset
    print("\nLoading TRAIN dataset...")
    train_dataset = WoundDataset(json_path, data_dir, split='train', num_points=1024)
    
    # Test test dataset  
    print("\nLoading TEST dataset...")
    test_dataset = WoundDataset(json_path, data_dir, split='test', num_points=1024)
    
    print("\n" + "="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    # Test loading first sample from train
    if len(train_dataset) > 0:
        point_cloud, label = train_dataset[0]
        print(f"\nTrain Sample 0:")
        print(f"  Point cloud shape: {point_cloud.shape}")
        print(f"  Label: {label} ({train_dataset.get_class_name(label)})")
        print(f"  Point cloud range: [{point_cloud.min():.3f}, {point_cloud.max():.3f}]")
    
    # Test loading first sample from test
    if len(test_dataset) > 0:
        point_cloud, label = test_dataset[0]
        print(f"\nTest Sample 0:")
        print(f"  Point cloud shape: {point_cloud.shape}")
        print(f"  Label: {label} ({test_dataset.get_class_name(label)})")
        print(f"  Point cloud range: [{point_cloud.min():.3f}, {point_cloud.max():.3f}]")
    
    print("\n✓ Dataset loading successful!")
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test with your paths
    JSON_PATH = "D:/MD_Implementations/PointNet++/dict_wounds.json"
    DATA_DIR = "D:/MD_Implementations/PointNet++/sorted_data"
    
    test_dataset(JSON_PATH, DATA_DIR)

class WoundSegDataset(Dataset):
    """
    Dataset for Point Cloud Segmentation (Background vs Wound)
    """
    def __init__(self, json_path, data_dir, split='train', num_points=4096, mask_column='mask'):
        self.data_dir = data_dir
        self.num_points = num_points
        self.split = split
        self.mask_column = mask_column # The column name we find in Step 1
        
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
            
        split_data = data_dict[split]
        self.files = []
        self.class_names = sorted(list(split_data.keys()))
        
        # Collect all files
        for class_name, file_list in split_data.items():
            for filename in file_list:
                filepath = os.path.join(self.data_dir, split, class_name, filename)
                if os.path.exists(filepath):
                    self.files.append((filepath, class_name))
                    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filepath, _ = self.files[idx]
        
        # Read the dataframe
        df = pd.read_hdf(filepath, 'df')
        
        # Extract XYZ coordinates
        xyz = df[['x', 'y', 'z']].values.astype(np.float32)
        
        # Extract the point-level mask (0 for healthy, 1 for wound)
        # We will update 'mask' based on the output of your check script
        if self.mask_column in df.columns:
            mask = df[self.mask_column].values.astype(np.int64)
        else:
            # Fallback if no mask exists yet (just for testing the pipeline)
            mask = np.zeros((xyz.shape[0],), dtype=np.int64) 
            
        # Resample to fixed number of points (e.g., 4096 for segmentation)
        N = xyz.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.random.choice(N, self.num_points, replace=True)
            
        xyz = xyz[choice, :]
        mask = mask[choice]
        
        # Center the points
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        xyz = xyz / max_dist
        
        return torch.from_numpy(xyz).float(), torch.from_numpy(mask).long()