
import numpy as np
import json
import h5py
import os
from torch.utils.data import Dataset
import torch


class CustomH5Dataset(Dataset):
    """
    Custom Dataset for loading .h5 point cloud files
    
    Expected structure:
    - JSON file: {"filename1.h5": label_id, "filename2.h5": label_id, ...}
    - H5 files: Each containing point cloud data
    """
    
    def __init__(self, json_path, data_dir, num_points=1024, split='train', normalize=True):
        """
        Args:
            json_path: Path to JSON file with {filename: label} mapping
            data_dir: Directory containing .h5 files
            num_points: Number of points to sample from each cloud
            split: 'train' or 'test' (for future train/test split)
            normalize: Whether to normalize point clouds to unit sphere
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.normalize = normalize
        
        # Load JSON mapping
        with open(json_path, 'r') as f:
            self.file_to_label = json.load(f)
        
        # Get list of files and labels
        self.files = list(self.file_to_label.keys())
        self.labels = [self.file_to_label[f] for f in self.files]
        
        # Get unique classes
        self.classes = sorted(list(set(self.labels)))
        self.num_classes = len(self.classes)
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        
        print(f"Loaded {len(self.files)} samples")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Get filename and label
        filename = self.files[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        # Load point cloud from H5 file
        filepath = os.path.join(self.data_dir, filename)
        point_cloud = self.load_h5(filepath)
        
        # Sample or pad to fixed number of points
        point_cloud = self.resample_points(point_cloud, self.num_points)
        
        # Normalize if requested
        if self.normalize:
            point_cloud = self.normalize_point_cloud(point_cloud)
        
        # Convert to torch tensor
        point_cloud = torch.from_numpy(point_cloud).float()
        label_idx = torch.tensor(label_idx).long()
        
        return point_cloud, label_idx
    
    def load_h5(self, filepath):
        """
        Load point cloud from H5 file.
        Adjust the key name based on your H5 file structure.
        """
        with h5py.File(filepath, 'r') as f:
            # Common key names in H5 files: 'data', 'points', 'xyz', 'coordinates'
            # Try different possible keys
            possible_keys = ['data', 'points', 'xyz', 'coordinates', 'point_cloud']
            
            for key in possible_keys:
                if key in f.keys():
                    points = f[key][:]
                    return points
            
            # If none of the common keys found, print available keys
            print(f"Available keys in {filepath}: {list(f.keys())}")
            # Use the first available key
            first_key = list(f.keys())[0]
            print(f"Using key: {first_key}")
            points = f[first_key][:]
            return points
    
    def resample_points(self, points, num_points):
        """
        Resample point cloud to fixed number of points.
        """
        N = points.shape[0]
        
        if N >= num_points:
            # Random sampling
            choice = np.random.choice(N, num_points, replace=False)
        else:
            # Random sampling with replacement (pad)
            choice = np.random.choice(N, num_points, replace=True)
        
        points = points[choice, :]
        return points
    
    def normalize_point_cloud(self, points):
        """
        Normalize point cloud to unit sphere centered at origin.
        """
        # Center at origin
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        return points


# Alternative: If your H5 files contain multiple point clouds
class MultiSampleH5Dataset(Dataset):
    """
    Dataset loader for H5 files where each file contains multiple samples.
    
    Expected H5 structure:
    - 'data': shape (N, num_points, 3) - point clouds
    - 'label': shape (N,) - labels for each point cloud
    """
    
    def __init__(self, h5_files, num_points=1024, normalize=True):
        """
        Args:
            h5_files: List of H5 file paths
            num_points: Number of points per cloud
            normalize: Whether to normalize
        """
        self.num_points = num_points
        self.normalize = normalize
        
        # Load all data
        self.data = []
        self.labels = []
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                data = f['data'][:]
                labels = f['label'][:]
                
                self.data.append(data)
                self.labels.append(labels)
        
        # Concatenate all data
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        self.num_classes = len(np.unique(self.labels))
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        point_cloud = self.data[idx]
        label = self.labels[idx]
        
        # Resample
        if point_cloud.shape[0] != self.num_points:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, 
                                     replace=(point_cloud.shape[0] < self.num_points))
            point_cloud = point_cloud[choice, :]
        
        # Normalize
        if self.normalize:
            centroid = np.mean(point_cloud, axis=0)
            point_cloud = point_cloud - centroid
            max_dist = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
            if max_dist > 0:
                point_cloud = point_cloud / max_dist
        
        point_cloud = torch.from_numpy(point_cloud).float()
        label = torch.tensor(label).long()
        
        return point_cloud, label


def inspect_h5_file(filepath):
    """
    Utility function to inspect the structure of an H5 file.
    Run this first to understand your data format.
    """
    print(f"\nInspecting: {filepath}")
    print("=" * 60)
    
    with h5py.File(filepath, 'r') as f:
        print("Keys in file:")
        for key in f.keys():
            data = f[key]
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
            
            # Print first few values if it's small
            if data.size < 20:
                print(f"    Values: {data[:]}")
        
        # Try to load and display some info
        if 'data' in f.keys():
            points = f['data'][:]
            print(f"\nPoint cloud info:")
            print(f"  Shape: {points.shape}")
            print(f"  Min: {points.min()}, Max: {points.max()}")
            print(f"  Mean: {points.mean()}")


if __name__ == "__main__":
    # Example usage:
    
    # 1. First inspect your H5 file to see the structure
    print("Inspecting H5 file structure...")
    inspect_h5_file("path/to/your/sample.h5")
    
    # 2. Load dataset with JSON mapping
    print("\n\nLoading dataset...")
    dataset = CustomH5Dataset(
        json_path="path/to/your/mapping.json",
        data_dir="path/to/h5/files",
        num_points=1024,
        normalize=True
    )
    
    # 3. Test loading a sample
    print("\nTesting data loading...")
    point_cloud, label = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Label: {label}")
    print(f"Point cloud range: [{point_cloud.min():.3f}, {point_cloud.max():.3f}]")