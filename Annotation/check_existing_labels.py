import h5py
import numpy as np

def check_labels(filepath):
    """Check what labels are in data"""
    with h5py.File(filepath, 'r') as f:
        # Get the label data
        block0_values = f['df/block0_values'][0]  # It's stored as object array
        
        print(f"Label data type: {type(block0_values)}")
        print(f"Label shape: {block0_values.shape if hasattr(block0_values, 'shape') else 'N/A'}")
        print(f"Label sample: {block0_values[:20]}")  # First 20 labels
        print(f"Unique labels: {np.unique(block0_values)}")
        print(f"Label counts: {np.bincount(block0_values) if block0_values.dtype == np.uint8 else 'N/A'}")

filepath = "D:/MD_Implementations/DGCNN/sorted_data/train/Diabetic Neuropathic Ulcer/kaggle_D1_0.h5"
check_labels(filepath)
