import os
import h5py
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ===============================
# Configuration
# ===============================
DATA_DIR = r"D:\MD_Implementations\Woundcare\sorted_data"
OUTPUT_FILE = "woundcare_dataset.pkl"
NUM_POINTS = 1024  # number of points per point cloud

data = joblib.load("woundcare_dataset.pkl")
print(data.keys())
# ===============================
# Helper Functions
# ===============================
def decode_bytes(arr):
    """Decode byte strings to regular strings."""
    if isinstance(arr, (np.ndarray, list)):
        return [x.decode('utf-8') if isinstance(x, bytes) else x for x in arr]
    elif isinstance(arr, bytes):
        return arr.decode('utf-8')
    return arr


def extract_point_clouds(file_path, num_points=NUM_POINTS):
    """Load an HDF5 file and extract one point cloud sample (numeric columns only)."""
    df = pd.read_hdf(file_path, key='df')
    
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Remove rows with NaN or Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.shape[0] == 0:
        raise ValueError(f"No valid numeric data in {file_path}")
    
    # Sample points
    if df.shape[0] >= num_points:
        indices = np.random.choice(df.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(df.shape[0], num_points, replace=True)

    return df.iloc[indices].values.astype(np.float32)


def load_split(split_path):
    """Load data from a given split directory (train/val/test)."""
    X, y = [], []
    for label_name in sorted(os.listdir(split_path)):
        label_path = os.path.join(split_path, label_name)
        if not os.path.isdir(label_path):
            continue

        print(f"Loading class: {label_name}")
        for file_name in os.listdir(label_path):
            if not file_name.endswith(".h5"):
                continue
            file_path = os.path.join(label_path, file_name)
            try:
                pc = extract_point_clouds(file_path)
                X.append(pc)
                y.append(label_name)
            except Exception as e:
                print(f"Skipping {file_name}: {e}")
    return np.array(X, dtype=np.float32), np.array(y)


# ===============================
# Main Processing
# ===============================
splits = {}
le = LabelEncoder()

# Collect all class names (from train split)
train_path = os.path.join(DATA_DIR, "train")
classes = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
le.fit(classes)
print(f" Found {len(classes)} classes: {classes}")

# Load each split
for split in ["train", "val", "test"]:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        print(f" No '{split}' folder found, skipping...")
        continue
    print(f"\n Loading {split} data...")
    X, y = load_split(split_path)
    y_encoded = le.transform(y)
    splits[f"X_{split}"] = X
    splits[f"y_{split}"] = y_encoded
    print(f" {split} loaded: {X.shape[0]} samples")

# ===============================
# Save Dataset
# ===============================
splits["label_encoder"] = le
joblib.dump(splits, OUTPUT_FILE)

print("\n Dataset ready for DGCNN or PointCNN!")
print(f" Saved to: {OUTPUT_FILE}")
for split in ["train", "val", "test"]:
    if f"X_{split}" in splits:
        print(f"  {split}: {splits[f'X_{split}'].shape[0]} samples")


