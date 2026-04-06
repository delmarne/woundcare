import os
import h5py
import pandas as pd
import numpy as np

DATA_DIR = r"D:\MD_Implementations\Woundcare\sorted_data"
NUM_POINTS = 1024  # optional, for point cloud checks

def extract_point_clouds(file_path):
    """Try to load an HDF5 file and return True if it contains valid numeric data."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Check common keys
            if 'df' in f:
                data = f['df'][:]
            elif 'df/block1_values' in f:
                data = f['df/block1_values'][:]
            else:
                print(f"Skipped {file_path}: no valid dataset found")
                return False

            # Convert to DataFrame to check numeric
            df = pd.DataFrame(data)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if df.shape[0] == 0:
                print(f"Skipped {file_path}: no valid data after cleaning")
                return False
        return True
    except Exception as e:
        print(f"Skipped {file_path}: {e}")
        return False

# --------------------------
# Scan all classes
# --------------------------
class_counts = {}

for split in ["train", "val", "test"]:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        continue

    print(f"\nScanning {split} split...")
    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        valid_files = 0
        total_files = 0
        for file_name in os.listdir(class_path):
            if not file_name.endswith(".h5"):
                continue
            total_files += 1
            file_path = os.path.join(class_path, file_name)
            if extract_point_clouds(file_path):
                valid_files += 1

        class_counts[(split, class_name)] = (valid_files, total_files)
        print(f"Class '{class_name}': {valid_files}/{total_files} valid files")

# --------------------------
# Summary
# --------------------------
print("\n=== Dataset Summary ===")
for (split, class_name), (valid, total) in class_counts.items():
    print(f"{split} - {class_name}: {valid}/{total} valid samples")
