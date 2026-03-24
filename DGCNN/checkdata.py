import os
import glob
import pandas as pd

# === Step 1: Find all h5 files ===
DATA_DIR = "/Users/mariannydeleon/Woundcare/Data/*.h5"
files = glob.glob(DATA_DIR)

print(f"Found {len(files)} h5 files in {os.path.dirname(DATA_DIR)}")

if len(files) == 0:
    raise FileNotFoundError("No .h5 files found. Check your path.")

# === Step 2: Load the first file as Pandas DataFrame ===
first_file = files[0]
print(f"\nInspecting file: {first_file}")

try:
    df = pd.read_hdf(first_file, key="df")
    print("\n✅ Successfully loaded as Pandas DataFrame!")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
except Exception as e:
    print("\n⚠️ Could not load file with Pandas. Error:", e)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
labels_int = le.fit_transform(df['woundType'])
# Now labels_int is numeric (0,1,2...) instead of string

point_clouds = []
point_labels = []

for wound, group in df.groupby('woundID'):
    points = group[['x','y','z']].to_numpy()  # shape: (num_points_in_wound, 3)
    point_clouds.append(points)
    point_labels.append(le.transform([group['woundType'].iloc[0]])[0])

# import h5py
# import numpy as np
# import glob
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# import h5py

# with h5py.File("Data/kaggle_1_0.h5", "r") as f:
#     def print_h5_structure(name, obj):
#         print(name, "->", type(obj))
#     f.visititems(print_h5_structure)

# # === Step 1: Locate your .h5 files ===
# # Replace this path with your dataset folder
# DATA_DIR = "/Users/mariannydeleon/Woundcare/Data/*.h5"
# files = glob.glob(DATA_DIR)

# print(f"Found {len(files)} h5 files in {os.path.dirname(DATA_DIR)}")

# # === Step 2: Inspect the first file ===
# if len(files) > 0:
#     with h5py.File(files[0], "r") as f:
#         print("\nKeys inside first file:", list(f.keys()))
#         for key in f.keys():
#             print(f"{key} shape:", f[key].shape)

#     # Load the first file
#     with h5py.File(files[0], "r") as f:
#         data = np.array(f['data'])
#         labels = np.array(f['label'])

#     print("\nData shape:", data.shape)     # (N, P, 3) expected
#     print("Label shape:", labels.shape)    # (N, 1) expected
#     print("Unique labels:", np.unique(labels))

#     # === Step 3: Visualize one sample ===
#     sample = data[0]  # first wound sample
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=5, c='blue')
#     ax.set_title(f"Example wound point cloud - Label {labels[0]}")
#     plt.show()

# # === Step 4: Load ALL files and compute class distribution ===
# all_data, all_labels = [], []
# for f in files:
#     with h5py.File(f, "r") as h5_file:
#         all_data.append(np.array(h5_file["data"]))
#         all_labels.append(np.array(h5_file["label"]))

# all_data = np.concatenate(all_data, axis=0)
# all_labels = np.concatenate(all_labels, axis=0)

# print("\n=== Dataset Summary ===")
# print("Total samples:", all_data.shape[0])
# print("Points per sample:", all_data.shape[1] if all_data.ndim > 1 else "unknown")
# print("Number of classes:", len(np.unique(all_labels)))
# print("Class distribution:")

# unique, counts = np.unique(all_labels, return_counts=True)
# for u, c in zip(unique, counts):
#     print(f"  Class {u}: {c} samples")