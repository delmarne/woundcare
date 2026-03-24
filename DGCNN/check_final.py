import joblib
from collections import Counter

# Load the woundcare dataset
dataset = joblib.load("woundcare_dataset.pkl")

# Inspect keys
print("Keys in dataset:", dataset.keys())

# Access train/val splits
X_train = dataset["X_train"]
y_train = dataset["y_train"]
X_val = dataset["X_val"]
y_val = dataset["y_val"]

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Number of features per point cloud: {X_train.shape[1:]}")  # (num_points, num_features)

# Optional: count per-class distribution
train_dist = Counter(y_train)
val_dist   = Counter(y_val)

print("\n=== Training set class distribution ===")
for label, count in sorted(train_dist.items()):
    pct = count / len(y_train) * 100
    print(f"Class {label}: {count} samples ({pct:.2f}%)")

print("\n=== Validation set class distribution ===")
for label, count in sorted(val_dist.items()):
    pct = count / len(y_val) * 100
    print(f"Class {label}: {count} samples ({pct:.2f}%)")
