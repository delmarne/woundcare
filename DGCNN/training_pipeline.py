import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import numpy as np
import sys
from sklearn.model_selection import train_test_split

from types import SimpleNamespace

sys.path.append(r"D:\MD_Implementations\Woundcare\DGCNN Cloned Repo\dgcnn\pytorch")

from model import DGCNN


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --------------------------
# Configuration
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DATA_FILE = "woundcare_dataset.pkl"



# --------------------------
# Dataset Class
# --------------------------
class PointCloudDataset(Dataset):
    def __init__(self, X, y):
        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        # Optional: check
        if self.y.min() < 0 or self.y.max() >= 1085:
            raise ValueError(f"Labels out of bounds: min={self.y.min()}, max={self.y.max()}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------
# Load Dataset
# --------------------------
data = joblib.load(DATA_FILE)
X_train_full, y_train_full = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]  # keep test set if needed

# --------------------------
# Split training into train/val
# --------------------------


X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

# --------------------------
# Create Datasets and Loaders
# --------------------------
train_dataset = PointCloudDataset(X_train, y_train)
val_dataset = PointCloudDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


args = SimpleNamespace(
    k=5,               # number of nearest neighbors
    emb_dims=1024,      # embedding dimension
    dropout=0.5,        # dropout rate
    output_channels=,  # number of classes
    use_bn=True         # batch norm
)

# --------------------------
# Model Setup (Dynamic Class Count)
# --------------------------
# Number of classes
num_classes = len(data["label_encoder"].classes_)
print(f"Initializing DGCNN for {num_classes} classes")

# Initialize model
model = DGCNN(args, output_channels=args.output_channels).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --------------------------
# Sanity Check: Make sure labels match the model's output range
# --------------------------
all_labels = []
for _, labels in train_loader:
    all_labels.append(labels)
all_labels = torch.cat(all_labels)

print("Label min:", all_labels.min().item())
print("Label max:", all_labels.max().item())
print("Expected range: 0 to", num_classes - 1)

# Optional: keep a single reference variable for consistency
NUM_CLASSES = num_classes
# --------------------------


# --------------------------

# Training Loop with Safeguards
# --------------------------
original_k = getattr(model, "k", 20)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    skipped_batches = 0

    for batch_idx, (pc, labels) in enumerate(train_loader):
        pc, labels = pc.to(DEVICE), labels.to(DEVICE)

        # --------------------------
        # Ensure proper shapes and types
        # --------------------------
        labels = labels.squeeze()           # remove extra dimensions
        labels = labels.long()              # ensure integer type

        if labels.ndim != 1 or labels.size(0) != pc.size(0):
            print(f"Skipping batch {batch_idx}: labels shape invalid {labels.shape}")
            skipped_batches += 1
            continue

        if (labels < 0).any() or (labels >= NUM_CLASSES).any():
            print(f"Skipping batch {batch_idx}: invalid label range {labels.min().item()}–{labels.max().item()}")
            skipped_batches += 1
            continue

        # --------------------------
        # Adjust k for small point clouds
        # --------------------------
        pc = pc[..., :3]  # only XYZ
        batch_size, num_points, _ = pc.shape
        required_k = getattr(model, "k", 20)
        if num_points <= required_k:
            if num_points > 1:
                model.k = num_points - 1
            else:
                print(f"Skipping batch {batch_idx}: too few points ({num_points})")
                skipped_batches += 1
                continue

        # --------------------------
        # Forward, Loss, Backward
        # --------------------------
        optimizer.zero_grad()
        preds = model(pc.transpose(1, 2))

        if preds.shape[0] != labels.shape[0]:
            print(f"Skipping batch {batch_idx}: batch size mismatch preds {preds.shape[0]} vs labels {labels.shape[0]}")
            skipped_batches += 1
            continue

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        model.k = original_k  # restore k

    avg_loss = total_loss / max(1, len(train_loader) - skipped_batches)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — Train Loss: {avg_loss:.4f} (skipped {skipped_batches} batches)")

    # --------------------------
    # Validation
    # --------------------------
    model.eval()
    correct, total = 0, 0
    skipped_val_batches = 0
    with torch.no_grad():
        for batch_idx, (pc, labels) in enumerate(val_loader):
            pc, labels = pc.to(DEVICE), labels.to(DEVICE)
            labels = labels.squeeze().long()

            if labels.ndim != 1 or labels.size(0) != pc.size(0):
                skipped_val_batches += 1
                continue

            if (labels < 0).any() or (labels >= NUM_CLASSES).any():
                skipped_val_batches += 1
                continue

            pc = pc[..., :3]
            batch_size, num_points, _ = pc.shape
            required_k = getattr(model, "k", 20)
            if num_points <= required_k:
                if num_points > 1:
                    model.k = num_points - 1
                else:
                    skipped_val_batches += 1
                    continue

            preds = model(pc.transpose(1, 2))
            if preds.shape[0] != labels.shape[0]:
                skipped_val_batches += 1
                continue

            predicted = preds.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            model.k = original_k

    val_acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — Val Accuracy: {val_acc:.4f} (skipped {skipped_val_batches} batches)")

print("Training complete.")

MODEL_PATH = "trained_dgcnn_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")