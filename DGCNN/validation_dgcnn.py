import sys
import torch
import joblib
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from types import SimpleNamespace

# --------------------------
# Paths & Config
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "woundcare_dataset.pkl"
MODEL_FILE = "trained_dgcnn_model.pth"
BATCH_SIZE = 32

# --------------------------
# Import correct DGCNN
# --------------------------
sys.path.insert(0, r"D:\MD_Implementations\Woundcare\DGCNN Cloned Repo\dgcnn\pytorch")
from model import DGCNN

# --------------------------
# Load dataset
# --------------------------
data = joblib.load(DATA_FILE)
X_val, y_val = data["X_test"], data["y_test"]  # or "X_val"/"y_val" if you split
label_encoder = data["label_encoder"]

# Create Dataset class
class PointCloudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

val_dataset = PointCloudDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# Initialize model
# --------------------------
args = SimpleNamespace(
    k=5,
    emb_dims=1024,
    dropout=0.5,
    output_channels=7,  # MUST match training!
    use_bn=True
)

model = DGCNN(args, output_channels=args.output_channels).to(DEVICE)

# Verify the file being imported
import model as imported_model
print("Using model.py from:", imported_model.__file__)

# --------------------------
# Load trained weights
# --------------------------
checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

# --------------------------
# Evaluation
# --------------------------
class_correct = defaultdict(int)
class_total = defaultdict(int)

with torch.no_grad():
    for pcs, labels in val_loader:
        pcs, labels = pcs.to(DEVICE), labels.to(DEVICE)
        pcs = pcs[..., :3]  # use XYZ only
        preds = model(pcs.transpose(1, 2))
        predicted = preds.argmax(dim=1)

        for l, p in zip(labels, predicted):
            class_total[int(l)] += 1
            if l == p:
                class_correct[int(l)] += 1

print("\nValidation Results:")
for class_idx in sorted(class_total.keys()):
    acc = class_correct[class_idx] / class_total[class_idx] if class_total[class_idx] > 0 else 0
    class_name = label_encoder.inverse_transform([class_idx])[0] if label_encoder else str(class_idx)
    print(f"Class '{class_name}': Accuracy {acc:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")

overall_acc = sum(class_correct.values()) / sum(class_total.values())
print(f"\nOverall Accuracy: {overall_acc:.4f}")

