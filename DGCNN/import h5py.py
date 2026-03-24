import torch

import h5py
import numpy as np
import glob

from torch.utils.data import Dataset, DataLoader

files = glob.glob("/Users/mariannydeleon/Woundcare/Data/*.h5")
all_data = []
all_labels = []

for f in files:
    with h5py.File(f, 'r') as h5_file:
        data = np.array(h5_file['data'])
        label = np.array(h5_file['label'])
        all_data.append(data)
        all_labels.append(label)

all_data = np.array(all_data)       # shape: (num_samples, num_points, 3)
all_labels = np.array(all_labels)   # shape: (num_samples,)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)



class WoundDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = WoundDataset(X_train, y_train)
val_dataset = WoundDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)