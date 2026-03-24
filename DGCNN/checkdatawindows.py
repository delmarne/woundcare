import os
import h5py

DATA_DIR = r"D:\2d_h5_files_alpha"

def scan_h5_file(file_path):
    """
    Recursively scan an HDF5 file and print all datasets with type, shape, and dtype.
    """
    print(f"\nScanning file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        def print_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_dataset)


def scan_all_files(data_dir):
    for fname in os.listdir(data_dir):
        if fname.endswith(".h5"):
            file_path = os.path.join(data_dir, fname)
            scan_h5_file(file_path)


# Run the scan
scan_all_files(DATA_DIR)
