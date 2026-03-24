import os
import shutil
import json

# ======== CONFIG =========
UNSORTED_DIR = r"D:\2d_h5_files_alpha"
SORTED_DIR = r"D:\MD_Implementations\Woundcare\sorted_data"
DICT_FILE = r"D:\MD_Implementations\Woundcare\dict_wounds.json"  # path to your dictionary file
# =========================

# Load your train/val/test dictionary
with open(DICT_FILE, "r") as f:
    split_dict = json.load(f)

# Loop through splits (train, val, test)
for split, classes in split_dict.items():
    for class_name, files in classes.items():
        class_dir = os.path.join(SORTED_DIR, split, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for fname in files:
            src_path = os.path.join(UNSORTED_DIR, fname)
            dst_path = os.path.join(class_dir, fname)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f" Missing file: {fname}")

print(" All available files copied to sorted_data successfully!")
