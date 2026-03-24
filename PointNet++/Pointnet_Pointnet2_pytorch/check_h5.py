import pandas as pd
import numpy as np

# Replace this with a path to any of your actual .h5 files
filepath = "D:/MD_Implementations/PointNet++/sorted_data/train/Unstageable/kaggle_Unstageable_008.h5" 

df = pd.read_hdf(filepath, 'df')
print("Columns in H5 file:", df.columns.tolist())
print("Data shape:", df.shape)
print("\nFirst 3 rows:\n", df.head(3))