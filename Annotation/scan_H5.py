import h5py
import numpy as np
import os

def scan_h5_file(filepath):
    """Comprehensively scan an H5 file structure"""
    filepath = os.path.normpath(filepath)
    
    print(f"\n{'='*60}")
    print(f"Scanning: {filepath}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at: {filepath}")
        return
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(" File Structure:")
            print("-" * 60)
            
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent} {name}")
                    print(f"{indent}   Shape: {obj.shape}")
                    print(f"{indent}   Dtype: {obj.dtype}")
                    print(f"{indent}   Size: {obj.size} elements")
                    
                    # Show sample only for numeric data
                    if np.issubdtype(obj.dtype, np.number):
                        if obj.size < 10:
                            print(f"{indent}   Data: {obj[:]}")
                        else:
                            print(f"{indent}   Sample: {obj[:min(5, obj.shape[0])]}")
                    elif obj.dtype.kind in ['S', 'O', 'U']:  # String or object
                        print(f"{indent}   Sample: {obj[:min(5, len(obj))]}")
                    print()
                elif isinstance(obj, h5py.Group):
                    print(f"{indent} {name}/")
            
            f.visititems(print_structure)
            
            # Summary
            print("\n" + "="*60)
            print(" Data Summary:")
            print("-" * 60)
            
            # Parse the pandas DataFrame structure
            if 'df' in f:
                print("\n This is a pandas DataFrame in HDF5 format!")
                print("\nAvailable data:")
                
                # Get coordinates (x, y, z)
                if 'df/block1_values' in f:
                    coords = f['df/block1_values'][:]
                    print(f"   Coordinates (XYZ): {coords[:, :3].shape}")
                    print(f"     X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
                    print(f"     Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
                    print(f"     Z range: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
                    
                    if coords.shape[1] >= 6:
                        print(f"   Normals (Nx,Ny,Nz): {coords[:, 3:6].shape}")
                
                # Get colors
                if 'df/block2_values' in f:
                    colors = f['df/block2_values'][:]
                    print(f"   Colors (RGB): {colors.shape}")
                    print(f"     R range: [{colors[:, 0].min()}, {colors[:, 0].max()}]")
                    print(f"     G range: [{colors[:, 1].min()}, {colors[:, 1].max()}]")
                    print(f"     B range: [{colors[:, 2].min()}, {colors[:, 2].max()}]")
                
                # Get labels
                if 'df/block0_values' in f:
                    print(f"    Labels available in df/block0_values")
                
                print(f"\n  Total points: {coords.shape[0]:,}")
                    
    except Exception as e:
        print(f" ERROR reading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    filepath = "D:/MD_Implementations/DGCNN/sorted_data/train/Diabetic Neuropathic Ulcer/kaggle_D1_0.h5"
    scan_h5_file(filepath)
