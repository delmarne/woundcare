


import h5py
import json
import os
import numpy as np


def inspect_json(json_path):
    """Inspect JSON file structure"""
    print("\n" + "="*60)
    print("INSPECTING JSON FILE")
    print("="*60)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"JSON file: {json_path}")
    print(f"Total entries: {len(data)}")
    
    # Show first 5 entries
    print("\nFirst 5 entries:")
    for i, (key, value) in enumerate(list(data.items())[:5]):
        print(f"  {key}: {value}")
    
    # Count labels
    labels = list(data.values())
    unique_labels = set(labels)
    print(f"\nUnique labels: {len(unique_labels)}")
    print(f"Labels: {sorted(unique_labels)}")
    
    # Label distribution
    print("\nLabel distribution:")
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")
    
    return data


def inspect_h5_file(h5_path):
    """Inspect a single H5 file structure"""
    print("\n" + "="*60)
    print("INSPECTING H5 FILE")
    print("="*60)
    
    print(f"File: {h5_path}")
    
    if not os.path.exists(h5_path):
        print(f"ERROR: File not found!")
        return None
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nKeys in file: {list(f.keys())}")
        
        for key in f.keys():
            data = f[key]
            print(f"\nKey: '{key}'")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            
            # Load data for analysis
            arr = data[:]
            print(f"  Min value: {arr.min()}")
            print(f"  Max value: {arr.max()}")
            print(f"  Mean value: {arr.mean():.4f}")
            
            # If it's 3D point cloud data
            if len(data.shape) == 2 and data.shape[1] == 3:
                print(f"  ✓ Looks like point cloud data (N x 3)")
                print(f"  Number of points: {data.shape[0]}")
            elif len(data.shape) == 3 and data.shape[2] == 3:
                print(f"  ✓ Looks like batched point cloud data (Batch x N x 3)")
                print(f"  Batch size: {data.shape[0]}")
                print(f"  Points per cloud: {data.shape[1]}")
            
            # Show sample of small arrays
            if arr.size <= 20:
                print(f"  Values: {arr}")
    
    return True


def inspect_all_h5_in_directory(data_dir, max_files=5):
    """Inspect multiple H5 files in a directory"""
    print("\n" + "="*60)
    print("INSPECTING H5 FILES IN DIRECTORY")
    print("="*60)
    
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    print(f"Found {len(h5_files)} H5 files in {data_dir}")
    
    print(f"\nInspecting first {min(max_files, len(h5_files))} files...")
    
    for i, filename in enumerate(h5_files[:max_files]):
        filepath = os.path.join(data_dir, filename)
        print(f"\n--- File {i+1}: {filename} ---")
        
        with h5py.File(filepath, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            for key in f.keys():
                print(f"  {key}: {f[key].shape}")


def create_sample_json(data_dir, output_json="sample_mapping.json"):
    """Create a sample JSON mapping file from H5 files in directory"""
    print("\n" + "="*60)
    print("CREATING SAMPLE JSON MAPPING")
    print("="*60)
    
    h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    # Create dummy mapping
    mapping = {}
    for i, filename in enumerate(h5_files):
        # Assign labels (you should replace this with your actual labels)
        label = i % 10  # Just cycle through 0-9 as example
        mapping[filename] = label
    
    with open(output_json, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Created sample JSON mapping: {output_json}")
    print(f"Total files: {len(mapping)}")
    print(f"⚠ WARNING: This uses dummy labels! Update with your actual labels.")
    
    return mapping


def main():
    """Main inspection function"""
    print("="*60)
    print("H5 DATA INSPECTOR")
    print("="*60)
    
    # Update these paths to match your data
    JSON_PATH = "D:\MD_Implementations\PointNet++\dict_wounds.json"
    DATA_DIR = "D:\MD_Implementations\PointNet++\sorted_data"
    SAMPLE_H5 = "D:\MD_Implementations\PointNet++\sorted_data\test\Diabetic Neuropathic Ulcer\kaggle_D100_0.h5"  # One example H5 file
    
    print("\nInstructions:")
    print("1. Update the paths above to match your data location")
    print("2. Run this script to inspect your data")
    print("3. Check that your data format matches what PointNet++ expects")
    
    # Check if paths exist
    if os.path.exists(JSON_PATH):
        inspect_json(JSON_PATH)
    else:
        print(f"\n⚠ JSON file not found: {JSON_PATH}")
    
    if os.path.exists(DATA_DIR):
        inspect_all_h5_in_directory(DATA_DIR)
    else:
        print(f"\n⚠ Data directory not found: {DATA_DIR}")
    
    if os.path.exists(SAMPLE_H5):
        inspect_h5_file(SAMPLE_H5)
    else:
        print(f"\n⚠ Sample H5 file not found: {SAMPLE_H5}")
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Quick usage examples:
    
    # Example 1: Inspect a single H5 file
    # inspect_h5_file("your_file.h5")
    
    # Example 2: Inspect JSON mapping
    # inspect_json("your_mapping.json")
    
    # Example 3: Check all H5 files in directory
    # inspect_all_h5_in_directory("your_data_folder")
    
    # Run main function
    main()