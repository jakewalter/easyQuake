#!/usr/bin/env python3

import h5py
import numpy as np

# Examine the original HDF5 weights file
print("Examining model_pol_best.hdf5 structure:")
with h5py.File('model_pol_best.hdf5', 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
            if 'weight' in name.lower() and obj.size < 50:
                print(f"    Values: {obj[:]}")
        else:
            print(f"  Group: {name}")
    
    f.visititems(print_structure)
    
    # Look for specific weight patterns
    print("\nSearching for final layer weights...")
    def find_final_weights(name, obj):
        if isinstance(obj, h5py.Dataset) and ('dense_3' in name or 'activation_7' in name):
            print(f"FINAL LAYER: {name} - Shape: {obj.shape}")
            if obj.size < 100:
                print(f"Values: {obj[:]}")
    
    f.visititems(find_final_weights)
