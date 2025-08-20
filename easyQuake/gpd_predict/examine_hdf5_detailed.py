#!/usr/bin/env python3

import h5py
import numpy as np

def examine_hdf5_structure(filename):
    print(f"Examining HDF5 file: {filename}")
    print("=" * 50)
    
    with h5py.File(filename, 'r') as f:
        def print_structure(name, obj):
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Group):
                print(f"{indent}GROUP: {name}")
                # Print attributes if any
                if obj.attrs:
                    for attr_name, attr_val in obj.attrs.items():
                        print(f"{indent}  ATTR: {attr_name} = {attr_val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}DATASET: {name}")
                print(f"{indent}  Shape: {obj.shape}")
                print(f"{indent}  Type: {obj.dtype}")
                print(f"{indent}  Size: {obj.size} elements")
                
                # For conv1d layers, check if shapes make sense for CNN
                if 'conv1d' in name.lower() and 'kernel' in name.lower():
                    print(f"{indent}  -> CNN kernel weights detected")
                elif 'dense' in name.lower() and 'kernel' in name.lower():
                    print(f"{indent}  -> Dense layer weights detected")
                
                # Show small arrays
                if obj.size <= 20:
                    print(f"{indent}  Values: {obj[:]}")
        
        print("\nHDF5 structure:")
        f.visititems(print_structure)
        
        # Look for specific CNN layer patterns
        print("\n" + "=" * 50)
        print("CNN Layer Analysis:")
        
        def find_cnn_layers(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Look for conv1d patterns
                if 'conv1d' in name.lower():
                    shape = obj.shape
                    if len(shape) == 3:  # Conv1D kernel shape should be (kernel_size, input_channels, output_channels)
                        print(f"Conv1D kernel found: {name} -> shape {shape}")
                    elif len(shape) == 1:  # bias
                        print(f"Conv1D bias found: {name} -> shape {shape}")
                        
                # Look for dense patterns
                elif 'dense' in name.lower():
                    shape = obj.shape
                    if len(shape) == 2:  # Dense kernel shape should be (input_units, output_units)
                        print(f"Dense kernel found: {name} -> shape {shape}")
                    elif len(shape) == 1:  # bias
                        print(f"Dense bias found: {name} -> shape {shape}")
        
        f.visititems(find_cnn_layers)

if __name__ == "__main__":
    examine_hdf5_structure('model_pol_best.hdf5')
