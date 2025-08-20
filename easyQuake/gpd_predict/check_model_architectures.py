#!/usr/bin/env python3

import tensorflow as tf
import os

# Check which models can load and have the correct CNN architecture
models_to_check = [
    'model_pol_new.keras',
    'updated_model.keras',
]

for model_file in models_to_check:
    if not os.path.exists(model_file):
        print(f"SKIP: {model_file} - not found")
        continue
        
    try:
        print(f"\n{'='*60}")
        print(f"Examining: {model_file}")
        print('='*60)
        
        model = tf.keras.models.load_model(model_file, compile=False)
        
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Total parameters: {model.count_params():,}")
        
        print("\nLayer summary:")
        for i, layer in enumerate(model.layers):
            print(f"{i:2d}. {layer.name:20s} {type(layer).__name__:20s} {str(layer.output_shape):20s}")
            
            # Check for CNN layers
            if 'conv1d' in layer.name.lower():
                if hasattr(layer, 'filters'):
                    print(f"     -> Conv1D: {layer.filters} filters, kernel_size={layer.kernel_size}")
            elif 'dense' in layer.name.lower():
                if hasattr(layer, 'units'):
                    print(f"     -> Dense: {layer.units} units")
        
        # Check if this looks like the original CNN architecture
        conv_layers = [l for l in model.layers if 'conv1d' in l.name.lower()]
        dense_layers = [l for l in model.layers if 'dense' in l.name.lower()]
        
        print(f"\nArchitecture analysis:")
        print(f"Conv1D layers: {len(conv_layers)}")
        print(f"Dense layers: {len(dense_layers)}")
        
        # Expected architecture from JSON:
        # 4 Conv1D layers (32, 64, 128, 256 filters)
        # 3 Dense layers (200, 200, 3 units)
        expected_conv_filters = [32, 64, 128, 256]
        expected_dense_units = [200, 200, 3]
        
        conv_filters = [l.filters for l in conv_layers if hasattr(l, 'filters')]
        dense_units = [l.units for l in dense_layers if hasattr(l, 'units')]
        
        print(f"Conv filters found: {conv_filters}")
        print(f"Dense units found: {dense_units}")
        
        if conv_filters == expected_conv_filters and dense_units == expected_dense_units:
            print("✅ MATCHES expected CNN architecture from JSON!")
        else:
            print("❌ Does not match expected architecture")
            
    except Exception as e:
        print(f"ERROR loading {model_file}: {e}")

print(f"\n{'='*60}")
print("SUMMARY:")
print("Expected architecture from model_pol.json:")
print("- Conv1D layers: 32, 64, 128, 256 filters")  
print("- Dense layers: 200, 200, 3 units")
print("- Input: (400, 3)")
print("- This should be the model that produced 489 picks")
print('='*60)
