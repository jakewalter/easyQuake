#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("Testing original model_pol.json + model_pol_best.hdf5...")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Check files exist
json_file = 'model_pol.json'
hdf5_file = 'model_pol_best.hdf5'

if not os.path.exists(json_file):
    print(f"ERROR: {json_file} not found")
    exit(1)
    
if not os.path.exists(hdf5_file):
    print(f"ERROR: {hdf5_file} not found")
    exit(1)

print(f"✓ Found {json_file} ({os.path.getsize(json_file)} bytes)")
print(f"✓ Found {hdf5_file} ({os.path.getsize(hdf5_file)} bytes)")

try:
    # Load JSON architecture
    with open(json_file, 'r') as f:
        model_json = f.read()
    
    print("\n" + "="*50)
    print("LOADING MODEL FROM JSON + HDF5")
    print("="*50)
    
    # Try to load model from JSON
    model = keras.models.model_from_json(model_json)
    print("✓ Successfully loaded model architecture from JSON")
    
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Load weights
    model.load_weights(hdf5_file)
    print(f"✓ Successfully loaded weights from {hdf5_file}")
    
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
    
    # Test with dummy data to see if it produces reasonable outputs
    print("\n" + "="*50)
    print("TESTING MODEL PREDICTIONS")
    print("="*50)
    
    # Create test input matching expected shape (batch_size, 400, 3)
    test_input = np.random.randn(1, 400, 3).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")
    
    predictions = model.predict(test_input, verbose=0)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"Sample predictions: {predictions[0][:10]}")  # First 10 values
    
    # Check if predictions look reasonable (should be probabilities)
    if predictions.min() >= 0 and predictions.max() <= 1:
        print("✓ Predictions appear to be valid probabilities")
    else:
        print("⚠ Predictions outside [0,1] range - may need sigmoid activation")
    
    # Check for high-confidence predictions (> 0.994 threshold)
    high_conf = np.sum(predictions > 0.994)
    print(f"Number of predictions > 0.994 threshold: {high_conf}")
    
    print("\n" + "="*50)
    print("MODEL LOADING SUCCESS!")
    print("Ready to test with real seismic data")
    print("="*50)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
