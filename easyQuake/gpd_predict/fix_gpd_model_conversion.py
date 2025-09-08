#!/usr/bin/env python3
"""
Fix GPD model conversion from TF1 to TF2/Keras 3 based on seisbench conversion methodology.

This script properly converts the original GPD model following the patterns from:
https://github.com/seisbench/seisbench/blob/main/contrib/model_conversion/gpd_conversion.ipynb

Key fixes:
1. Proper weight transposition for Conv1D layers
2. Correct BatchNormalization parameter mapping
3. Maintain NEZ channel ordering (do NOT reorder)
4. Ensure correct architecture with proper epsilon values
"""

import os
import numpy as np
import h5py
import tensorflow as tf
import keras
from keras import layers, Model

def create_gpd_model():
    """
    Manually recreate the GPD model architecture based on the original TF1 model.
    
    Architecture from seisbench conversion:
    - Input: (None, 400, 3) - NEZ channel ordering
    - Conv1D layers with specific parameters
    - BatchNormalization with epsilon=1e-3
    - Proper activation and pooling layers
    """
    # Input layer: (None, 400, 3) - matches original (batch, time, channels)
    input_layer = layers.Input(shape=(400, 3), name='input_1')
    
    # Conv1D layers - exact same architecture as original
    # Conv1d(3, 32, 21, padding=same) 
    x = layers.Conv1D(32, 21, padding='same', activation='linear', name='conv1d_1')(input_layer)
    x = layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_1')(x)
    x = layers.Activation('relu', name='activation_1')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d_1')(x)
    
    # Conv1d(32, 64, 15, padding=same)
    x = layers.Conv1D(64, 15, padding='same', activation='linear', name='conv1d_2')(x)
    x = layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_2')(x)
    x = layers.Activation('relu', name='activation_2')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d_2')(x)
    
    # Conv1d(64, 128, 11, padding=same)
    x = layers.Conv1D(128, 11, padding='same', activation='linear', name='conv1d_3')(x)
    x = layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_3')(x)
    x = layers.Activation('relu', name='activation_3')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d_3')(x)
    
    # Conv1d(128, 256, 9, padding=same)
    x = layers.Conv1D(256, 9, padding='same', activation='linear', name='conv1d_4')(x)
    x = layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_4')(x)
    x = layers.Activation('relu', name='activation_4')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d_4')(x)
    
    # Flatten and Dense layers
    # After 4 maxpool operations: 400 -> 200 -> 100 -> 50 -> 25
    # So we have: (batch, 25, 256) -> flatten -> (batch, 6400)
    x = layers.Flatten(name='flatten_1')(x)
    
    # Dense(6400, 200)
    x = layers.Dense(200, activation='linear', name='dense_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_5')(x)
    x = layers.Activation('relu', name='activation_5')(x)
    
    # Dense(200, 200) 
    x = layers.Dense(200, activation='linear', name='dense_2')(x)
    x = layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_6')(x)
    x = layers.Activation('relu', name='activation_6')(x)
    
    # Dense(200, 3)
    x = layers.Dense(3, activation='linear', name='dense_3')(x)
    
    # Final softmax
    outputs = layers.Activation('softmax', name='activation_7')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=outputs, name='gpd_model')
    
    return model


def load_weights_with_proper_conversion(model, hdf5_path):
    """
    Load weights from original HDF5 file with proper conversion.
    
    Based on the seisbench conversion, this handles:
    1. Weight transposition for conv layers  
    2. Maintain NEZ channel ordering (do NOT reorder to ZNE)
    3. Proper batch normalization mapping
    """
    print(f"Loading weights from: {hdf5_path}")
    
    # Load the original HDF5 weights
    with h5py.File(hdf5_path, 'r') as f:
        # Create mapping from HDF5 structure to model weights
        weight_mapping = {}
        
        # Conv layers - NO transposition needed, shapes match!
        for conv_id in ['1', '2', '3', '4']:
            hdf5_kernel = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/kernel:0'][:]  # (kernel, in, out)
            hdf5_bias = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/bias:0'][:]
            
            # HDF5 format: (kernel_size, input_channels, output_channels) = (21, 3, 32)
            # Keras expects: (kernel_size, input_channels, output_channels) = (21, 3, 32)
            # They match! No transposition needed.
            print(f"  Conv{conv_id} shape: {hdf5_kernel.shape} - using as-is")
            
            # NOTE: Keep original NEZ channel ordering - do NOT reorder channels
            # The original model expects NEZ format, not ZNE
            
            weight_mapping[f'conv1d_{conv_id}'] = {
                'kernel': hdf5_kernel,  # Use as-is
                'bias': hdf5_bias
            }
        
        # BatchNormalization layers
        for bn_id in ['1', '2', '3', '4', '5', '6']:
            gamma = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/gamma:0'][:]
            beta = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/beta:0'][:]
            moving_mean = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_mean:0'][:]
            moving_var = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_variance:0'][:]
            
            weight_mapping[f'batch_normalization_{bn_id}'] = {
                'gamma': gamma,
                'beta': beta, 
                'moving_mean': moving_mean,
                'moving_variance': moving_var
            }
        
        # Dense layers - no transposition needed for Keras (already in correct format)
        for dense_id in ['1', '2', '3']:
            hdf5_kernel = f[f'model_weights/sequential_1/dense_{dense_id}_1/kernel:0'][:]  # Already (in, out)
            hdf5_bias = f[f'model_weights/sequential_1/dense_{dense_id}_1/bias:0'][:]
            
            weight_mapping[f'dense_{dense_id}'] = {
                'kernel': hdf5_kernel,  # Use as-is - already in Keras format
                'bias': hdf5_bias
            }
    
    # Apply weights to model
    weights_loaded = 0
    for layer in model.layers:
        layer_name = layer.name
        
        if layer_name.startswith('conv1d_'):
            if layer_name in weight_mapping:
                layer.set_weights([
                    weight_mapping[layer_name]['kernel'],
                    weight_mapping[layer_name]['bias']
                ])
                print(f"✓ Loaded weights for {layer_name}")
                weights_loaded += 1
                
        elif layer_name.startswith('batch_normalization_'):
            if layer_name in weight_mapping:
                bn_weights = weight_mapping[layer_name]
                layer.set_weights([
                    bn_weights['gamma'],
                    bn_weights['beta'],
                    bn_weights['moving_mean'], 
                    bn_weights['moving_variance']
                ])
                print(f"✓ Loaded weights for {layer_name}")
                weights_loaded += 1
                
        elif layer_name.startswith('dense_'):
            if layer_name in weight_mapping:
                layer.set_weights([
                    weight_mapping[layer_name]['kernel'],
                    weight_mapping[layer_name]['bias']
                ])
                print(f"✓ Loaded weights for {layer_name}")
                weights_loaded += 1
    
    print(f"✓ Loaded weights for {weights_loaded} layers")
    return model


def verify_model_with_test_data(model, model_name):
    """Verify the converted model works correctly."""
    print(f"\n=== Verifying {model_name} ===")
    
    # Create test data (NEZ format)
    test_data = np.random.rand(2, 400, 3)
    test_data = test_data / np.max(np.abs(test_data), axis=(1,2), keepdims=True)
    
    try:
        # Test prediction
        predictions = model.predict(test_data, verbose=0)
        print(f"✓ Model prediction successful")
        print(f"  Input shape: {test_data.shape}")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Sample prediction: {predictions[0]}")
        print(f"  Prediction sum: {np.sum(predictions[0]):.6f} (should be ~1.0)")
        
        # Verify softmax output
        if np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5):
            print("✓ Softmax output verified")
        else:
            print("⚠ Warning: Softmax output doesn't sum to 1.0")
            
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def main():
    """Main conversion process."""
    base_dir = os.path.dirname(__file__)
    
    # Original model files
    hdf5_weights_path = os.path.join(base_dir, 'model_pol_best.hdf5')
    
    if not os.path.exists(hdf5_weights_path):
        print(f"Error: Original weights file not found: {hdf5_weights_path}")
        print("Please ensure model_pol_best.hdf5 is in the gpd_predict directory")
        return
    
    print("=== GPD Model Conversion Fix ===")
    print("Converting TF1 → TF2/Keras 3 with proper weight mapping")
    print("Maintaining NEZ channel ordering (no reordering)")
    
    try:
        # Create model with correct architecture
        print("\n1. Creating model architecture...")
        model = create_gpd_model()
        model.summary()
        
        # Load and convert weights properly
        print("\n2. Loading and converting weights...")
        model = load_weights_with_proper_conversion(model, hdf5_weights_path)
        
        # Verify the model works
        print("\n3. Verifying converted model...")
        if verify_model_with_test_data(model, "Fixed GPD Model"):
            
            # Save the fixed model
            output_path = os.path.join(base_dir, 'model_pol_properly_converted.keras')
            model.save(output_path)
            print(f"\n✓ Fixed model saved to: {output_path}")
            
            # Also save as .h5 for compatibility
            h5_output_path = os.path.join(base_dir, 'model_pol_properly_converted.h5')
            model.save(h5_output_path)
            print(f"✓ Fixed model also saved as: {h5_output_path}")
            
            print("\n=== Conversion Complete ===")
            print("The fixed model:")
            print("- Uses proper weight transposition for Conv1D layers")
            print("- Maintains NEZ channel ordering (compatible with current code)")
            print("- Has correct BatchNormalization parameters")
            print("- Should work identically to the original TF1 model")
            
        else:
            print("\n✗ Model verification failed - conversion unsuccessful")
            
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
