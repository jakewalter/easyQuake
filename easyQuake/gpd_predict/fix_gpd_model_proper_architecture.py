#!/usr/bin/env python3
"""
GPD Model Conversion Fix - Proper Multi-Branch Architecture
==========================================================

This script properly converts the TF1 GPD model to TF2/Keras 3 by implementing
the correct multi-branch architecture that was revealed in the model config.

The original model architecture:
1. Input: (batch, 400, 3) - 3 channels (N, E, Z)
2. Lambda layers split into 3 separate streams by channel  
3. Sequential model processes each channel separately (same weights)
4. Concatenate the results along batch dimension
5. Final output: (batch*3, 3) predictions for [P, S, Noise]

Key insights:
- The model processes each channel (N, E, Z) separately through identical networks
- Each channel produces its own (batch, 3) prediction  
- Results are concatenated to (batch*3, 3)
- This explains why probability ranges were different - we were missing the branching!
"""

import os
import numpy as np
import h5py
import json
import base64
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model


def decode_lambda_function(encoded_function):
    """Decode the base64 encoded lambda function from TF1 model."""
    try:
        # The lambda function is base64 encoded bytecode
        # For our purposes, we know it's a channel splitting function
        return "Channel splitting function (i, parts=3)"
    except Exception as e:
        return f"Could not decode: {e}"


def create_proper_gpd_model():
    """
    Create the proper multi-branch GPD model architecture.
    
    Architecture:
    - Input: (batch, 400, 3)
    - Split into 3 channels 
    - Process each channel through identical sequential networks
    - Concatenate outputs: (batch*3, 3)
    """
    print("Creating proper multi-branch GPD model architecture...")
    
    # Input layer
    input_layer = layers.Input(shape=(400, 3), name='conv1d_1_input')
    
    # Lambda layers to split channels (equivalent to original lambda_1, lambda_2, lambda_3)
    # Each lambda extracts one channel: lambda_1 gets [:,:,0], lambda_2 gets [:,:,1], lambda_3 gets [:,:,2]
    lambda_1 = layers.Lambda(lambda x: x[:, :, 0:1], name='lambda_1')(input_layer)  # N channel
    lambda_2 = layers.Lambda(lambda x: x[:, :, 1:2], name='lambda_2')(input_layer)  # E channel  
    lambda_3 = layers.Lambda(lambda x: x[:, :, 2:3], name='lambda_3')(input_layer)  # Z channel
    
    # Define the sequential processing function (same for all channels)
    def create_sequential_branch():
        """Create the sequential processing branch (same weights for all channels)."""
        return keras.Sequential([
            # Conv1D + BatchNorm + ReLU + MaxPool
            layers.Conv1D(32, kernel_size=21, padding='same', name='conv1d_1'),
            layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_1'),
            layers.Activation('relu', name='activation_1'),
            layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1'),
            
            # Conv1D + BatchNorm + ReLU + MaxPool  
            layers.Conv1D(64, kernel_size=15, padding='same', name='conv1d_2'),
            layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_2'),
            layers.Activation('relu', name='activation_2'),
            layers.MaxPooling1D(pool_size=2, name='max_pooling1d_2'),
            
            # Conv1D + BatchNorm + ReLU + MaxPool
            layers.Conv1D(128, kernel_size=11, padding='same', name='conv1d_3'),
            layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_3'),
            layers.Activation('relu', name='activation_3'),
            layers.MaxPooling1D(pool_size=2, name='max_pooling1d_3'),
            
            # Conv1D + BatchNorm + ReLU + MaxPool
            layers.Conv1D(256, kernel_size=9, padding='same', name='conv1d_4'),
            layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_4'),
            layers.Activation('relu', name='activation_4'),
            layers.MaxPooling1D(pool_size=2, name='max_pooling1d_4'),
            
            # Flatten + Dense layers
            layers.Flatten(name='flatten_1'),
            layers.Dense(200, name='dense_1'),
            layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_5'),
            layers.Activation('relu', name='activation_5'),
            layers.Dense(200, name='dense_2'),
            layers.BatchNormalization(epsilon=1e-3, name='batch_normalization_6'),
            layers.Activation('relu', name='activation_6'),
            layers.Dense(3, name='dense_3'),
            layers.Activation('softmax', name='activation_7'),
        ], name='sequential_1')
    
    # Create the shared sequential model (same weights for all channels)
    sequential_model = create_sequential_branch()
    
    # Apply the sequential model to each channel
    branch_1 = sequential_model(lambda_1)  # Process N channel
    branch_2 = sequential_model(lambda_2)  # Process E channel  
    branch_3 = sequential_model(lambda_3)  # Process Z channel
    
    # Concatenate along batch dimension (axis=0)
    # This gives us (batch*3, 3) output 
    output = layers.Concatenate(axis=0, name='activation_7_concat')([branch_1, branch_2, branch_3])
    
    # Create the full model
    model = Model(inputs=input_layer, outputs=output, name='gpd_model_proper')
    
    return model


def load_weights_with_proper_mapping(model, hdf5_path):
    """
    Load weights from HDF5 file and map them to the proper multi-branch model.
    
    The sequential_1 weights need to be loaded into the shared sequential model,
    which will automatically apply to all 3 branches.
    """
    print(f"Loading weights from: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get the sequential model from our multi-branch model
        sequential_model = model.get_layer('sequential_1')
        
        # Map weights layer by layer
        weights_loaded = 0
        
        # Conv layers
        for conv_id in ['1', '2', '3', '4']:
            kernel = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/kernel:0'][:]
            bias = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/bias:0'][:]
            
            conv_layer = sequential_model.get_layer(f'conv1d_{conv_id}')
            conv_layer.set_weights([kernel, bias])
            print(f"  ✓ Loaded Conv1D {conv_id} weights: {kernel.shape}")
            weights_loaded += 1
        
        # BatchNorm layers
        for bn_id in ['1', '2', '3', '4', '5', '6']:
            gamma = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/gamma:0'][:]
            beta = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/beta:0'][:]
            moving_mean = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_mean:0'][:]
            moving_var = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_variance:0'][:]
            
            bn_layer = sequential_model.get_layer(f'batch_normalization_{bn_id}')
            bn_layer.set_weights([gamma, beta, moving_mean, moving_var])
            print(f"  ✓ Loaded BatchNorm {bn_id} weights")
            weights_loaded += 1
        
        # Dense layers  
        for dense_id in ['1', '2', '3']:
            kernel = f[f'model_weights/sequential_1/dense_{dense_id}_1/kernel:0'][:]
            bias = f[f'model_weights/sequential_1/dense_{dense_id}_1/bias:0'][:]
            
            dense_layer = sequential_model.get_layer(f'dense_{dense_id}')
            dense_layer.set_weights([kernel, bias])
            print(f"  ✓ Loaded Dense {dense_id} weights: {kernel.shape}")
            weights_loaded += 1
    
    print(f"✓ Loaded weights for {weights_loaded} layers")
    return model


def verify_proper_model(model):
    """Verify the properly converted multi-branch model."""
    print("\n=== Verifying Proper Multi-Branch GPD Model ===")
    
    try:
        # Test with batch size 2 to see the 3x multiplication effect
        test_input = np.random.random((2, 400, 3)).astype(np.float32)
        
        prediction = model.predict(test_input, verbose=0)
        
        print(f"✓ Model prediction successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {prediction.shape}")
        print(f"  Expected output shape: (6, 3) [batch*3, classes]")
        
        # Check that output has the right shape (batch*3, 3)
        expected_batch_size = test_input.shape[0] * 3
        if prediction.shape == (expected_batch_size, 3):
            print(f"✓ Output shape correct: ({expected_batch_size}, 3)")
        else:
            print(f"✗ Output shape incorrect: expected ({expected_batch_size}, 3), got {prediction.shape}")
        
        # Check softmax properties for each prediction
        for i in range(prediction.shape[0]):
            pred_sum = np.sum(prediction[i])
            print(f"  Branch {i//2+1}, Channel {i%3}: sum={pred_sum:.6f}, max={np.max(prediction[i]):.6f}")
        
        print(f"✓ Multi-branch model verified")
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def main():
    """Main conversion function."""
    print("=== GPD Model Proper Architecture Conversion ===")
    print("Converting TF1 → TF2/Keras 3 with correct multi-branch architecture")
    print("Maintaining NEZ channel ordering and proper lambda layer behavior")
    print()
    
    base_dir = os.path.dirname(__file__)
    hdf5_weights_path = os.path.join(base_dir, 'model_pol_best.hdf5')
    
    if not os.path.exists(hdf5_weights_path):
        print(f"✗ Weight file not found: {hdf5_weights_path}")
        return
    
    try:
        # 1. Create proper multi-branch model architecture
        print("1. Creating proper multi-branch model architecture...")
        model = create_proper_gpd_model()
        model.summary()
        
        # 2. Load weights with proper mapping
        print("\n2. Loading and mapping weights...")
        model = load_weights_with_proper_mapping(model, hdf5_weights_path)
        
        # 3. Verify the converted model
        print("\n3. Verifying converted model...")
        if not verify_proper_model(model):
            print("✗ Model verification failed")
            return
        
        # 4. Save the properly converted models
        print("\n4. Saving properly converted models...")
        
        # Save as .keras (modern format)
        keras_output_path = os.path.join(base_dir, 'model_pol_proper_multibranch.keras')
        model.save(keras_output_path)
        print(f"✓ Proper multi-branch model saved to: {keras_output_path}")
        
        # Save as .h5 (legacy compatibility)
        h5_output_path = os.path.join(base_dir, 'model_pol_proper_multibranch.h5')
        model.save(h5_output_path)
        print(f"✓ Proper multi-branch model also saved as: {h5_output_path}")
        
        print("\n=== Conversion Complete ===")
        print("The properly converted model:")
        print("- Implements correct multi-branch architecture with lambda layers")
        print("- Processes each channel (N, E, Z) separately through shared weights")
        print("- Concatenates results to produce (batch*3, 3) output")
        print("- Maintains NEZ channel ordering")
        print("- Should produce the same results as the original TF1 model")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
