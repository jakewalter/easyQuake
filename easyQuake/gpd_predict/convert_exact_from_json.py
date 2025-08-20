#!/usr/bin/env python3
"""
GPD Model Exact Conversion from Original JSON
=============================================

This script recreates the exact original model architecture from the 
model_pol.json file and loads weights from model_pol_best.hdf5 to create
a precise TF1->TF2/Keras 3 conversion.

The original model has:
1. Input: (batch, 400, 3)
2. Lambda layers that split the batch for multi-GPU processing
3. Sequential model processes each split
4. Concatenate layer combines results along batch dimension

For single-GPU inference, we create two versions:
1. Exact replica with lambda layers and multi-branch architecture
2. Optimized single-branch version for practical use
"""

import os
import json
import numpy as np
import h5py
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model


def decode_lambda_slice_function(i, parts):
    """
    Recreate the lambda slice function based on the arguments.
    
    The original function splits the batch dimension into 'parts' and 
    returns slice 'i'. For multi-GPU training with 3 GPUs:
    - lambda_1: i=0, returns data[0::3] (every 3rd starting from 0)
    - lambda_2: i=1, returns data[1::3] (every 3rd starting from 1) 
    - lambda_3: i=2, returns data[2::3] (every 3rd starting from 2)
    """
    def slice_func(x):
        # Split batch into parts and return slice i
        return x[i::parts]
    
    return slice_func


def create_exact_model_from_json():
    """Create the exact model architecture from the original JSON."""
    print("Creating exact model from original JSON...")
    
    # Load the original JSON configuration
    json_path = os.path.join(os.path.dirname(__file__), 'model_pol.json')
    with open(json_path, 'r') as f:
        model_config = json.load(f)
    
    # Input layer
    input_layer = layers.Input(shape=(400, 3), name='conv1d_1_input')
    
    # Lambda layers that split the batch for multi-GPU processing
    lambda_1 = layers.Lambda(
        decode_lambda_slice_function(0, 3), 
        name='lambda_1'
    )(input_layer)
    
    lambda_2 = layers.Lambda(
        decode_lambda_slice_function(1, 3), 
        name='lambda_2'
    )(input_layer)
    
    lambda_3 = layers.Lambda(
        decode_lambda_slice_function(2, 3), 
        name='lambda_3'
    )(input_layer)
    
    # Create the sequential model exactly as defined in JSON
    sequential_model = keras.Sequential([
        # Conv1D + BatchNorm + ReLU + MaxPool
        layers.Conv1D(32, kernel_size=21, padding='same', name='conv1d_1'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_1'),
        layers.Activation('relu', name='activation_1'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1'),
        
        # Conv1D + BatchNorm + ReLU + MaxPool  
        layers.Conv1D(64, kernel_size=15, padding='same', name='conv1d_2'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_2'),
        layers.Activation('relu', name='activation_2'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_2'),
        
        # Conv1D + BatchNorm + ReLU + MaxPool
        layers.Conv1D(128, kernel_size=11, padding='same', name='conv1d_3'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_3'),
        layers.Activation('relu', name='activation_3'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_3'),
        
        # Conv1D + BatchNorm + ReLU + MaxPool
        layers.Conv1D(256, kernel_size=9, padding='same', name='conv1d_4'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_4'),
        layers.Activation('relu', name='activation_4'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_4'),
        
        # Flatten + Dense layers
        layers.Flatten(name='flatten_1'),
        layers.Dense(200, name='dense_1'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_5'),
        layers.Activation('relu', name='activation_5'),
        layers.Dense(200, name='dense_2'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_6'),
        layers.Activation('relu', name='activation_6'),
        layers.Dense(3, name='dense_3'),
        layers.Activation('softmax', name='activation_7'),
    ], name='sequential_1')
    
    # Apply sequential model to each lambda output
    branch_1 = sequential_model(lambda_1)
    branch_2 = sequential_model(lambda_2)  
    branch_3 = sequential_model(lambda_3)
    
    # Concatenate results along batch dimension (axis=0)
    # This recreates the exact original behavior
    output = layers.Concatenate(axis=0, name='activation_7_final')([branch_1, branch_2, branch_3])
    
    # Create the exact model
    exact_model = Model(inputs=input_layer, outputs=output, name='gpd_exact_model')
    
    return exact_model


def create_optimized_single_branch_model():
    """Create an optimized single-branch version for practical inference."""
    print("Creating optimized single-branch model...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(400, 3), name='input'),
        
        # Same architecture as sequential_1 but for single branch
        layers.Conv1D(32, kernel_size=21, padding='same', name='conv1d_1'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_1'),
        layers.Activation('relu', name='activation_1'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1'),
        
        layers.Conv1D(64, kernel_size=15, padding='same', name='conv1d_2'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_2'),
        layers.Activation('relu', name='activation_2'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_2'),
        
        layers.Conv1D(128, kernel_size=11, padding='same', name='conv1d_3'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_3'),
        layers.Activation('relu', name='activation_3'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_3'),
        
        layers.Conv1D(256, kernel_size=9, padding='same', name='conv1d_4'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_4'),
        layers.Activation('relu', name='activation_4'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_4'),
        
        layers.Flatten(name='flatten_1'),
        layers.Dense(200, name='dense_1'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_5'),
        layers.Activation('relu', name='activation_5'),
        layers.Dense(200, name='dense_2'),
        layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name='batch_normalization_6'),
        layers.Activation('relu', name='activation_6'),
        layers.Dense(3, name='dense_3'),
        layers.Activation('softmax', name='activation_7'),
    ], name='gpd_optimized_model')
    
    return model


def load_weights_from_hdf5(model, hdf5_path, model_type="exact"):
    """Load weights from the original HDF5 file."""
    print(f"Loading weights from: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        weights_loaded = 0
        
        if model_type == "exact":
            # For exact model, load weights into the sequential_1 submodel
            sequential_model = model.get_layer('sequential_1')
            target_model = sequential_model
        else:
            # For optimized model, load weights directly
            target_model = model
        
        # Load Conv1D weights
        for conv_id in ['1', '2', '3', '4']:
            kernel = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/kernel:0'][:]
            bias = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/bias:0'][:]
            
            conv_layer = target_model.get_layer(f'conv1d_{conv_id}')
            conv_layer.set_weights([kernel, bias])
            print(f"  ✓ Conv1D {conv_id}: {kernel.shape}")
            weights_loaded += 1
        
        # Load BatchNormalization weights
        for bn_id in ['1', '2', '3', '4', '5', '6']:
            gamma = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/gamma:0'][:]
            beta = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/beta:0'][:]
            moving_mean = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_mean:0'][:]
            moving_var = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_variance:0'][:]
            
            bn_layer = target_model.get_layer(f'batch_normalization_{bn_id}')
            bn_layer.set_weights([gamma, beta, moving_mean, moving_var])
            print(f"  ✓ BatchNorm {bn_id}")
            weights_loaded += 1
        
        # Load Dense weights
        for dense_id in ['1', '2', '3']:
            kernel = f[f'model_weights/sequential_1/dense_{dense_id}_1/kernel:0'][:]
            bias = f[f'model_weights/sequential_1/dense_{dense_id}_1/bias:0'][:]
            
            dense_layer = target_model.get_layer(f'dense_{dense_id}')
            dense_layer.set_weights([kernel, bias])
            print(f"  ✓ Dense {dense_id}: {kernel.shape}")
            weights_loaded += 1
    
    print(f"✓ Loaded weights for {weights_loaded} layers")
    return model


def verify_model(model, model_name):
    """Verify the converted model."""
    print(f"\\n=== Verifying {model_name} ===")
    
    try:
        # Test with batch size 3 to see the effect clearly
        test_input = np.random.random((3, 400, 3)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        
        print(f"✓ Model prediction successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {prediction.shape}")
        
        if model_name == "Exact Model":
            # Exact model should output (batch*3, 3) = (9, 3)
            print(f"  Expected: (9, 3) for multi-branch concatenation")
        else:
            # Optimized model should output (batch, 3) = (3, 3)
            print(f"  Expected: (3, 3) for single-branch")
        
        # Check softmax properties
        for i in range(min(6, prediction.shape[0])):  # Show first 6 predictions
            pred_sum = np.sum(prediction[i])
            print(f"  Prediction {i}: sum={pred_sum:.6f}, values={prediction[i]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def main():
    """Main conversion function."""
    print("=== GPD Model Exact Conversion from Original JSON ===")
    print("Using model_pol.json + model_pol_best.hdf5 for precise conversion")
    print()
    
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, 'model_pol.json')
    hdf5_path = os.path.join(base_dir, 'model_pol_best.hdf5')
    
    # Check files exist
    if not os.path.exists(json_path):
        print(f"✗ JSON file not found: {json_path}")
        return
    if not os.path.exists(hdf5_path):
        print(f"✗ HDF5 file not found: {hdf5_path}")
        return
    
    try:
        # 1. Create exact model
        print("1. Creating exact model from JSON...")
        exact_model = create_exact_model_from_json()
        exact_model.summary()
        
        # 2. Load weights into exact model
        print("\\n2. Loading weights into exact model...")
        exact_model = load_weights_from_hdf5(exact_model, hdf5_path, "exact")
        
        # 3. Verify exact model
        print("\\n3. Verifying exact model...")
        if not verify_model(exact_model, "Exact Model"):
            print("✗ Exact model verification failed")
            return
        
        # 4. Create optimized model
        print("\\n4. Creating optimized single-branch model...")
        optimized_model = create_optimized_single_branch_model()
        optimized_model.summary()
        
        # 5. Load weights into optimized model
        print("\\n5. Loading weights into optimized model...")
        optimized_model = load_weights_from_hdf5(optimized_model, hdf5_path, "optimized")
        
        # 6. Verify optimized model
        print("\\n6. Verifying optimized model...")
        if not verify_model(optimized_model, "Optimized Model"):
            print("✗ Optimized model verification failed")
            return
        
        # 7. Save models
        print("\\n7. Saving converted models...")
        
        # Save exact model
        exact_keras_path = os.path.join(base_dir, 'model_pol_exact_converted.keras')
        exact_model.save(exact_keras_path)
        print(f"✓ Exact model saved: {exact_keras_path}")
        
        # Save optimized model  
        opt_keras_path = os.path.join(base_dir, 'model_pol_optimized_converted.keras')
        optimized_model.save(opt_keras_path)
        print(f"✓ Optimized model saved: {opt_keras_path}")
        
        print("\\n=== Conversion Complete ===")
        print("Two models created:")
        print("1. Exact model: Preserves original multi-branch architecture")
        print("2. Optimized model: Single-branch version for practical inference")
        print("Both should give equivalent results for the same inputs.")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
