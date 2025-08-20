#!/usr/bin/env python3
"""
GPD Model Conversion - Final Fix 
=================================

After examining the model config, I now understand the architecture:
- The lambda layers are for multi-GPU batch splitting (not channel splitting)
- Each lambda takes the full input (400, 3) and processes it
- The sequential model processes each "part" of the batch 
- Results are concatenated along batch dimension

For inference on single GPU, we can simplify this to a single sequential model.
"""

import os
import numpy as np
import h5py
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model


def create_simplified_gpd_model():
    """
    Create simplified single-branch GPD model for inference.
    
    Since the lambda layers were for multi-GPU batch splitting during training,
    we can use a single sequential model for inference that processes 
    the full input (400, 3) directly.
    """
    print("Creating simplified GPD model for inference...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(400, 3), name='input'),
        
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
    ], name='gpd_model_simplified')
    
    return model


def load_weights_direct_mapping(model, hdf5_path):
    """Load weights directly from HDF5 file."""
    print(f"Loading weights from: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        weights_loaded = 0
        
        # Conv layers - use weights as-is (no transposition needed)
        for conv_id in ['1', '2', '3', '4']:
            kernel = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/kernel:0'][:]
            bias = f[f'model_weights/sequential_1/conv1d_{conv_id}_1/bias:0'][:]
            
            conv_layer = model.get_layer(f'conv1d_{conv_id}')
            conv_layer.set_weights([kernel, bias])
            print(f"  ✓ Conv1D {conv_id}: {kernel.shape}")
            weights_loaded += 1
        
        # BatchNorm layers
        for bn_id in ['1', '2', '3', '4', '5', '6']:
            gamma = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/gamma:0'][:]
            beta = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/beta:0'][:]
            moving_mean = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_mean:0'][:]
            moving_var = f[f'model_weights/sequential_1/batch_normalization_{bn_id}_1/moving_variance:0'][:]
            
            bn_layer = model.get_layer(f'batch_normalization_{bn_id}')
            bn_layer.set_weights([gamma, beta, moving_mean, moving_var])
            print(f"  ✓ BatchNorm {bn_id}")
            weights_loaded += 1
        
        # Dense layers  
        for dense_id in ['1', '2', '3']:
            kernel = f[f'model_weights/sequential_1/dense_{dense_id}_1/kernel:0'][:]
            bias = f[f'model_weights/sequential_1/dense_{dense_id}_1/bias:0'][:]
            
            dense_layer = model.get_layer(f'dense_{dense_id}')
            dense_layer.set_weights([kernel, bias])
            print(f"  ✓ Dense {dense_id}: {kernel.shape}")
            weights_loaded += 1
    
    print(f"✓ Loaded weights for {weights_loaded} layers")
    return model


def verify_simplified_model(model):
    """Verify the simplified model works correctly."""
    print("\\n=== Verifying Simplified GPD Model ===")
    
    try:
        test_input = np.random.random((2, 400, 3)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        
        print(f"✓ Model prediction successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {prediction.shape}")
        
        # Check softmax properties
        for i in range(prediction.shape[0]):
            pred_sum = np.sum(prediction[i])
            print(f"  Sample {i}: sum={pred_sum:.6f}, prediction={prediction[i]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def main():
    """Main conversion function."""
    print("=== GPD Model Final Conversion ===")
    print("Creating simplified single-branch model for inference")
    print()
    
    base_dir = os.path.dirname(__file__)
    hdf5_weights_path = os.path.join(base_dir, 'model_pol_best.hdf5')
    
    if not os.path.exists(hdf5_weights_path):
        print(f"✗ Weight file not found: {hdf5_weights_path}")
        return
    
    try:
        # 1. Create simplified model  
        print("1. Creating simplified model architecture...")
        model = create_simplified_gpd_model()
        model.summary()
        
        # 2. Load weights
        print("\\n2. Loading weights...")
        model = load_weights_direct_mapping(model, hdf5_weights_path)
        
        # 3. Verify model
        print("\\n3. Verifying model...")
        if not verify_simplified_model(model):
            print("✗ Model verification failed")
            return
        
        # 4. Save converted models
        print("\\n4. Saving converted models...")
        
        keras_path = os.path.join(base_dir, 'model_pol_final_converted.keras')
        model.save(keras_path)
        print(f"✓ Final model saved: {keras_path}")
        
        h5_path = os.path.join(base_dir, 'model_pol_final_converted.h5')
        model.save(h5_path)
        print(f"✓ Final model saved: {h5_path}")
        
        print("\\n=== Conversion Complete ===")
        print("The simplified model should work correctly for inference.")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
