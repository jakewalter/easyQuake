#!/usr/bin/env python3

import tensorflow as tf
import keras
import numpy as np
import os

def create_lambda_functions():
    """Create custom lambda functions compatible with Keras 3"""
    
    def get_slice(data, i, parts):
        """Custom slice function to replace the serialized lambda"""
        # This replicates the logic from the original lambda functions
        batch_size = tf.shape(data)[0]
        input_shape = tf.shape(data)
        step = input_shape[1] // parts  # 400 // 3 = 133
        size = step
        stride = step
        start = i * step
        
        # Use tf.slice instead of complex slicing logic
        if i < parts - 1:
            sliced = tf.slice(data, [0, start, 0], [batch_size, size, input_shape[2]])
        else:
            # Last slice gets remaining data
            sliced = tf.slice(data, [0, start, 0], [batch_size, input_shape[1] - start, input_shape[2]])
        
        return sliced
    
    # Create the three lambda functions
    lambda_1 = lambda x: get_slice(x, 0, 3)
    lambda_2 = lambda x: get_slice(x, 1, 3) 
    lambda_3 = lambda x: get_slice(x, 2, 3)
    
    return {
        'lambda_1': lambda_1,
        'lambda_2': lambda_2, 
        'lambda_3': lambda_3,
        'get_slice': get_slice,
        'tf': tf  # Also include tf itself
    }

def test_custom_loading():
    """Test loading the model with custom lambda functions"""
    
    # Define paths
    json_file = 'model_pol.json'
    hdf5_file = 'model_pol_best.hdf5'
    
    if not os.path.exists(json_file) or not os.path.exists(hdf5_file):
        print(f"Missing files: {json_file} or {hdf5_file}")
        return None
    
    try:
        print("Attempting to load with custom lambda functions...")
        
        # Create custom objects
        custom_objects = create_lambda_functions()
        
        # Load JSON
        with open(json_file, 'r') as f:
            model_json = f.read()
        
        # Try to load with custom objects
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
        model.load_weights(hdf5_file)
        
        print("✅ Successfully loaded with custom lambda functions!")
        
        # Test the model
        test_input = np.random.randn(1, 400, 3).astype(np.float32)
        predictions = model.predict(test_input, verbose=0)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        return model
        
    except Exception as e:
        print(f"❌ Custom loading failed: {e}")
        return None

if __name__ == "__main__":
    model = test_custom_loading()
