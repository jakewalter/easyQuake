#!/usr/bin/env python3

import tensorflow as tf
import keras
import json
import os

def test_unsafe_deserialization():
    """Test with safe_mode=False to allow lambda deserialization"""
    
    json_file = 'model_pol.json'
    hdf5_file = 'model_pol_best.hdf5'
    
    if not os.path.exists(json_file) or not os.path.exists(hdf5_file):
        print(f"Missing files: {json_file} or {hdf5_file}")
        return
    
    print("Testing unsafe deserialization approach...")
    print("This should work if we can bypass the safety restrictions")
    
    with open(json_file, 'r') as f:
        model_json = f.read()
    
    # Parse JSON to modify it for Keras 3 compatibility
    model_config = json.loads(model_json)
    
    print("\n" + "="*60)
    print("APPROACH 1: Enable unsafe deserialization globally")
    print("="*60)
    try:
        # Enable unsafe deserialization globally
        keras.config.enable_unsafe_deserialization()
        
        custom_objects = {
            'tf': tf,
            'Model': keras.Model,
            'Sequential': keras.Sequential,
            'Lambda': keras.layers.Lambda,
            'Concatenate': keras.layers.Concatenate
        }
        
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
        print("✅ SUCCESS: JSON loading with unsafe deserialization worked")
        model.load_weights(hdf5_file)
        print("✅ SUCCESS: Weights loaded")
        
        # Test the model
        import numpy as np
        test_input = np.random.randn(1, 400, 3).astype(np.float32)
        predictions = model.predict(test_input, verbose=0)
        print(f"Model output shape: {predictions.shape}")
        print(f"Model output range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        return model
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n" + "="*60)
    print("APPROACH 2: Use safe_mode=False in from_config")
    print("="*60)
    try:
        # Try with safe_mode=False in the deserialization
        from keras.src.saving import serialization_lib
        
        custom_objects = {
            'tf': tf,
            'Model': keras.Model,
            'Sequential': keras.Sequential,
            'Lambda': keras.layers.Lambda,
            'Concatenate': keras.layers.Concatenate
        }
        
        model = serialization_lib.deserialize_keras_object(
            model_config,
            custom_objects=custom_objects,
            safe_mode=False  # This should allow lambda deserialization
        )
        print("✅ SUCCESS: Direct deserialization with safe_mode=False worked")
        model.load_weights(hdf5_file)
        print("✅ SUCCESS: Weights loaded")
        
        # Test the model
        import numpy as np
        test_input = np.random.randn(1, 400, 3).astype(np.float32)
        predictions = model.predict(test_input, verbose=0)
        print(f"Model output shape: {predictions.shape}")
        print(f"Model output range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        return model
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n❌ Both unsafe approaches failed")
    return None

if __name__ == "__main__":
    model = test_unsafe_deserialization()
