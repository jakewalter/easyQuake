#!/usr/bin/env python3

import tensorflow as tf
import keras
import json
import os

def test_keras3_json_loading():
    """Test different approaches for loading Keras 2 JSON in Keras 3"""
    
    json_file = 'model_pol.json'
    hdf5_file = 'model_pol_best.hdf5'
    
    if not os.path.exists(json_file) or not os.path.exists(hdf5_file):
        print(f"Missing files: {json_file} or {hdf5_file}")
        return
    
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {keras.__version__}")
    
    # Load and examine the JSON content
    with open(json_file, 'r') as f:
        model_json = f.read()
    
    # Parse JSON to see the structure
    model_config = json.loads(model_json)
    print(f"Model class: {model_config.get('class_name')}")
    print(f"Keras version in JSON: {model_config.get('keras_version')}")
    print(f"Backend: {model_config.get('backend')}")
    
    print("\n" + "="*60)
    print("APPROACH 1: Direct model_from_json (original approach)")
    print("="*60)
    try:
        model = keras.models.model_from_json(model_json, custom_objects={'tf': tf})
        print("✅ SUCCESS: Direct JSON loading worked")
        model.load_weights(hdf5_file)
        print("✅ SUCCESS: Weights loaded")
        return model
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n" + "="*60)
    print("APPROACH 2: Try with legacy tf.keras")
    print("="*60)
    try:
        model = tf.keras.models.model_from_json(model_json, custom_objects={'tf': tf})
        print("✅ SUCCESS: tf.keras JSON loading worked")
        model.load_weights(hdf5_file)
        print("✅ SUCCESS: Weights loaded")
        return model
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n" + "="*60)
    print("APPROACH 3: Add Model to custom_objects")
    print("="*60)
    try:
        custom_objects = {
            'tf': tf,
            'Model': keras.Model,
            'Sequential': keras.Sequential,
            'Lambda': keras.layers.Lambda,
            'Concatenate': keras.layers.Concatenate
        }
        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
        print("✅ SUCCESS: JSON loading with custom Model class worked")
        model.load_weights(hdf5_file)
        print("✅ SUCCESS: Weights loaded")
        return model
    except Exception as e:
        print(f"❌ FAILED: {e}")
        
    print("\n" + "="*60)
    print("APPROACH 4: Try deserialize_keras_object directly")
    print("="*60)
    try:
        from keras.src.saving import serialization_lib
        custom_objects = {'tf': tf, 'Model': keras.Model}
        model = serialization_lib.deserialize_keras_object(
            model_config, 
            custom_objects=custom_objects,
            safe_mode=False
        )
        print("✅ SUCCESS: Direct deserialization worked")
        model.load_weights(hdf5_file)
        print("✅ SUCCESS: Weights loaded")
        return model
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    print("\n❌ All approaches failed")
    return None

if __name__ == "__main__":
    model = test_keras3_json_loading()
