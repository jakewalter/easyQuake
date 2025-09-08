#!/usr/bin/env python3
"""Test script to compare model probability outputs between original and converted models."""

import numpy as np
import os
import sys

def test_original_model():
    """Test the original JSON + HDF5 model from development branch."""
    print("Testing original model (development branch)...")
    
    try:
        # Import the original model loading approach
        sys.path.insert(0, '/home/jwalter/easyQuake/easyQuake')
        from keras.models import model_from_json
        import tensorflow as tf
        
        # Load original model
        model_dir = '/home/jwalter/easyQuake/easyQuake/gpd_predict'
        json_file = open(os.path.join(model_dir, 'model_pol.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, custom_objects={'tf': tf})
        model.load_weights(os.path.join(model_dir, "model_pol_best.hdf5"))
        
        # Create test data (same format as GPD expects: 3-channel, 400 samples)
        test_data = np.random.randn(10, 400, 3).astype('float32')
        
        # Get predictions
        predictions = model.predict(test_data, verbose=0)
        
        print(f"Original model output shape: {predictions.shape}")
        print(f"Original model min probability: {predictions.min():.6f}")
        print(f"Original model max probability: {predictions.max():.6f}")
        print(f"Original model mean probability: {predictions.mean():.6f}")
        print(f"Sample predictions (first 3 outputs):")
        for i in range(min(3, len(predictions))):
            print(f"  Sample {i}: P={predictions[i,0]:.6f}, S={predictions[i,1]:.6f}, N={predictions[i,2]:.6f}")
        
        return predictions
        
    except Exception as e:
        print(f"Error testing original model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_converted_model():
    """Test the converted .keras model."""
    print("\nTesting converted model (easyquake_seisbench branch)...")
    
    try:
        import tensorflow as tf
        
        # Load converted model
        model_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
        model = tf.keras.models.load_model(model_path)
        
        # Create same test data
        test_data = np.random.randn(10, 400, 3).astype('float32')
        
        # Get predictions
        predictions = model.predict(test_data, verbose=0)
        
        print(f"Converted model output shape: {predictions.shape}")
        print(f"Converted model min probability: {predictions.min():.6f}")
        print(f"Converted model max probability: {predictions.max():.6f}")
        print(f"Converted model mean probability: {predictions.mean():.6f}")
        print(f"Sample predictions (first 3 outputs):")
        for i in range(min(3, len(predictions))):
            print(f"  Sample {i}: P={predictions[i,0]:.6f}, S={predictions[i,1]:.6f}, N={predictions[i,2]:.6f}")
        
        return predictions
        
    except Exception as e:
        print(f"Error testing converted model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Model Calibration Comparison Test")
    print("=" * 50)
    
    # Test original model
    orig_preds = test_original_model()
    
    # Switch to easyquake_seisbench branch for converted model
    print("\nSwitching to easyquake_seisbench branch...")
    os.system("cd /home/jwalter/easyQuake && git checkout easyquake_seisbench")
    
    # Test converted model
    conv_preds = test_converted_model()
    
    # Compare if both worked
    if orig_preds is not None and conv_preds is not None:
        print("\n" + "="*50)
        print("COMPARISON SUMMARY:")
        print(f"Original max prob: {orig_preds.max():.6f}")
        print(f"Converted max prob: {conv_preds.max():.6f}")
        print(f"Ratio (converted/original): {conv_preds.max()/orig_preds.max():.6f}")
        
        # Check if distributions are similar
        orig_sum = np.sum(orig_preds, axis=1)
        conv_sum = np.sum(conv_preds, axis=1)
        print(f"Original probability sums (should be ~1.0): {orig_sum[:3]}")
        print(f"Converted probability sums (should be ~1.0): {conv_sum[:3]}")
