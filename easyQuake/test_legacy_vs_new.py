#!/usr/bin/env python3
"""Test the legacy H5 model vs the new keras model to understand probability scaling differences."""

import numpy as np
import tensorflow as tf

def test_legacy_h5_model():
    """Test the legacy H5 model file."""
    print("Testing legacy H5 model...")
    
    try:
        model_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_legacy.h5'
        model = tf.keras.models.load_model(model_path)
        
        # Create test data
        test_data = np.random.randn(10, 400, 3).astype('float32')
        
        # Get predictions
        predictions = model.predict(test_data, verbose=0)
        
        print(f"Legacy H5 model output shape: {predictions.shape}")
        print(f"Legacy H5 model min probability: {predictions.min():.6f}")
        print(f"Legacy H5 model max probability: {predictions.max():.6f}")
        print(f"Legacy H5 model mean probability: {predictions.mean():.6f}")
        print(f"Sample predictions (first 3 outputs):")
        for i in range(min(3, len(predictions))):
            print(f"  Sample {i}: P={predictions[i,0]:.6f}, S={predictions[i,1]:.6f}, N={predictions[i,2]:.6f}")
        
        return predictions
        
    except Exception as e:
        print(f"Error testing legacy H5 model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_new_keras_model():
    """Test the new .keras model."""
    print("\nTesting new .keras model...")
    
    try:
        model_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
        model = tf.keras.models.load_model(model_path)
        
        # Create same test data
        test_data = np.random.randn(10, 400, 3).astype('float32')
        
        # Get predictions
        predictions = model.predict(test_data, verbose=0)
        
        print(f"New .keras model output shape: {predictions.shape}")
        print(f"New .keras model min probability: {predictions.min():.6f}")
        print(f"New .keras model max probability: {predictions.max():.6f}")
        print(f"New .keras model mean probability: {predictions.mean():.6f}")
        print(f"Sample predictions (first 3 outputs):")
        for i in range(min(3, len(predictions))):
            print(f"  Sample {i}: P={predictions[i,0]:.6f}, S={predictions[i,1]:.6f}, N={predictions[i,2]:.6f}")
        
        return predictions
        
    except Exception as e:
        print(f"Error testing new .keras model: {e}")
        import traceback
        traceback.print_exc()
        return None

def inspect_model_architectures():
    """Compare the architectures of both models."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*60)
    
    try:
        legacy_model = tf.keras.models.load_model('/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_legacy.h5')
        print("Legacy model summary:")
        legacy_model.summary()
        
        print("\n" + "-"*60)
        
        new_model = tf.keras.models.load_model('/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras')
        print("New model summary:")
        new_model.summary()
        
        # Check if final layers are identical
        print("\n" + "-"*60)
        print("FINAL LAYER COMPARISON:")
        
        legacy_final = legacy_model.layers[-1]
        new_final = new_model.layers[-1]
        
        print(f"Legacy final layer: {legacy_final.name} ({legacy_final.__class__.__name__})")
        print(f"New final layer: {new_final.name} ({new_final.__class__.__name__})")
        
        # Check activation functions
        if hasattr(legacy_final, 'activation'):
            print(f"Legacy activation: {legacy_final.activation}")
        if hasattr(new_final, 'activation'):
            print(f"New activation: {new_final.activation}")
        
    except Exception as e:
        print(f"Error inspecting architectures: {e}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Legacy vs New Model Probability Comparison")
    print("=" * 60)
    
    # Test both models
    legacy_preds = test_legacy_h5_model()
    new_preds = test_new_keras_model()
    
    # Compare if both worked
    if legacy_preds is not None and new_preds is not None:
        print("\n" + "="*60)
        print("PROBABILITY COMPARISON SUMMARY:")
        print(f"Legacy H5 max prob: {legacy_preds.max():.6f}")
        print(f"New .keras max prob: {new_preds.max():.6f}")
        print(f"Ratio (new/legacy): {new_preds.max()/legacy_preds.max():.6f}")
        
        # Check if they're exactly the same
        if np.allclose(legacy_preds, new_preds, atol=1e-6):
            print("✓ Models produce identical predictions!")
        else:
            print("✗ Models produce different predictions")
            print(f"Max absolute difference: {np.abs(legacy_preds - new_preds).max():.6f}")
    
    # Inspect architectures
    inspect_model_architectures()
