#!/usr/bin/env python3
"""
Direct fix for the converted GPD model to restore temperature scaling.
Instead of trying to reconstruct from original, modify the converted model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import json

def load_converted_model():
    """Load the current converted model"""
    model_path = "model_pol_new.keras"
    print(f"Loading converted model: {model_path}")
    model = keras.models.load_model(model_path)
    return model

def analyze_converted_model(model):
    """Analyze the converted model structure"""
    print("\n=== CONVERTED MODEL ANALYSIS ===")
    print(f"Model type: {type(model)}")
    print(f"Number of layers: {len(model.layers)}")
    
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name} ({type(layer).__name__})")
        if hasattr(layer, 'activation') and layer.activation is not None:
            print(f"  Activation: {layer.activation}")
    
    return model

def create_temperature_scaled_model(original_model, temperature=0.05):
    """Create a new model with temperature scaling applied before softmax"""
    print(f"\n=== CREATING TEMPERATURE SCALED MODEL (T={temperature}) ===")
    
    # Find the last dense layer (before softmax)
    last_dense_idx = None
    softmax_idx = None
    
    for i, layer in enumerate(original_model.layers):
        if isinstance(layer, keras.layers.Dense):
            last_dense_idx = i
        if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
            if layer.activation.__name__ == 'softmax':
                softmax_idx = i
    
    print(f"Last dense layer index: {last_dense_idx}")
    print(f"Softmax layer index: {softmax_idx}")
    
    if last_dense_idx is None or softmax_idx is None:
        print("Could not find dense or softmax layers")
        return None
    
    # Create new model by copying layers up to (but not including) softmax
    inputs = original_model.input
    x = inputs
    
    # Copy all layers except the softmax
    for i, layer in enumerate(original_model.layers[1:], 1):  # Skip input layer
        if i == softmax_idx:
            # Before softmax, apply temperature scaling
            print(f"Applying temperature scaling (1/{temperature} = {1/temperature:.1f}x) before softmax")
            x = keras.layers.Lambda(lambda logits: logits / temperature, name='temperature_scaling')(x)
            # Now apply softmax
            x = keras.layers.Activation('softmax', name='softmax_scaled')(x)
        else:
            if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
                if layer.activation.__name__ == 'softmax':
                    continue  # Skip original softmax, we replaced it
            x = layer(x)
    
    # Create the new model
    scaled_model = keras.Model(inputs=inputs, outputs=x, name='gpd_model_temperature_scaled')
    
    # Copy weights from original model (they match exactly except for the new layers)
    for i, (orig_layer, new_layer) in enumerate(zip(original_model.layers, scaled_model.layers)):
        if orig_layer.get_weights():  # If layer has weights
            if len(orig_layer.get_weights()) == len(new_layer.get_weights()):
                print(f"Copying weights for layer {i}: {orig_layer.name}")
                new_layer.set_weights(orig_layer.get_weights())
    
    return scaled_model

def test_scaling_effect(original_model, scaled_model, test_input):
    """Test the scaling effect on probability outputs"""
    print("\n=== TESTING SCALING EFFECT ===")
    
    # Get predictions from both models
    orig_pred = original_model.predict(test_input, verbose=0)
    scaled_pred = scaled_model.predict(test_input, verbose=0)
    
    print("Original model predictions (first 3 samples):")
    for i in range(min(3, len(orig_pred))):
        print(f"  Sample {i}: {orig_pred[i]}")
        print(f"    Max prob: {np.max(orig_pred[i]):.4f}")
    
    print("\nTemperature scaled model predictions (first 3 samples):")
    for i in range(min(3, len(scaled_pred))):
        print(f"  Sample {i}: {scaled_pred[i]}")
        print(f"    Max prob: {np.max(scaled_pred[i]):.4f}")
    
    # Check if scaling achieves the desired confidence levels
    high_confidence_orig = np.sum(np.max(orig_pred, axis=1) > 0.99)
    high_confidence_scaled = np.sum(np.max(scaled_pred, axis=1) > 0.99)
    
    print(f"\nSamples with >99% confidence:")
    print(f"  Original: {high_confidence_orig}/{len(orig_pred)} ({100*high_confidence_orig/len(orig_pred):.1f}%)")
    print(f"  Scaled: {high_confidence_scaled}/{len(scaled_pred)} ({100*high_confidence_scaled/len(scaled_pred):.1f}%)")
    
    return scaled_pred

def save_corrected_model(model, filename="model_pol_corrected.keras"):
    """Save the corrected model"""
    print(f"\n=== SAVING CORRECTED MODEL ===")
    print(f"Saving to: {filename}")
    model.save(filename)
    print("✓ Model saved successfully")

def main():
    print("GPD Model Temperature Scaling Fix")
    print("=" * 50)
    
    # Load converted model
    try:
        converted_model = load_converted_model()
    except Exception as e:
        print(f"✗ Failed to load converted model: {e}")
        return
    
    # Analyze structure
    analyze_converted_model(converted_model)
    
    # Create test input (realistic seismic data shape)
    print("\n=== CREATING TEST INPUT ===")
    test_input = np.random.randn(10, 400, 3).astype(np.float32)  # 10 samples
    print(f"Test input shape: {test_input.shape}")
    
    # Test different temperature values
    best_model = None
    best_temp = None
    best_confidence = 0
    
    for temperature in [0.1, 0.05, 0.03, 0.02, 0.01]:
        print(f"\n{'='*60}")
        print(f"TESTING TEMPERATURE = {temperature}")
        print(f"{'='*60}")
        
        try:
            scaled_model = create_temperature_scaled_model(converted_model, temperature)
            if scaled_model is None:
                continue
                
            scaled_pred = test_scaling_effect(converted_model, scaled_model, test_input)
            
            # Count high confidence predictions
            high_conf_count = np.sum(np.max(scaled_pred, axis=1) > 0.99)
            confidence_rate = high_conf_count / len(scaled_pred)
            
            if confidence_rate > best_confidence:
                best_confidence = confidence_rate
                best_model = scaled_model
                best_temp = temperature
                
        except Exception as e:
            print(f"✗ Failed with temperature {temperature}: {e}")
    
    if best_model is not None:
        print(f"\n{'='*60}")
        print(f"BEST RESULT: Temperature = {best_temp}")
        print(f"Confidence rate: {best_confidence:.1%}")
        print(f"{'='*60}")
        
        # Save the best model
        save_corrected_model(best_model, "model_pol_temperature_corrected.keras")
        
        # Test with threshold 0.994
        print(f"\n=== TESTING WITH THRESHOLD 0.994 ===")
        test_predictions = best_model.predict(test_input, verbose=0)
        picks_above_threshold = np.sum(np.max(test_predictions, axis=1) > 0.994)
        print(f"Samples above 0.994 threshold: {picks_above_threshold}/{len(test_predictions)} ({100*picks_above_threshold/len(test_predictions):.1f}%)")
        
    else:
        print("\n✗ No suitable temperature scaling found")

if __name__ == "__main__":
    main()
