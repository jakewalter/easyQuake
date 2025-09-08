#!/usr/bin/env python3
"""Deep dive into the model layers to find what changed."""

import tensorflow as tf
import numpy as np

def inspect_final_layers():
    """Inspect the final layers in detail."""
    
    model_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
    model = tf.keras.models.load_model(model_path)
    
    print("=== DETAILED LAYER INSPECTION ===")
    
    # Get the sequential model (main part)
    sequential_model = model.layers[1]  # The Sequential layer
    
    print(f"Sequential model has {len(sequential_model.layers)} layers")
    
    # Look at the last few layers
    for i, layer in enumerate(sequential_model.layers[-5:]):
        layer_idx = len(sequential_model.layers) - 5 + i
        print(f"\nLayer {layer_idx}: {layer.name} ({layer.__class__.__name__})")
        
        if hasattr(layer, 'activation'):
            print(f"  Activation: {layer.activation}")
        if hasattr(layer, 'units'):
            print(f"  Units: {layer.units}")
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                print(f"  Weights shape: {[w.shape for w in weights]}")
                for j, w in enumerate(weights):
                    print(f"    Weight {j}: min={np.min(w):.6f}, max={np.max(w):.6f}, mean={np.mean(w):.6f}")
    
    # Test intermediate outputs
    print("\n=== INTERMEDIATE OUTPUTS ===")
    
    # Create test input
    test_input = np.random.randn(1, 400, 3).astype('float32')
    
    # Get output from second-to-last layer (before softmax)
    if len(sequential_model.layers) >= 2:
        pre_softmax_model = tf.keras.Model(
            inputs=sequential_model.input,
            outputs=sequential_model.layers[-2].output
        )
        
        pre_softmax_output = pre_softmax_model.predict(test_input, verbose=0)
        print(f"Pre-softmax output: {pre_softmax_output[0]}")
        print(f"Pre-softmax range: {np.min(pre_softmax_output)} to {np.max(pre_softmax_output)}")
        
        # Manually apply softmax to see if that's the issue
        manual_softmax = tf.nn.softmax(pre_softmax_output).numpy()
        print(f"Manual softmax: {manual_softmax[0]}")
        
        # Check if pre-softmax values are too similar (no discrimination)
        pre_soft = pre_softmax_output[0]
        max_diff = np.max(pre_soft) - np.min(pre_soft)
        print(f"Pre-softmax max difference: {max_diff:.6f}")
        
        if max_diff < 0.1:
            print("WARNING: Pre-softmax outputs are very similar - model has lost discriminative power!")
    
    return model

def test_different_inputs():
    """Test with different types of input to see if the model responds."""
    
    print("\n" + "="*60)
    print("TESTING MODEL RESPONSE TO DIFFERENT INPUTS")
    print("="*60)
    
    model_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
    model = tf.keras.models.load_model(model_path)
    
    # Test 1: All zeros
    zeros_input = np.zeros((1, 400, 3)).astype('float32')
    zeros_output = model.predict(zeros_input, verbose=0)
    print(f"All zeros input: {zeros_output[0]}")
    
    # Test 2: All ones
    ones_input = np.ones((1, 400, 3)).astype('float32')
    ones_output = model.predict(ones_input, verbose=0)
    print(f"All ones input: {ones_output[0]}")
    
    # Test 3: Large values
    large_input = np.ones((1, 400, 3)).astype('float32') * 10.0
    large_output = model.predict(large_input, verbose=0)
    print(f"Large values input: {large_output[0]}")
    
    # Test 4: Sinusoidal (P-wave like)
    t = np.linspace(0, 4*np.pi, 400)
    sin_input = np.zeros((1, 400, 3)).astype('float32')
    sin_input[0, :, 2] = np.sin(t) * 0.5  # Put signal in Z channel
    sin_output = model.predict(sin_input, verbose=0)
    print(f"Sinusoidal Z input: {sin_output[0]}")
    
    # Test 5: High frequency noise
    noise_input = np.random.randn(1, 400, 3).astype('float32') * 5.0
    noise_output = model.predict(noise_input, verbose=0)
    print(f"High noise input: {noise_output[0]}")
    
    print(f"\nObservation: All outputs are very similar regardless of input!")
    print(f"This suggests the model weights may be corrupted or normalized incorrectly.")

if __name__ == "__main__":
    model = inspect_final_layers()
    test_different_inputs()
