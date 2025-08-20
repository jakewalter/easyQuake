#!/usr/bin/env python3
"""Check if the model weights were properly converted."""

import tensorflow as tf
import numpy as np
import json
import os

def compare_original_vs_converted():
    """Compare the original model architecture with converted model."""
    
    print("=== COMPARING ORIGINAL VS CONVERTED MODEL ===")
    
    # Check if we have the original JSON file
    json_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol.json'
    weights_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_best.hdf5'
    converted_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
    
    print(f"Original JSON exists: {os.path.exists(json_path)}")
    print(f"Original weights exist: {os.path.exists(weights_path)}")
    print(f"Converted model exists: {os.path.exists(converted_path)}")
    
    # Load the converted model
    converted_model = tf.keras.models.load_model(converted_path)
    
    # Get the final dense layer weights from converted model
    sequential_layer = converted_model.layers[1]
    final_dense = None
    
    for layer in sequential_layer.layers:
        if 'dense_3' in layer.name or (hasattr(layer, 'units') and layer.units == 3):
            final_dense = layer
            break
    
    if final_dense:
        weights, biases = final_dense.get_weights()
        print(f"\nConverted model final dense layer:")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weights range: {np.min(weights):.6f} to {np.max(weights):.6f}")
        print(f"  Weights mean: {np.mean(weights):.6f}")
        print(f"  Biases: {biases}")
        print(f"  Bias sum: {np.sum(biases):.6f}")
        
        # Check if biases are all zero (sign of untrained model)
        if np.allclose(biases, 0.0):
            print("  ❌ WARNING: All biases are zero - model may not be properly trained!")
        
        # Check if weights are small/uniform (another sign)
        weight_std = np.std(weights)
        print(f"  Weight standard deviation: {weight_std:.6f}")
        
        if weight_std < 0.01:
            print("  ❌ WARNING: Weights have very low variance - model may not be trained!")
    
    # Try to load the original model for comparison (if possible)
    print(f"\n=== ATTEMPTING TO LOAD ORIGINAL MODEL ===")
    
    try:
        # Read the JSON architecture
        with open(json_path, 'r') as f:
            model_json = f.read()
        
        print("✓ Original JSON loaded successfully")
        
        # Try to parse it to see the original architecture
        import json
        model_config = json.loads(model_json)
        
        # Look for final layer configuration
        if 'config' in model_config and 'layers' in model_config['config']:
            layers = model_config['config']['layers']
            
            # Find the dense layer with 3 units
            for layer in layers:
                if (layer.get('class_name') == 'Dense' and 
                    layer.get('config', {}).get('units') == 3):
                    print(f"Original final dense layer config: {layer['config']}")
                    
                    # Check activation function
                    activation = layer['config'].get('activation', 'unknown')
                    print(f"Original activation: {activation}")
                    
                    if activation != 'linear':
                        print(f"❌ WARNING: Original model final dense activation was '{activation}', not 'linear'!")
                        print("This could explain the probability calibration issue!")
        
    except Exception as e:
        print(f"❌ Could not load original model: {e}")
    
    return converted_model

def test_hypothesis_about_final_activation():
    """Test if the issue is related to the final activation function."""
    
    print(f"\n=== TESTING FINAL ACTIVATION HYPOTHESIS ===")
    
    # Load converted model
    converted_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
    model = tf.keras.models.load_model(converted_path)
    
    # Get test input
    test_input = np.random.randn(1, 400, 3).astype('float32')
    
    # Get the sequential part
    sequential_model = model.layers[1]
    
    # Create a model that outputs the pre-softmax values
    pre_softmax_layers = sequential_model.layers[:-1]  # All layers except the last (softmax)
    
    # Manually build a model to get pre-softmax outputs
    temp_input = tf.keras.Input(shape=(400, 3))
    x = temp_input
    
    for layer in pre_softmax_layers:
        x = layer(x)
    
    pre_softmax_model = tf.keras.Model(inputs=temp_input, outputs=x)
    
    pre_softmax_output = pre_softmax_model.predict(test_input, verbose=0)
    
    print(f"Pre-softmax output: {pre_softmax_output[0]}")
    print(f"Pre-softmax range: {np.min(pre_softmax_output):.6f} to {np.max(pre_softmax_output):.6f}")
    
    # Check if pre-softmax values are all very similar
    pre_vals = pre_softmax_output[0]
    max_diff = np.max(pre_vals) - np.min(pre_vals)
    print(f"Pre-softmax max difference: {max_diff:.6f}")
    
    if max_diff < 0.1:
        print("❌ Pre-softmax outputs are nearly identical!")
        print("   This means the model has no discriminative power.")
        print("   The weights may have been corrupted during conversion.")
    else:
        print("✓ Pre-softmax outputs show some variation")
    
    # Test what happens if we scale the pre-softmax values
    print(f"\nTesting scaled pre-softmax values:")
    for scale in [1, 2, 5, 10, 20]:
        scaled = pre_softmax_output * scale
        manual_softmax = tf.nn.softmax(scaled).numpy()
        print(f"  Scale {scale:2d}: {manual_softmax[0]}")

if __name__ == "__main__":
    model = compare_original_vs_converted()
    test_hypothesis_about_final_activation()
