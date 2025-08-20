#!/usr/bin/env python3
"""
Examine the detailed structure of the converted GPD model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

def examine_model_structure():
    """Examine the detailed structure of the converted model"""
    model_path = "model_pol_new.keras"
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"\n=== TOP-LEVEL MODEL ===")
    print(f"Model type: {type(model)}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i}: {layer.name} ({type(layer).__name__})")
        if hasattr(layer, 'input_shape'):
            print(f"  Input shape: {layer.input_shape}")
        if hasattr(layer, 'output_shape'):
            print(f"  Output shape: {layer.output_shape}")
        if hasattr(layer, 'activation') and layer.activation is not None:
            print(f"  Activation: {layer.activation}")
    
    # Look for Sequential model within
    sequential_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.models.Sequential):
            sequential_layer = layer
            break
    
    if sequential_layer:
        print(f"\n=== SEQUENTIAL MODEL INSIDE ===")
        print(f"Sequential model layers: {len(sequential_layer.layers)}")
        
        for i, layer in enumerate(sequential_layer.layers):
            print(f"\nSeq Layer {i}: {layer.name} ({type(layer).__name__})")
            if hasattr(layer, 'input_shape'):
                print(f"  Input shape: {layer.input_shape}")
            if hasattr(layer, 'output_shape'):
                print(f"  Output shape: {layer.output_shape}")
            if hasattr(layer, 'activation') and layer.activation is not None:
                print(f"  Activation: {layer.activation}")
                if hasattr(layer.activation, '__name__'):
                    print(f"  Activation name: {layer.activation.__name__}")
            if hasattr(layer, 'units'):
                print(f"  Units: {layer.units}")
    
    # Test with sample input
    print(f"\n=== TESTING WITH SAMPLE INPUT ===")
    test_input = np.random.randn(1, 400, 3).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")
    
    prediction = model.predict(test_input, verbose=0)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction: {prediction[0]}")
    print(f"Sum of probabilities: {np.sum(prediction[0]):.6f}")
    print(f"Max probability: {np.max(prediction[0]):.6f}")
    
    return model, sequential_layer

def create_temperature_scaled_version(model, temperature=0.05):
    """Create temperature scaled version by modifying the sequential part"""
    print(f"\n=== CREATING TEMPERATURE SCALED VERSION (T={temperature}) ===")
    
    # Get the sequential model
    sequential_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.models.Sequential):
            sequential_layer = layer
            break
    
    if not sequential_layer:
        print("No sequential layer found")
        return None
    
    # Find the softmax layer in the sequential model
    softmax_idx = None
    for i, layer in enumerate(sequential_layer.layers):
        if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
            if layer.activation.__name__ == 'softmax':
                softmax_idx = i
                print(f"Found softmax at sequential layer {i}: {layer.name}")
                break
    
    if softmax_idx is None:
        print("No softmax layer found in sequential model")
        return None
    
    # Create new sequential model
    new_layers = []
    
    for i, layer in enumerate(sequential_layer.layers):
        if i == softmax_idx:
            # Replace softmax with linear activation
            print(f"Replacing softmax layer {layer.name} with linear + temperature scaling")
            
            # Copy the layer but with linear activation
            if isinstance(layer, keras.layers.Dense):
                new_layer = keras.layers.Dense(
                    units=layer.units,
                    activation='linear',  # Change to linear
                    use_bias=layer.use_bias,
                    name=layer.name + '_linear'
                )
                new_layers.append(new_layer)
                
                # Add temperature scaling
                temp_layer = keras.layers.Lambda(
                    lambda x: x / temperature,
                    name='temperature_scaling'
                )
                new_layers.append(temp_layer)
                
                # Add softmax
                softmax_layer = keras.layers.Activation('softmax', name='softmax_scaled')
                new_layers.append(softmax_layer)
            else:
                # If it's an Activation layer with softmax
                # Skip it and add our own scaling + softmax
                temp_layer = keras.layers.Lambda(
                    lambda x: x / temperature,
                    name='temperature_scaling'
                )
                new_layers.append(temp_layer)
                
                softmax_layer = keras.layers.Activation('softmax', name='softmax_scaled')
                new_layers.append(softmax_layer)
        else:
            # Copy layer as-is
            new_layers.append(layer)
    
    # Build new sequential model
    new_sequential = keras.models.Sequential(new_layers, name='scaled_sequential')
    
    # Build the model to set up weights
    new_sequential.build(input_shape=sequential_layer.input_shape)
    
    # Copy weights (matching layers only)
    for i, (old_layer, new_layer) in enumerate(zip(sequential_layer.layers, new_sequential.layers)):
        if old_layer.get_weights() and new_layer.get_weights():
            if len(old_layer.get_weights()) == len(new_layer.get_weights()):
                print(f"Copying weights for layer {i}")
                new_layer.set_weights(old_layer.get_weights())
            else:
                print(f"Weight shape mismatch for layer {i}")
    
    # Create new full model
    inputs = model.input
    outputs = new_sequential(inputs)
    new_model = keras.Model(inputs=inputs, outputs=outputs, name='gpd_temperature_scaled')
    
    return new_model

def main():
    print("GPD Model Structure Examination")
    print("=" * 50)
    
    try:
        model, sequential_layer = examine_model_structure()
        
        if sequential_layer:
            # Try creating temperature scaled version
            for temp in [0.1, 0.05, 0.03]:
                print(f"\n{'='*60}")
                print(f"TRYING TEMPERATURE = {temp}")
                
                try:
                    scaled_model = create_temperature_scaled_version(model, temp)
                    if scaled_model:
                        # Test it
                        test_input = np.random.randn(5, 400, 3).astype(np.float32)
                        
                        orig_pred = model.predict(test_input, verbose=0)
                        scaled_pred = scaled_model.predict(test_input, verbose=0)
                        
                        print(f"Original max probs: {np.max(orig_pred, axis=1)}")
                        print(f"Scaled max probs: {np.max(scaled_pred, axis=1)}")
                        
                        high_conf = np.sum(np.max(scaled_pred, axis=1) > 0.99)
                        print(f"High confidence predictions: {high_conf}/{len(scaled_pred)}")
                        
                        if high_conf > 0:
                            print(f"✓ Success with temperature {temp}")
                            scaled_model.save(f"model_pol_temp_{temp}.keras")
                            print(f"Saved model with temperature {temp}")
                            break
                            
                except Exception as e:
                    print(f"Failed with temperature {temp}: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
