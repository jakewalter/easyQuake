#!/usr/bin/env python3
"""
Create a wrapper for the exact multi-branch model that outputs the expected format.

The exact model outputs (batch*3, 3) but the GPD picker expects (batch, 3).
This wrapper reshapes and averages the multi-branch output.
"""

import os
import numpy as np
import keras
from keras import layers
from keras.models import Model


def create_gpd_wrapper_model():
    """
    Create a wrapper around the exact multi-branch model that:
    1. Loads the exact converted model
    2. Reshapes output from (batch*3, 3) to (batch, 3, 3) 
    3. Averages across the 3 branches to get (batch, 3)
    """
    base_dir = os.path.dirname(__file__)
    exact_model_path = os.path.join(base_dir, 'model_pol_exact_converted.keras')
    
    if not os.path.exists(exact_model_path):
        raise FileNotFoundError(f"Exact model not found: {exact_model_path}")
    
    # Load the exact model
    exact_model = keras.models.load_model(exact_model_path, safe_mode=False)
    
    # Create wrapper
    input_layer = layers.Input(shape=(400, 3), name='wrapper_input')
    
    # Pass through exact model
    exact_output = exact_model(input_layer)  # Shape: (batch*3, 3)
    
    # Reshape to (batch, 3, 3) and average across branches
    def reshape_and_average(x):
        import tensorflow as tf
        batch_size = tf.shape(x)[0] // 3
        # Reshape from (batch*3, 3) to (batch, 3, 3)
        reshaped = tf.reshape(x, [batch_size, 3, 3])
        # Average across the 3 branches (middle dimension)
        averaged = tf.reduce_mean(reshaped, axis=1)  # Shape: (batch, 3)
        return averaged
    
    wrapper_output = layers.Lambda(reshape_and_average, name='average_branches')(exact_output)
    
    # Create the wrapper model
    wrapper_model = Model(inputs=input_layer, outputs=wrapper_output, name='gpd_wrapper_model')
    
    return wrapper_model


def main():
    """Create and save the wrapper model."""
    print("=== Creating GPD Wrapper Model ===")
    print("Wrapping exact multi-branch model for single-output format")
    
    try:
        # Create wrapper model
        print("1. Creating wrapper model...")
        wrapper_model = create_gpd_wrapper_model()
        wrapper_model.summary()
        
        # Test the wrapper
        print("\\n2. Testing wrapper model...")
        test_input = np.random.random((2, 400, 3)).astype(np.float32)
        prediction = wrapper_model.predict(test_input, verbose=0)
        
        print(f"✓ Wrapper test successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {prediction.shape}")
        print(f"  Expected: (2, 3)")
        
        # Verify softmax properties
        for i in range(prediction.shape[0]):
            pred_sum = np.sum(prediction[i])
            print(f"  Sample {i}: sum={pred_sum:.6f}, prediction={prediction[i]}")
        
        # Save wrapper model
        print("\\n3. Saving wrapper model...")
        base_dir = os.path.dirname(__file__)
        wrapper_path = os.path.join(base_dir, 'model_pol_wrapped_exact.keras')
        wrapper_model.save(wrapper_path)
        print(f"✓ Wrapper model saved: {wrapper_path}")
        
        print("\\n=== Wrapper Creation Complete ===")
        print("The wrapper model converts multi-branch output to single-branch format")
        print("compatible with the existing GPD picker code.")
        
    except Exception as e:
        print(f"✗ Wrapper creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
