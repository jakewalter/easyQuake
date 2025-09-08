#!/usr/bin/env python3
"""
GPD Model Wrapper for Exact Conversion
=====================================

This creates a wrapper around the exact multi-branch model that:
1. Takes the (batch*3, 3) output from the exact model
2. Reshapes and averages to produce (batch, 3) output
3. Makes it compatible with the existing GPD picker code
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model


def create_exact_model_wrapper(exact_model_path):
    """
    Create a wrapper around the exact multi-branch model to make it 
    compatible with the GPD picker expectations.
    """
    print(f"Creating wrapper for exact model: {exact_model_path}")
    
    # Load the exact model
    exact_model = keras.models.load_model(exact_model_path)
    
    # Create wrapper input
    wrapper_input = layers.Input(shape=(400, 3), name='wrapper_input')
    
    # Get prediction from exact model
    exact_output = exact_model(wrapper_input)  # Shape: (batch*3, 3)
    
    # Create a custom layer to reshape and average the multi-branch output
    class MultiBranchAverager(layers.Layer):
        def __init__(self, **kwargs):
            super(MultiBranchAverager, self).__init__(**kwargs)
        
        def call(self, inputs):
            # Input shape: (batch*3, 3)
            # We need to reshape to (batch, 3, 3) and average across the middle dimension
            
            # Get batch size (will be batch*3)
            batch_times_3 = tf.shape(inputs)[0]
            original_batch = batch_times_3 // 3
            
            # Reshape: (batch*3, 3) -> (batch, 3, 3)
            reshaped = tf.reshape(inputs, [original_batch, 3, 3])
            
            # Average across the 3 branches: (batch, 3, 3) -> (batch, 3)
            averaged = tf.reduce_mean(reshaped, axis=1)
            
            return averaged
        
        def compute_output_shape(self, input_shape):
            return (input_shape[0] // 3, input_shape[1])
    
    # Apply the averaging layer
    wrapper_output = MultiBranchAverager(name='multi_branch_averager')(exact_output)
    
    # Create the wrapper model
    wrapper_model = Model(inputs=wrapper_input, outputs=wrapper_output, name='gpd_exact_wrapper')
    
    return wrapper_model


def verify_wrapper_model(wrapper_model):
    """Verify the wrapper model produces the expected output format."""
    print("\\n=== Verifying Wrapper Model ===")
    
    try:
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            test_input = np.random.random((batch_size, 400, 3)).astype(np.float32)
            prediction = wrapper_model.predict(test_input, verbose=0)
            
            print(f"  Batch {batch_size}: input {test_input.shape} -> output {prediction.shape}")
            
            # Check output format
            expected_shape = (batch_size, 3)
            if prediction.shape == expected_shape:
                print(f"    ✓ Shape correct: {prediction.shape}")
            else:
                print(f"    ✗ Shape incorrect: expected {expected_shape}, got {prediction.shape}")
                return False
            
            # Check softmax properties
            for i in range(prediction.shape[0]):
                pred_sum = np.sum(prediction[i])
                if abs(pred_sum - 1.0) < 1e-5:
                    print(f"    ✓ Sample {i+1}: softmax sum = {pred_sum:.6f}")
                else:
                    print(f"    ✗ Sample {i+1}: softmax sum = {pred_sum:.6f} (should be ~1.0)")
                    return False
        
        return True
        
    except Exception as e:
        print(f"✗ Wrapper verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Create and save the wrapper model."""
    print("=== Creating GPD Exact Model Wrapper ===")
    print("Wrapping exact multi-branch model for GPD picker compatibility")
    print()
    
    base_dir = os.path.dirname(__file__)
    exact_model_path = os.path.join(base_dir, 'model_pol_exact_conversion.keras')
    
    if not os.path.exists(exact_model_path):
        print(f"✗ Exact model not found: {exact_model_path}")
        print("Run exact_gpd_conversion.py first to create the exact model")
        return
    
    try:
        # 1. Create wrapper model
        print("1. Creating wrapper model...")
        wrapper_model = create_exact_model_wrapper(exact_model_path)
        wrapper_model.summary()
        
        # 2. Verify wrapper
        print("\\n2. Verifying wrapper model...")
        if not verify_wrapper_model(wrapper_model):
            print("✗ Wrapper verification failed")
            return
        
        # 3. Save wrapper model
        print("\\n3. Saving wrapper model...")
        wrapper_output_path = os.path.join(base_dir, 'model_pol_exact_wrapper.keras')
        wrapper_model.save(wrapper_output_path)
        print(f"✓ Wrapper model saved: {wrapper_output_path}")
        
        # Also save as h5 for compatibility
        wrapper_h5_path = os.path.join(base_dir, 'model_pol_exact_wrapper.h5')
        wrapper_model.save(wrapper_h5_path)
        print(f"✓ Wrapper model saved: {wrapper_h5_path}")
        
        print("\\n=== Wrapper Creation Complete ===")
        print("The wrapper model:")
        print("- Loads the exact multi-branch model")
        print("- Averages the 3 branch outputs to produce (batch, 3) format")
        print("- Is compatible with existing GPD picker code")
        print("- Should produce the most accurate results matching TF1 model")
        
    except Exception as e:
        print(f"✗ Wrapper creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
