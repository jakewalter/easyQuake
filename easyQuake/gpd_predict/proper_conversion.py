#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import h5py
import json

def create_slicing_layer(input_layer, slice_index, total_parts):
    """Create a slicing layer equivalent to the original Lambda layers"""
    def slice_func(x):
        # This mimics the original get_slice function from multi_gpu_utils
        batch_size = tf.shape(x)[0]
        input_shape = x.shape[1:]  # (400, 3)
        
        # Calculate slice parameters
        step = input_shape[1] // total_parts  # 3 // 3 = 1
        start = slice_index * step  # 0, 1, or 2
        size = step  # 1
        
        # Slice the last dimension (channels)
        return x[:, :, start:start+size]
    
    return tf.keras.layers.Lambda(slice_func, output_shape=(400, 1))(input_layer)

def rebuild_model_from_original():
    """Rebuild the model architecture without relying on pickled functions"""
    
    # Load the sequential model weights from HDF5
    with h5py.File('model_pol_best.hdf5', 'r') as f:
        print("HDF5 structure:")
        def print_structure(name, obj):
            print(f"  {name}: {type(obj)}")
        f.visititems(print_structure)
    
    # Create input layer
    input_layer = tf.keras.layers.Input(shape=(400, 3), name='conv1d_1_input')
    
    # Create three slicing layers (equivalent to lambda_1, lambda_2, lambda_3)
    slice_1 = create_slicing_layer(input_layer, 0, 3)  # Channel 0
    slice_2 = create_slicing_layer(input_layer, 1, 3)  # Channel 1  
    slice_3 = create_slicing_layer(input_layer, 2, 3)  # Channel 2
    
    # Create the sequential CNN model (shared architecture)
    def create_sequential_cnn():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 21, padding='same', activation='linear', name='conv1d_1'),
            tf.keras.layers.BatchNormalization(name='batch_normalization_1'),
            tf.keras.layers.Activation('relu', name='activation_1'),
            tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_1'),
            
            tf.keras.layers.Conv1D(64, 15, padding='same', activation='linear', name='conv1d_2'),
            tf.keras.layers.BatchNormalization(name='batch_normalization_2'),
            tf.keras.layers.Activation('relu', name='activation_2'),
            tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_2'),
            
            tf.keras.layers.Conv1D(128, 11, padding='same', activation='linear', name='conv1d_3'),
            tf.keras.layers.BatchNormalization(name='batch_normalization_3'),
            tf.keras.layers.Activation('relu', name='activation_3'),
            tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_3'),
            
            tf.keras.layers.Conv1D(256, 9, padding='same', activation='linear', name='conv1d_4'),
            tf.keras.layers.BatchNormalization(name='batch_normalization_4'),
            tf.keras.layers.Activation('relu', name='activation_4'),
            tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_4'),
            
            tf.keras.layers.Flatten(name='flatten_1'),
            tf.keras.layers.Dense(200, activation='linear', name='dense_1'),
            tf.keras.layers.BatchNormalization(name='batch_normalization_5'),
            tf.keras.layers.Activation('relu', name='activation_5'),
            tf.keras.layers.Dense(200, activation='linear', name='dense_2'),
            tf.keras.layers.BatchNormalization(name='batch_normalization_6'),
            tf.keras.layers.Activation('relu', name='activation_6'),
            tf.keras.layers.Dense(3, activation='linear', name='dense_3'),
            tf.keras.layers.Activation('softmax', name='activation_7')
        ], name='sequential_1')
        return model
    
    # Apply the same sequential model to each slice
    sequential_cnn = create_sequential_cnn()
    
    output_1 = sequential_cnn(slice_1)
    output_2 = sequential_cnn(slice_2) 
    output_3 = sequential_cnn(slice_3)
    
    # Concatenate outputs along batch dimension (axis=0)
    # Note: This is unusual - typically concatenate along feature dimension
    # But the original model does axis=0 concatenation
    final_output = tf.keras.layers.Concatenate(axis=0, name='activation_7_concat')([output_1, output_2, output_3])
    
    # Create the complete model
    model = tf.keras.Model(inputs=input_layer, outputs=final_output)
    
    return model

def load_weights_manually(model):
    """Load weights from the HDF5 file manually"""
    try:
        # Try to load weights by name matching
        print("Loading weights from model_pol_best.hdf5...")
        model.load_weights('model_pol_best.hdf5', by_name=True)
        print("Weights loaded successfully!")
        return True
    except Exception as e:
        print(f"Weight loading failed: {e}")
        return False

if __name__ == "__main__":
    print("Rebuilding model architecture...")
    
    # Actually, let's try a simpler approach first
    # Let's see if we can just fix the existing converted model
    
    # Load an existing converted model and examine its structure
    try:
        print("Loading existing converted model...")
        model = tf.keras.models.load_model('model_pol_new.keras')
        print("Model loaded. Architecture:")
        model.summary()
        
        # Check what the model actually does with test input
        test_input = np.random.randn(1, 400, 3)
        output = model.predict(test_input, verbose=0)
        print(f"Test output shape: {output.shape}")
        print(f"Test output: {output[0]}")
        print(f"Output sum: {np.sum(output[0]):.6f}")
        
    except Exception as e:
        print(f"Failed to load existing model: {e}")
        
        # If that fails, try the manual rebuild
        print("\nAttempting manual rebuild...")
        try:
            model = rebuild_model_from_original()
            print("Manual model created:")
            model.summary()
            
            if load_weights_manually(model):
                print("Saving properly converted model...")
                model.save('model_pol_proper_conversion.keras')
                print("Saved as model_pol_proper_conversion.keras")
            else:
                print("Could not load weights properly")
                
        except Exception as e2:
            print(f"Manual rebuild also failed: {e2}")
