#!/usr/bin/env python3

import tensorflow as tf
import keras
import numpy as np
import os

def create_manual_model():
    """Manually recreate the model architecture from the JSON description"""
    
    print("Manually recreating the model architecture...")
    
    # Input layer: (None, 400, 3)
    input_layer = keras.layers.Input(shape=(400, 3), name='conv1d_1_input')
    
    # Lambda layers for splitting the input into 3 parts
    # From JSON: lambda functions split input into 3 parts of ~133 samples each
    def get_slice_0(x):
        return x[:, 0:133, :]
    
    def get_slice_1(x):
        return x[:, 133:266, :]
    
    def get_slice_2(x):
        return x[:, 266:400, :]
    
    lambda_1 = keras.layers.Lambda(get_slice_0, name='lambda_1')(input_layer)
    lambda_2 = keras.layers.Lambda(get_slice_1, name='lambda_2')(input_layer) 
    lambda_3 = keras.layers.Lambda(get_slice_2, name='lambda_3')(input_layer)
    
    # Create the sequential CNN part (this is applied to each lambda output)
    def create_cnn_sequential():
        model = keras.Sequential([
            # Conv1D block 1: 32 filters, kernel=21
            keras.layers.Conv1D(32, 21, padding='same', activation='linear', name='conv1d_1'),
            keras.layers.BatchNormalization(name='batch_normalization_1'),
            keras.layers.Activation('relu', name='activation_1'),
            keras.layers.MaxPooling1D(2, name='max_pooling1d_1'),
            
            # Conv1D block 2: 64 filters, kernel=15  
            keras.layers.Conv1D(64, 15, padding='same', activation='linear', name='conv1d_2'),
            keras.layers.BatchNormalization(name='batch_normalization_2'),
            keras.layers.Activation('relu', name='activation_2'),
            keras.layers.MaxPooling1D(2, name='max_pooling1d_2'),
            
            # Conv1D block 3: 128 filters, kernel=11
            keras.layers.Conv1D(128, 11, padding='same', activation='linear', name='conv1d_3'),
            keras.layers.BatchNormalization(name='batch_normalization_3'),
            keras.layers.Activation('relu', name='activation_3'),
            keras.layers.MaxPooling1D(2, name='max_pooling1d_3'),
            
            # Conv1D block 4: 256 filters, kernel=9
            keras.layers.Conv1D(256, 9, padding='same', activation='linear', name='conv1d_4'),
            keras.layers.BatchNormalization(name='batch_normalization_4'),
            keras.layers.Activation('relu', name='activation_4'),
            keras.layers.MaxPooling1D(2, name='max_pooling1d_4'),
            
            # Dense layers
            keras.layers.Flatten(name='flatten_1'),
            keras.layers.Dense(200, activation='linear', name='dense_1'),
            keras.layers.BatchNormalization(name='batch_normalization_5'),
            keras.layers.Activation('relu', name='activation_5'),
            keras.layers.Dense(200, activation='linear', name='dense_2'),
            keras.layers.BatchNormalization(name='batch_normalization_6'),
            keras.layers.Activation('relu', name='activation_6'),
            keras.layers.Dense(3, activation='linear', name='dense_3'),
            keras.layers.Activation('softmax', name='activation_7')
        ], name='sequential_1')
        return model
    
    # Apply the CNN to each lambda output
    cnn = create_cnn_sequential()
    
    # Process each slice through the same CNN
    out1 = cnn(lambda_1)
    out2 = cnn(lambda_2) 
    out3 = cnn(lambda_3)
    
    # Concatenate the outputs along axis 0 (batch dimension)
    # Note: This is tricky - original uses axis=0 which concatenates batches
    # For inference, we might need to handle this differently
    concatenated = keras.layers.Concatenate(axis=0, name='activation_7_concat')([out1, out2, out3])
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=concatenated, name='model_1')
    
    return model

def test_manual_reconstruction():
    """Test the manually reconstructed model"""
    
    hdf5_file = 'model_pol_best.hdf5'
    
    if not os.path.exists(hdf5_file):
        print(f"Missing file: {hdf5_file}")
        return None
    
    try:
        print("Creating manual model reconstruction...")
        model = create_manual_model()
        
        print("✅ Model architecture created successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Try to load weights
        print("Attempting to load weights...")
        model.load_weights(hdf5_file, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded (with skip_mismatch=True)")
        
        # Test with dummy data
        test_input = np.random.randn(1, 400, 3).astype(np.float32)
        print(f"Test input shape: {test_input.shape}")
        
        predictions = model.predict(test_input, verbose=0)
        print(f"✅ Model prediction successful")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        return model
        
    except Exception as e:
        print(f"❌ Manual reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = test_manual_reconstruction()
