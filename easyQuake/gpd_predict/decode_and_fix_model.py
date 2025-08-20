#!/usr/bin/env python3

import base64
import pickle
import json

def decode_lambda_functions():
    """Decode the serialized lambda functions from the JSON"""
    
    with open('model_pol.json', 'r') as f:
        model_config = json.loads(f.read())
    
    # Find the lambda layers
    layers = model_config['config']['layers']
    lambda_layers = [l for l in layers if l['class_name'] == 'Lambda']
    
    print(f"Found {len(lambda_layers)} Lambda layers")
    
    for i, layer in enumerate(lambda_layers):
        print(f"\n=== Lambda {i+1}: {layer['name']} ===")
        function_data = layer['config']['function'][0]
        arguments = layer['config']['arguments']
        print(f"Arguments: {arguments}")
        
        # Try to decode the base64 function
        try:
            decoded = base64.b64decode(function_data)
            print(f"Decoded function size: {len(decoded)} bytes")
            
            # The original function seems to be doing slicing
            # Based on arguments: i=0,1,2 and parts=3
            # Let's recreate this logic
            i = arguments['i']
            parts = arguments['parts']
            
            print(f"This lambda does slice {i} of {parts} parts")
            
        except Exception as e:
            print(f"Failed to decode: {e}")

def create_correct_model():
    """Create model with correct slicing logic"""
    
    import tensorflow as tf
    import keras
    import numpy as np
    
    # Input: (None, 400, 3)
    input_layer = keras.layers.Input(shape=(400, 3), name='conv1d_1_input')
    
    # Correct slicing logic based on the original function
    # The function divides 400 into 3 parts: step = 400 // 3 = 133
    # slice 0: [0:133]   = 133 samples  
    # slice 1: [133:266] = 133 samples
    # slice 2: [266:400] = 134 samples (gets the remainder)
    
    def get_slice_0(x):
        return x[:, 0:133, :]    # First 133 samples
    
    def get_slice_1(x):
        return x[:, 133:266, :]  # Next 133 samples
    
    def get_slice_2(x):
        return x[:, 266:400, :]  # Last 134 samples
    
    lambda_1 = keras.layers.Lambda(get_slice_0, name='lambda_1')(input_layer)
    lambda_2 = keras.layers.Lambda(get_slice_1, name='lambda_2')(input_layer) 
    lambda_3 = keras.layers.Lambda(get_slice_2, name='lambda_3')(input_layer)
    
    print(f"Slice shapes will be: {133}, {133}, {134}")
    
    # Since the slices have different sizes, we need separate CNNs or padding
    # Let's create flexible CNNs that can handle different input sizes
    
    def create_flexible_cnn(name_suffix=""):
        """Create CNN that can handle variable input sizes"""
        return keras.Sequential([
            keras.layers.Conv1D(32, 21, padding='same', activation='linear', name=f'conv1d_1{name_suffix}'),
            keras.layers.BatchNormalization(name=f'batch_normalization_1{name_suffix}'),
            keras.layers.Activation('relu', name=f'activation_1{name_suffix}'),
            keras.layers.MaxPooling1D(2, name=f'max_pooling1d_1{name_suffix}'),
            
            keras.layers.Conv1D(64, 15, padding='same', activation='linear', name=f'conv1d_2{name_suffix}'),
            keras.layers.BatchNormalization(name=f'batch_normalization_2{name_suffix}'),
            keras.layers.Activation('relu', name=f'activation_2{name_suffix}'),
            keras.layers.MaxPooling1D(2, name=f'max_pooling1d_2{name_suffix}'),
            
            keras.layers.Conv1D(128, 11, padding='same', activation='linear', name=f'conv1d_3{name_suffix}'),
            keras.layers.BatchNormalization(name=f'batch_normalization_3{name_suffix}'),
            keras.layers.Activation('relu', name=f'activation_3{name_suffix}'),
            keras.layers.MaxPooling1D(2, name=f'max_pooling1d_3{name_suffix}'),
            
            keras.layers.Conv1D(256, 9, padding='same', activation='linear', name=f'conv1d_4{name_suffix}'),
            keras.layers.BatchNormalization(name=f'batch_normalization_4{name_suffix}'),
            keras.layers.Activation('relu', name=f'activation_4{name_suffix}'),
            keras.layers.MaxPooling1D(2, name=f'max_pooling1d_4{name_suffix}'),
            
            keras.layers.GlobalAveragePooling1D(),  # This handles variable sizes
            keras.layers.Dense(200, activation='linear', name=f'dense_1{name_suffix}'),
            keras.layers.BatchNormalization(name=f'batch_normalization_5{name_suffix}'),
            keras.layers.Activation('relu', name=f'activation_5{name_suffix}'),
            keras.layers.Dense(200, activation='linear', name=f'dense_2{name_suffix}'),
            keras.layers.BatchNormalization(name=f'batch_normalization_6{name_suffix}'),
            keras.layers.Activation('relu', name=f'activation_6{name_suffix}'),
            keras.layers.Dense(3, activation='linear', name=f'dense_3{name_suffix}'),
            keras.layers.Activation('softmax', name=f'activation_7{name_suffix}')
        ])
    
    # But wait - the original model uses the SAME Sequential model for all 3 inputs
    # So the architecture is actually shared weights!
    
    # Create single CNN with shared weights
    shared_cnn = create_flexible_cnn()
    
    # Apply to all three slices (shared weights)
    out1 = shared_cnn(lambda_1)
    out2 = shared_cnn(lambda_2) 
    out3 = shared_cnn(lambda_3)
    
    # The original concatenates along axis=0 (batch dimension)
    # This is unusual but might be how they handle multiple predictions
    outputs = keras.layers.Concatenate(axis=-1)([out1, out2, out3])  # Try axis=-1 instead
    
    model = keras.Model(inputs=input_layer, outputs=outputs)
    
    return model

if __name__ == "__main__":
    decode_lambda_functions()
    print("\n" + "="*60)
    print("Creating corrected model...")
    try:
        model = create_correct_model()
        print("✅ Model created successfully")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
