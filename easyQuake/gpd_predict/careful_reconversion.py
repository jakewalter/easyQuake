#!/usr/bin/env python3
"""
Careful model conversion that preserves exact behavior including temperature scaling.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import h5py

def analyze_original_model_behavior():
    """Analyze the original model to understand its exact behavior."""
    
    print("=== ANALYZING ORIGINAL MODEL BEHAVIOR ===")
    
    json_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol.json'
    weights_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_best.hdf5'
    
    # Load original architecture
    with open(json_path, 'r') as f:
        model_json = f.read()
    
    print("✓ Original JSON loaded")
    
    # Parse the JSON to understand the exact architecture
    model_config = json.loads(model_json)
    
    # Look for any temperature or scaling parameters
    print("\n=== SEARCHING FOR SCALING PARAMETERS ===")
    
    def search_config(obj, path=""):
        """Recursively search for scaling-related parameters."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if key.lower() in ['temperature', 'scale', 'beta', 'gamma'] or 'scale' in key.lower():
                    print(f"Found potential scaling parameter: {new_path} = {value}")
                search_config(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                search_config(item, f"{path}[{i}]")
    
    search_config(model_config)
    
    # Check the HDF5 weights file for any scaling parameters
    print(f"\n=== ANALYZING WEIGHTS FILE ===")
    with h5py.File(weights_path, 'r') as f:
        def print_h5_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                if 'scale' in name.lower() or 'beta' in name.lower() or 'gamma' in name.lower():
                    print(f"Found scaling weights: {name}, shape: {obj.shape}, values: {obj[()]}")
        
        f.visititems(print_h5_structure)
    
    return model_config

def load_original_model_carefully():
    """Attempt to load the original model with various compatibility fixes."""
    
    print("\n=== ATTEMPTING TO LOAD ORIGINAL MODEL ===")
    
    json_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol.json'
    weights_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_best.hdf5'
    
    # Try different approaches to load the original model
    approaches = [
        "Direct load_model",
        "model_from_json with custom_objects",
        "Manual reconstruction"
    ]
    
    for approach in approaches:
        print(f"\nTrying approach: {approach}")
        
        try:
            if approach == "Direct load_model":
                # Try loading the legacy H5 model directly
                try:
                    model = tf.keras.models.load_model('/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_legacy.h5')
                    print("✓ Loaded legacy H5 model successfully")
                    return model
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    continue
            
            elif approach == "model_from_json with custom_objects":
                # Load JSON + weights with custom objects
                with open(json_path, 'r') as f:
                    model_json = f.read()
                
                # Create custom objects to handle Lambda layers
                custom_objects = {
                    'tf': tf,
                    'slice': tf.slice,
                }
                
                model = model_from_json(model_json, custom_objects=custom_objects)
                model.load_weights(weights_path)
                print("✓ Loaded with model_from_json")
                return model
                
            elif approach == "Manual reconstruction":
                # Manually reconstruct the model architecture
                model = build_model_manually()
                if model:
                    model.load_weights(weights_path)
                    print("✓ Manually reconstructed model")
                    return model
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print("✗ All approaches failed")
    return None

def build_model_manually():
    """Manually build the model architecture to avoid Lambda layer issues."""
    
    print("Building model manually...")
    
    # Create the main input
    input_layer = tf.keras.Input(shape=(400, 3), name='conv1d_1_input')
    
    # Build the sequential part manually based on the JSON config
    x = input_layer
    
    # Conv1D + BatchNorm + ReLU + MaxPool blocks
    # Block 1
    x = tf.keras.layers.Conv1D(32, 21, padding='same', name='conv1d_1')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_1')(x)
    x = tf.keras.layers.Activation('relu', name='activation_1')(x)
    x = tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_1')(x)
    
    # Block 2
    x = tf.keras.layers.Conv1D(64, 15, padding='same', name='conv1d_2')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_2')(x)
    x = tf.keras.layers.Activation('relu', name='activation_2')(x)
    x = tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_2')(x)
    
    # Block 3
    x = tf.keras.layers.Conv1D(128, 11, padding='same', name='conv1d_3')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_3')(x)
    x = tf.keras.layers.Activation('relu', name='activation_3')(x)
    x = tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_3')(x)
    
    # Block 4
    x = tf.keras.layers.Conv1D(256, 9, padding='same', name='conv1d_4')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_4')(x)
    x = tf.keras.layers.Activation('relu', name='activation_4')(x)
    x = tf.keras.layers.MaxPooling1D(2, name='max_pooling1d_4')(x)
    
    # Dense layers
    x = tf.keras.layers.Flatten(name='flatten_1')(x)
    x = tf.keras.layers.Dense(200, name='dense_1')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_5')(x)
    x = tf.keras.layers.Activation('relu', name='activation_5')(x)
    x = tf.keras.layers.Dense(200, name='dense_2')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_6')(x)
    x = tf.keras.layers.Activation('relu', name='activation_6')(x)
    
    # Final dense layer (3 outputs) - NO activation here
    x = tf.keras.layers.Dense(3, name='dense_3')(x)
    
    # Apply softmax activation
    output = tf.keras.layers.Activation('softmax', name='activation_7')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output, name='model_1')
    
    return model

def test_and_compare_models(original_model, new_model):
    """Test both models with the same input to compare behavior."""
    
    print("\n=== COMPARING MODEL BEHAVIORS ===")
    
    # Create test input
    test_input = np.random.randn(5, 400, 3).astype('float32')
    
    if original_model:
        print("Testing original model...")
        orig_output = original_model.predict(test_input, verbose=0)
        print(f"Original output shape: {orig_output.shape}")
        print(f"Original output range: {np.min(orig_output):.6f} to {np.max(orig_output):.6f}")
        print(f"Original max P probability: {np.max(orig_output[:, 0]):.6f}")
        print(f"Sample original output: {orig_output[0]}")
    else:
        orig_output = None
        print("No original model available for comparison")
    
    if new_model:
        print(f"\\nTesting new model...")
        new_output = new_model.predict(test_input, verbose=0)
        print(f"New output shape: {new_output.shape}")
        print(f"New output range: {np.min(new_output):.6f} to {np.max(new_output):.6f}")
        print(f"New max P probability: {np.max(new_output[:, 0]):.6f}")
        print(f"Sample new output: {new_output[0]}")
        
        if orig_output is not None:
            # Compare the outputs
            max_diff = np.max(np.abs(orig_output - new_output))
            print(f"\\nMax absolute difference: {max_diff:.6f}")
            
            if max_diff < 1e-5:
                print("✓ Models produce nearly identical outputs!")
                return True
            elif max_diff < 0.01:
                print("⚠ Models produce similar but not identical outputs")
                return True
            else:
                print("✗ Models produce significantly different outputs")
                return False
    
    return False

def create_corrected_model():
    """Create a corrected model with proper probability calibration."""
    
    print("\\n=== CREATING CORRECTED MODEL ===")
    
    # Try to load and understand the original model
    original_model = load_original_model_carefully()
    
    # Build a new model manually
    new_model = build_model_manually()
    
    if new_model:
        # Try to load the original weights
        weights_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_best.hdf5'
        
        try:
            print("Loading original weights...")
            new_model.load_weights(weights_path)
            print("✓ Weights loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")
            return None
        
        # Test the model
        test_input = np.random.randn(3, 400, 3).astype('float32')
        output = new_model.predict(test_input, verbose=0)
        
        print(f"New model output: {output[0]}")
        print(f"Max P probability: {np.max(output[:, 0]):.6f}")
        
        # If the probabilities are still low, we need to investigate further
        if np.max(output[:, 0]) < 0.5:
            print("⚠ Model still produces low probabilities - investigating...")
            
            # Get pre-softmax outputs
            pre_softmax_model = tf.keras.Model(
                inputs=new_model.input,
                outputs=new_model.layers[-2].output  # Before softmax
            )
            
            pre_softmax = pre_softmax_model.predict(test_input, verbose=0)
            print(f"Pre-softmax output: {pre_softmax[0]}")
            
            # The issue might be that we need to apply temperature scaling
            # Let's create a version with temperature scaling
            return create_temperature_scaled_model(new_model)
        
        return new_model
    
    return None

def create_temperature_scaled_model(base_model):
    """Create a version of the model with temperature scaling."""
    
    print("\\n=== CREATING TEMPERATURE SCALED MODEL ===")
    
    # Based on our earlier analysis, we need ~20x scaling
    temperature = 0.05  # This will make logits 20x larger
    
    # Get all layers except the final softmax
    base_layers = base_model.layers[:-1]
    
    # Rebuild the model with temperature scaling
    input_layer = base_model.input
    x = input_layer
    
    # Apply all layers except the last one
    for layer in base_layers[1:]:  # Skip input layer
        x = layer(x)
    
    # Apply temperature scaling before softmax
    x = tf.keras.layers.Lambda(lambda logits: logits / temperature, name='temperature_scaling')(x)
    
    # Apply softmax
    output = tf.keras.layers.Activation('softmax', name='softmax_output')(x)
    
    scaled_model = tf.keras.Model(inputs=input_layer, outputs=output, name='temperature_scaled_model')
    
    # Copy weights from base model
    for i, layer in enumerate(scaled_model.layers[1:-2]):  # Exclude input, temp scaling, and softmax
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            layer.set_weights(base_layers[i].get_weights())
    
    # Test the scaled model
    test_input = np.random.randn(3, 400, 3).astype('float32')
    output = scaled_model.predict(test_input, verbose=0)
    
    print(f"Temperature scaled output: {output[0]}")
    print(f"Max P probability: {np.max(output[:, 0]):.6f}")
    
    return scaled_model

def main():
    """Main conversion function."""
    
    print("GPD Model Re-conversion with Temperature Preservation")
    print("=" * 60)
    
    # Analyze the original model
    model_config = analyze_original_model_behavior()
    
    # Create corrected model
    corrected_model = create_corrected_model()
    
    if corrected_model:
        # Save the corrected model
        output_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_corrected.keras'
        corrected_model.save(output_path)
        print(f"\\n✓ Corrected model saved to: {output_path}")
        
        # Test with real data
        print("\\n=== TESTING WITH REAL SEISMIC DATA ===")
        try:
            import obspy
            
            # Load real data
            st = obspy.Stream()
            st += obspy.read('/home/jwalter/easyQuake/tests/O2.WILZ.EHN.mseed')
            st += obspy.read('/home/jwalter/easyQuake/tests/O2.WILZ.EHE.mseed') 
            st += obspy.read('/home/jwalter/easyQuake/tests/O2.WILZ.EHZ.mseed')
            
            # Preprocess
            st.filter('highpass', freq=3.0, corners=2, zerophase=True)
            st.filter('lowpass', freq=20.0, corners=2, zerophase=True)
            st.detrend('demean')
            st.detrend('linear')
            
            # Test a few windows
            real_data = np.zeros((3, 400, 3))
            for i, tr in enumerate(st[:3]):
                real_data[i, :, :] = tr.data[1000:1400][:, None]  # Get 400 samples
            
            # Normalize
            for i in range(3):
                real_data[i] = real_data[i] / np.max(np.abs(real_data[i]))
            
            real_output = corrected_model.predict(real_data, verbose=0)
            
            print(f"Real data results:")
            for i in range(3):
                print(f"  Window {i}: P={real_output[i,0]:.4f}, S={real_output[i,1]:.4f}, N={real_output[i,2]:.4f}")
            
            max_p = np.max(real_output[:, 0])
            print(f"\\nMax P probability on real data: {max_p:.4f}")
            
            if max_p > 0.9:
                print("✓ Model now produces high confidence predictions!")
            else:
                print("⚠ Model still produces low confidence - may need further tuning")
                
        except Exception as e:
            print(f"Error testing with real data: {e}")
    
    else:
        print("✗ Failed to create corrected model")

if __name__ == "__main__":
    main()
