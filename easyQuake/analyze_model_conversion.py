#!/usr/bin/env python3
"""Identify exactly what changed in the model conversion."""

import numpy as np
import tensorflow as tf
import os

def inspect_model_details():
    """Inspect the converted model to understand what changed."""
    
    print("=== MODEL INSPECTION ===")
    
    # Load the converted model
    model_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
    model = tf.keras.models.load_model(model_path)
    
    print(f"Model loaded from: {model_path}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Print model summary
    print("\n=== MODEL ARCHITECTURE ===")
    model.summary()
    
    # Check the last few layers
    print("\n=== FINAL LAYERS ===")
    for i, layer in enumerate(model.layers[-5:]):
        print(f"Layer {len(model.layers)-5+i}: {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'activation'):
            print(f"  Activation: {layer.activation}")
        if hasattr(layer, 'units'):
            print(f"  Units: {layer.units}")
    
    # Check if final layer has softmax
    final_layer = model.layers[-1]
    print(f"\nFinal layer: {final_layer.name}")
    print(f"Final layer type: {final_layer.__class__.__name__}")
    if hasattr(final_layer, 'activation'):
        print(f"Final activation: {final_layer.activation}")
    
    # Test with some data to see raw outputs
    print("\n=== RAW MODEL OUTPUTS ===")
    
    # Create test input (batch_size=3, time_steps=400, channels=3)
    test_input = np.random.randn(3, 400, 3).astype('float32')
    
    # Get outputs
    outputs = model.predict(test_input, verbose=0)
    print(f"Output shape: {outputs.shape}")
    print(f"Output dtype: {outputs.dtype}")
    
    # Check if outputs sum to 1 (indicating proper softmax)
    row_sums = np.sum(outputs, axis=1)
    print(f"Row sums (should be ~1.0 for softmax): {row_sums}")
    print(f"Row sums min/max: {np.min(row_sums):.6f} / {np.max(row_sums):.6f}")
    
    print(f"\nSample outputs:")
    for i in range(3):
        print(f"  Sample {i}: P={outputs[i,0]:.6f}, S={outputs[i,1]:.6f}, N={outputs[i,2]:.6f}, sum={np.sum(outputs[i]):.6f}")
    
    # Check for any numerical issues
    print(f"\nOutput statistics:")
    print(f"  Min value: {np.min(outputs):.6f}")
    print(f"  Max value: {np.max(outputs):.6f}")
    print(f"  Mean value: {np.mean(outputs):.6f}")
    print(f"  Std dev: {np.std(outputs):.6f}")
    
    return model, outputs

def test_with_real_data():
    """Test the model with actual seismic data to see probability ranges."""
    
    print("\n" + "="*60)
    print("TESTING WITH REAL SEISMIC DATA")
    print("="*60)
    
    import obspy
    
    # Load real data
    try:
        st = obspy.Stream()
        st += obspy.read('/home/jwalter/easyQuake/tests/O2.WILZ.EHN.mseed')
        st += obspy.read('/home/jwalter/easyQuake/tests/O2.WILZ.EHE.mseed') 
        st += obspy.read('/home/jwalter/easyQuake/tests/O2.WILZ.EHZ.mseed')
        
        print(f"Loaded {len(st)} traces")
        
        # Simple preprocessing
        st.filter('highpass', freq=3.0, corners=2, zerophase=True)
        st.filter('lowpass', freq=20.0, corners=2, zerophase=True)
        st.detrend('demean')
        st.detrend('linear')
        
        # Get a small sample for testing
        n_samples = 1000
        sample_data = np.zeros((1, 400, 3))  # 1 window
        
        for i, tr in enumerate(st[:3]):
            if len(tr.data) >= 400:
                sample_data[0, :, i] = tr.data[:400]
        
        # Normalize the sample
        sample_data = sample_data / np.max(np.abs(sample_data))
        
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data range: {np.min(sample_data):.6f} to {np.max(sample_data):.6f}")
        
        # Load model and predict
        model_path = '/home/jwalter/easyQuake/easyQuake/gpd_predict/model_pol_new.keras'
        model = tf.keras.models.load_model(model_path)
        
        outputs = model.predict(sample_data, verbose=0)
        
        print(f"\nReal data prediction:")
        print(f"  P-wave: {outputs[0,0]:.6f}")
        print(f"  S-wave: {outputs[0,1]:.6f}")
        print(f"  Noise:  {outputs[0,2]:.6f}")
        print(f"  Sum:    {np.sum(outputs[0]):.6f}")
        
        # Test multiple windows
        if len(st[0].data) >= 4000:
            multi_data = np.zeros((10, 400, 3))
            for w in range(10):
                start_idx = w * 400
                for i, tr in enumerate(st[:3]):
                    multi_data[w, :, i] = tr.data[start_idx:start_idx+400]
            
            # Normalize each window
            for w in range(10):
                multi_data[w] = multi_data[w] / np.max(np.abs(multi_data[w]))
            
            multi_outputs = model.predict(multi_data, verbose=0)
            
            print(f"\nMultiple window statistics:")
            print(f"  P-wave: min={np.min(multi_outputs[:,0]):.6f}, max={np.max(multi_outputs[:,0]):.6f}, mean={np.mean(multi_outputs[:,0]):.6f}")
            print(f"  S-wave: min={np.min(multi_outputs[:,1]):.6f}, max={np.max(multi_outputs[:,1]):.6f}, mean={np.mean(multi_outputs[:,1]):.6f}")
            print(f"  Noise:  min={np.min(multi_outputs[:,2]):.6f}, max={np.max(multi_outputs[:,2]):.6f}, mean={np.mean(multi_outputs[:,2]):.6f}")
        
    except Exception as e:
        print(f"Error loading real data: {e}")

def compare_with_original_expected_behavior():
    """Compare current behavior with what we expect from original model."""
    
    print("\n" + "="*60)
    print("COMPARISON WITH EXPECTED ORIGINAL BEHAVIOR")
    print("="*60)
    
    print("Original model (from development branch) expectations:")
    print("  - Should produce probabilities close to 0.994+ for strong signals")
    print("  - Should have clear discrimination between signal and noise")
    print("  - Threshold of 0.994 should work for phase detection")
    
    print("\nCurrent converted model observations:")
    print("  - Produces probabilities in 0.30-0.40 range")
    print("  - Still sums to 1.0 (proper softmax)")
    print("  - Requires threshold of ~0.25 to detect phases")
    
    print("\nPossible issues:")
    print("  1. Softmax temperature changed during conversion")
    print("  2. Different numerical precision affecting final layer")
    print("  3. Batch normalization parameters changed")
    print("  4. Different TensorFlow/Keras version behavior")
    
    print("\nTo investigate:")
    print("  - Check if original model had temperature scaling")
    print("  - Compare layer-by-layer outputs between models")
    print("  - Check if final dense layer weights/biases are different")

if __name__ == "__main__":
    print("GPD Model Conversion Analysis")
    print("="*60)
    
    # Inspect model architecture and outputs
    model, outputs = inspect_model_details()
    
    # Test with real data
    test_with_real_data()
    
    # Compare with expected behavior
    compare_with_original_expected_behavior()
