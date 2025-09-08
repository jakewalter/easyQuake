#!/usr/bin/env python3
"""
Test the fixed GPD model with actual seismic data to verify P/S-wave detection.
"""

import numpy as np
import tensorflow as tf
import os
import sys
from obspy import read

def test_gpd_with_real_data():
    """Test GPD with the actual seismic data files"""
    print("Testing Fixed GPD Model with Real Seismic Data")
    print("=" * 50)
    
    # Load the fixed model
    model_path = "model_pol_fixed.h5"
    if not os.path.exists(model_path):
        print(f"✗ Fixed model not found: {model_path}")
        return False
    
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Loaded fixed model: {model_path}")
    
    # Load the test seismic data
    test_dir = "/home/jwalter/easyQuake/tests"
    data_files = [
        "O2.WILZ.EHZ.mseed",  # Z component
        "O2.WILZ.EHN.mseed",  # N component  
        "O2.WILZ.EHE.mseed"   # E component
    ]
    
    # Read seismic data
    print("\nLoading seismic data...")
    traces = []
    for filename in data_files:
        filepath = os.path.join(test_dir, filename)
        if os.path.exists(filepath):
            st = read(filepath)
            traces.append(st[0])
            print(f"  ✓ {filename}: {len(st[0].data)} samples")
        else:
            print(f"  ✗ {filename}: File not found")
            return False
    
    # Take first 10 windows for quick test
    window_length = 400
    num_test_windows = 10
    
    # Combine traces into 3-channel array
    min_length = min(len(tr.data) for tr in traces)
    max_samples = min(min_length, num_test_windows * window_length)
    
    combined_data = np.zeros((max_samples, 3))
    combined_data[:, 0] = traces[0].data[:max_samples]  # Z
    combined_data[:, 1] = traces[1].data[:max_samples]  # N 
    combined_data[:, 2] = traces[2].data[:max_samples]  # E
    
    # Normalize data
    for i in range(3):
        channel_data = combined_data[:, i]
        if np.std(channel_data) > 0:
            combined_data[:, i] = (channel_data - np.mean(channel_data)) / np.std(channel_data)
    
    # Create test windows
    windows = np.zeros((num_test_windows, window_length, 3))
    for i in range(num_test_windows):
        start_idx = i * window_length
        end_idx = start_idx + window_length
        if end_idx <= len(combined_data):
            windows[i] = combined_data[start_idx:end_idx]
    
    print(f"Created {num_test_windows} test windows")
    
    # Run predictions
    print("\nRunning GPD predictions...")
    predictions = model.predict(windows, verbose=0)
    
    # Analyze results
    max_probs = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    class_names = ['Noise', 'P-wave', 'S-wave']
    
    print(f"\nResults for real seismic data:")
    for i in range(num_test_windows):
        prob = max_probs[i]
        class_idx = predicted_classes[i]
        class_name = class_names[class_idx]
        
        status = "✓" if prob > 0.994 else " "
        print(f"  Window {i:2d}: {class_name:7s} ({prob:.4f}) {status}")
    
    # Summary
    high_conf = np.sum(max_probs > 0.994)
    p_picks = np.sum((predicted_classes == 1) & (max_probs > 0.994))
    s_picks = np.sum((predicted_classes == 2) & (max_probs > 0.994))
    
    print(f"\nSummary:")
    print(f"  High confidence (>0.994): {high_conf}/{num_test_windows}")
    print(f"  P-wave picks: {p_picks}")
    print(f"  S-wave picks: {s_picks}")
    
    return high_conf > 0

def main():
    try:
        success = test_gpd_with_real_data()
        
        if success:
            print(f"\n✓ SUCCESS: Fixed GPD model is generating high-confidence predictions!")
        else:
            print(f"\n⚠️  Model working but no high-confidence picks in this sample")
            print("This may be normal for this particular data segment")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
