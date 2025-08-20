#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from obspy import read
from obspy.signal.trigger import trigger_onset

# List of large models to test (likely the CNN models)
large_models = [
    'model_pol_legacy.h5',
    'model_pol_legacy_fixed.h5', 
    'model_pol_legacy_fixed_full.h5',
    'model_pol_rebuilt_new.h5'
]

# Load test data and prepare it exactly like GPD does
print("Loading and preparing test data...")
st = read('testdata/Z.mseed')  # Using Z channel data

for tr in st:
    tr.detrend(type='linear')
    tr.filter('bandpass', freqmin=3.0, freqmax=20.0, corners=2, zerophase=True) 
    if tr.stats.sampling_rate != 100:
        tr.resample(100)

tr = st[0]
data = tr.data
print(f"Data shape: {data.shape}, duration: {len(data)/100:.1f}s")

# Create sliding window exactly like GPD
window_size = 400
step_size = 10
windows = []

for i in range(0, len(data) - window_size + 1, step_size):
    windows.append(data[i:i + window_size])

tr_win = np.array(windows)
print(f"Windows shape: {tr_win.shape}")

# Add channel dimension (expand single channel to 3 channels)
if tr_win.ndim == 2:
    tr_win = np.expand_dims(tr_win, axis=2)
    tr_win = np.repeat(tr_win, 3, axis=2)

print(f"Final input shape: {tr_win.shape}")

# Apply GPD normalization
tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]

print("\nTesting models:")
print("=" * 60)

results = []
for model_file in large_models:
    if not os.path.exists(model_file):
        print(f"SKIP: {model_file} - not found")
        continue
        
    try:
        print(f"\nTesting: {model_file}")
        model = tf.keras.models.load_model(model_file, compile=False)
        
        # Predict
        predictions = model.predict(tr_win, verbose=0)
        print(f"  Prediction shape: {predictions.shape}")
        
        # Extract P and S probabilities  
        prob_P = predictions[:, 1]
        prob_S = predictions[:, 2]
        
        # Check probability ranges
        print(f"  P-wave: min={np.min(prob_P):.4f}, max={np.max(prob_P):.4f}, mean={np.mean(prob_P):.4f}")
        print(f"  S-wave: min={np.min(prob_S):.4f}, max={np.max(prob_S):.4f}, mean={np.mean(prob_S):.4f}")
        
        # Count high probability detections
        threshold = 0.994
        p_high = np.sum(prob_P > threshold)
        s_high = np.sum(prob_S > threshold)
        print(f"  Detections > {threshold}: P={p_high}, S={s_high}")
        
        # Try lower thresholds to see behavior
        for thresh in [0.9, 0.8, 0.7, 0.5]:
            p_count = np.sum(prob_P > thresh)
            s_count = np.sum(prob_S > thresh)
            total = p_count + s_count
            print(f"  Detections > {thresh}: {total} total (P={p_count}, S={s_count})")
            
        # Trigger detection at 0.994
        try:
            p_triggers = trigger_onset(prob_P, threshold, 0.1)
            s_triggers = trigger_onset(prob_S, threshold, 0.1)
            total_picks = len(p_triggers) + len(s_triggers)
            print(f"  TRIGGER PICKS: {total_picks} (Target: 489)")
            
            results.append((model_file, total_picks, np.max(prob_P), np.max(prob_S)))
            
        except Exception as e:
            print(f"  Trigger detection failed: {e}")
            results.append((model_file, 0, np.max(prob_P), np.max(prob_S)))
            
    except Exception as e:
        print(f"ERROR loading {model_file}: {e}")

print("\n" + "=" * 60)
print("SUMMARY:")
print("Target: 489 picks from reference file")
for model_file, picks, max_p, max_s in results:
    print(f"{model_file:30s}: {picks:4d} picks (P_max:{max_p:.3f}, S_max:{max_s:.3f})")
    
# Find the best model
if results:
    best_model = max(results, key=lambda x: x[1])
    print(f"\nBest model: {best_model[0]} with {best_model[1]} picks")
