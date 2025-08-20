#!/usr/bin/env python3

import os
import sys
import traceback
import numpy as np
import tensorflow as tf
from obspy import read
from obspy.signal.trigger import trigger_onset

# List of models to test
models_to_test = [
    'model_pol_new.keras',
    'model_pol_legacy.h5', 
    'model_pol_legacy_fixed.h5',
    'model_pol_legacy_fixed_full.h5',
    'updated_model.keras',
    'model_pol_new_test.keras'
]

# Load test data
print("Loading test data...")
st = read('testdata/XC_POX_2020_013_033.mseed')

for tr in st:
    tr.detrend(type='linear')
    tr.filter('bandpass', freqmin=3.0, freqmax=20.0, corners=2, zerophase=True) 
    if tr.stats.sampling_rate != 100:
        tr.resample(100)

tr = st[0]
data = tr.data

# Create sliding window
window_size = 400
step_size = 10
windows = []

for i in range(0, len(data) - window_size + 1, step_size):
    windows.append(data[i:i + window_size])

tr_win = np.array(windows)

# Add channel dimension if needed
if tr_win.ndim == 2:
    tr_win = np.expand_dims(tr_win, axis=2)
    tr_win = np.repeat(tr_win, 3, axis=2)

print(f"Input shape: {tr_win.shape}")

# Test each model
results = []

for model_file in models_to_test:
    if not os.path.exists(model_file):
        print(f"SKIP: {model_file} - file not found")
        continue
        
    try:
        print(f"\nTesting {model_file}...")
        model = tf.keras.models.load_model(model_file)
        
        # Apply GPD normalization
        tr_win_norm = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
        
        # Predict
        predictions = model.predict(tr_win_norm, verbose=0)
        
        # Extract probabilities
        prob_P = predictions[:, 1]
        prob_S = predictions[:, 2]
        
        # Count triggers at 0.994 threshold
        try:
            p_triggers = trigger_onset(prob_P, 0.994, 0.1)
            s_triggers = trigger_onset(prob_S, 0.994, 0.1)
            total_picks = len(p_triggers) + len(s_triggers)
        except:
            total_picks = 0
            
        max_p = np.max(prob_P)
        max_s = np.max(prob_S)
        
        print(f"  Max P prob: {max_p:.4f}, Max S prob: {max_s:.4f}")
        print(f"  Total picks at 0.994: {total_picks}")
        
        results.append((model_file, total_picks, max_p, max_s))
        
    except Exception as e:
        print(f"ERROR with {model_file}: {e}")

print(f"\nSUMMARY:")
print(f"Target: 489 picks (reference)")
for model_file, picks, max_p, max_s in results:
    print(f"{model_file:30s}: {picks:4d} picks (max P:{max_p:.3f}, max S:{max_s:.3f})")
