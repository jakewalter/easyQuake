#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from obspy import read
from obspy.signal.trigger import trigger_onset

# Load the calibrated model
model = tf.keras.models.load_model('model_pol_gpd_calibrated_F80.h5')
print("Loaded F80 calibrated model")

# Load test data
st = read('testdata/XC_POX_2020_013_033.mseed')
print(f"Loaded data: {len(st)} traces")

for tr in st:
    tr.detrend(type='linear')
    tr.filter('bandpass', freqmin=3.0, freqmax=20.0, corners=2, zerophase=True) 
    if tr.stats.sampling_rate != 100:
        tr.resample(100)

# Process first trace
tr = st[0]
data = tr.data
print(f"Data shape: {data.shape}")

# Create sliding window (same as GPD)
window_size = 400
step_size = 10
windows = []

for i in range(0, len(data) - window_size + 1, step_size):
    windows.append(data[i:i + window_size])

tr_win = np.array(windows)
print(f"Windows shape: {tr_win.shape}")

# Add channel dimension (single channel data gets expanded to 3 channels)
if tr_win.ndim == 2:
    tr_win = np.expand_dims(tr_win, axis=2)
    tr_win = np.repeat(tr_win, 3, axis=2)

print(f"Final input shape: {tr_win.shape}")

# Apply GPD normalization
tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
print(f"After normalization - min: {np.min(tr_win):.4f}, max: {np.max(tr_win):.4f}")

# Predict
predictions = model.predict(tr_win)
print(f"Predictions shape: {predictions.shape}")

# Extract probabilities
prob_P = predictions[:, 1]  # P-wave probability
prob_S = predictions[:, 2]  # S-wave probability

print(f"P-wave probabilities - min: {np.min(prob_P):.4f}, max: {np.max(prob_P):.4f}")
print(f"S-wave probabilities - min: {np.min(prob_S):.4f}, max: {np.max(prob_S):.4f}")

# Count high probability predictions
threshold = 0.994
p_high = np.sum(prob_P > threshold)
s_high = np.sum(prob_S > threshold)
print(f"P-wave predictions > {threshold}: {p_high}")
print(f"S-wave predictions > {threshold}: {s_high}")

# Trigger detection
try:
    p_triggers = trigger_onset(prob_P, threshold, 0.1)
    s_triggers = trigger_onset(prob_S, threshold, 0.1)
    print(f"P-wave triggers: {len(p_triggers)}")
    print(f"S-wave triggers: {len(s_triggers)}")
    print(f"Total picks: {len(p_triggers) + len(s_triggers)}")
except Exception as e:
    print(f"Trigger detection failed: {e}")
