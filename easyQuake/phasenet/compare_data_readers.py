#!/usr/bin/env python3
"""Compare normalized waveform output from TF1 and TF2 data readers for a single mseed file.

Usage:
  python phasenet/compare_data_readers.py /path/to/mseed

It will instantiate each package's DataReader_pred and call their read_mseed method, then apply the same normalize_long used by the prediction pipeline and report shape and numerical differences.
"""
import sys
import numpy as np
from pprint import pformat

MSEED = sys.argv[1] if len(sys.argv) > 1 else None
if MSEED is None:
    print("Usage: python phasenet/compare_data_readers.py /path/to/mseed")
    sys.exit(1)

# Ensure project root is on sys.path
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import readers
try:
    from phasenet.data_reader import DataReader_pred as DataReaderTF2, normalize_long as normalize_long_tf2
except Exception as e:
    print("Failed to import TF2 data reader:", e)
    raise

try:
    from phasenet.phasenet_original.phasenet.data_reader import DataReader_pred as DataReaderTF1, normalize_long as normalize_long_tf1
except Exception as e:
    print("Failed to import TF1 data reader:", e)
    raise

print(f"Comparing readers on: {MSEED}")

# Create minimal instances without invoking __init__ (which expects data_list)
r2 = object.__new__(DataReaderTF2)
r2.sampling_rate = 100
r2.highpass_filter = 0.0
r1 = object.__new__(DataReaderTF1)
r1.sampling_rate = 100
r1.highpass_filter = 0.0

# Both readers expose read_mseed(fname, ...); call as unbound methods on our minimal instances
meta2 = DataReaderTF2.read_mseed(r2, MSEED, response=None, highpass_filter=0.0, sampling_rate=r2.sampling_rate)
meta1 = DataReaderTF1.read_mseed(r1, MSEED, response=None, highpass_filter=0.0, sampling_rate=r1.sampling_rate)

if 'data' not in meta2:
    print('TF2 reader returned no data')
    sys.exit(2)
if 'data' not in meta1:
    print('TF1 reader returned no data')
    sys.exit(2)

arr2 = meta2['data']
arr1 = meta1['data']
print('raw shapes: TF2', arr2.shape, 'TF1', arr1.shape)

# Apply normalize_long (used by DataReader_pred)
try:
    s2 = normalize_long_tf2(arr2)
except Exception as e:
    print('TF2 normalize_long failed:', e)
    s2 = arr2

try:
    s1 = normalize_long_tf1(arr1)
except Exception as e:
    print('TF1 normalize_long failed:', e)
    s1 = arr1

print('normalized shapes: TF2', s2.shape, 'TF1', s1.shape)

# Reduce to comparable shapes (truncate to min time length)
min_t = min(s1.shape[0], s2.shape[0])
s1c = s1[:min_t]
s2c = s2[:min_t]

# Stats
diff = s1c - s2c
max_abs = np.max(np.abs(diff))
mean_abs = np.mean(np.abs(diff))
print('\nNumeric comparison:')
print('max_abs_diff =', max_abs)
print('mean_abs_diff =', mean_abs)

# Per-channel stats
for ch in range(s1c.shape[-1]):
    chdiff = np.abs(diff[..., ch])
    print(f'channel {ch}: max={chdiff.max():.6g}, mean={chdiff.mean():.6g}')

# Save arrays for manual inspection
np.save('reader_tf1_norm.npy', s1c)
np.save('reader_tf2_norm.npy', s2c)
print('\nSaved normalized arrays as reader_tf1_norm.npy and reader_tf2_norm.npy')
print('If differences are non-zero, inspect these arrays or plot a few channels to see where they differ.')
