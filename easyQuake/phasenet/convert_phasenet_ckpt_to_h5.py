"""
Convert PhaseNet TF1 checkpoint to TF2/Keras3 .h5 format.
- Loads TF1 model and restores .ckpt weights
- Builds TF2/Keras3 model
- Transfers weights by name/shape
- Saves as .h5 for use in modern pipelines

Run with: python3 convert_phasenet_ckpt_to_h5.py
"""

import os
import numpy as np
import tensorflow as tf
import logging
import sys
import glob

# --- TF1 imports ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phasenet_original', 'phasenet'))
from model import ModelConfig, UNet as UNet_TF1

# --- TF2 imports ---
sys.path.insert(0, os.path.dirname(__file__))
from model_tf2 import ModelConfigTF2, UNet as UNet_TF2

# --- Paths ---
CKPT_DIR = os.path.join('phasenet', 'phasenet_original', 'model', '190703-214543')
TF2_H5 = os.path.join('phasenet', 'model', '190703-214543', 'model_95_converted.h5')

# Find the checkpoint prefix automatically
ckpt_prefixes = glob.glob(os.path.join(CKPT_DIR, '*.ckpt.index'))
if not ckpt_prefixes:
    print(f"ERROR: No .ckpt.index files found in {CKPT_DIR}")
    sys.exit(1)
TF1_CKPT = ckpt_prefixes[0][:-6]  # remove '.index' to get prefix
print(f"Using checkpoint: {TF1_CKPT}")

# --- Build TF1 model and restore weights ---
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()
config = ModelConfig()
model_tf1 = UNet_TF1(config=config, mode='pred')
saver = tf.compat.v1.train.Saver()
saver.restore(sess, TF1_CKPT)
print(f"Restored TF1 checkpoint: {TF1_CKPT}")

# --- Extract TF1 weights ---
tf1_vars = tf.compat.v1.global_variables()
tf1_weights = sess.run(tf1_vars)

tf1_var_dict = {v.name: w for v, w in zip(tf1_vars, tf1_weights)}
tf1_names = list(tf1_var_dict.keys())
tf1_values = list(tf1_var_dict.values())

# --- Build TF2 model ---
config2 = ModelConfigTF2()

model_tf2 = UNet_TF2(config2)
model_tf2.keras_model.build((None,)+tuple(config2.X_shape))
# Build a dict of TF2 weights by name

tf2_weights = {}
for w in model_tf2.keras_model.model.weights:
    # Remove ':0' for compatibility with TF1 names
    tf2_weights[w.name.replace(':0', '')] = tf.keras.backend.get_value(w)
tf2_weight_names = list(tf2_weights.keys())
tf2_weight_values = list(tf2_weights.values())

# --- Map weights by name/shape ---
# Diagnostic: print all TF2 and TF1 normalized names and shapes side-by-side
print("\nTF2 weights:")
for n, v in zip(tf2_weight_names, tf2_weight_values):
    print(f"  {n:40s} {str(v.shape):>15}")
print("\nTF1 weights:")
for n, v in zip(tf1_names, tf1_values):
    print(f"  {n:40s} {str(v.shape):>15}")


def normalize_name(name, is_tf1=False):
    # Remove trailing ':0'
    name = name.replace(':0', '')
    if is_tf1:
        # Remove variable scope prefix (everything up to and including first '/')
        if '/' in name:
            name = name.split('/', 1)[1]
    return name


tf2_weight_names = list(tf2_weights.keys())
tf2_weight_values = list(tf2_weights.values())
tf2_norm_names = [normalize_name(n, is_tf1=False) for n in tf2_weight_names]

tf1_names = list(tf1_var_dict.keys())
tf1_values = list(tf1_var_dict.values())
tf1_norm_names = [normalize_name(n, is_tf1=True) for n in tf1_names]

mapped_count = 0
matched_tf1 = set()
matched_tf2 = set()
new_weights = list(tf2_weight_values)

# Pass 1: Match by normalized name and shape
for i2, (k2, v2, n2) in enumerate(zip(tf2_weight_names, tf2_weight_values, tf2_norm_names)):
    found = False
    for i1, (k1, v1, n1) in enumerate(zip(tf1_names, tf1_values, tf1_norm_names)):
        if n1 == n2 and v1.shape == v2.shape:
            new_weights[i2] = v1
            print(f"Mapped: {k2} <= {k1}")
            mapped_count += 1
            matched_tf1.add(k1)
            matched_tf2.add(k2)
            found = True
            break
    if not found:
        print(f"No match for: {k2} (shape {v2.shape})")

# Pass 2: For any remaining unmapped TF2 weights, match by unique shape
for i2, (k2, v2) in enumerate(zip(tf2_weight_names, tf2_weight_values)):
    if k2 in matched_tf2:
        continue
    candidates = [(k1, v1) for k1, v1 in zip(tf1_names, tf1_values) if v1.shape == v2.shape and k1 not in matched_tf1]
    if len(candidates) == 1:
        k1, v1 = candidates[0]
        new_weights[i2] = v1
        print(f"Shape-mapped: {k2} <= {k1}")
        mapped_count += 1
        matched_tf1.add(k1)
        matched_tf2.add(k2)

model_tf2.keras_model.model.set_weights(new_weights)
print(f"Mapped {mapped_count} weights from TF1 to TF2 model.")
# Print unmatched TF1 weights
unmatched_tf1 = set(tf1_var_dict.keys()) - matched_tf1
if unmatched_tf1:
    print("\nUnmatched TF1 weights:")
    for k in unmatched_tf1:
        print(f"  {k} (shape {tf1_var_dict[k].shape})")

# --- Save as .h5 ---
out_dir = os.path.dirname(TF2_H5)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
model_tf2.keras_model.model.save(TF2_H5)
print(f"Saved TF2/Keras3 model to: {TF2_H5}")
