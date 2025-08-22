#!/usr/bin/env python3
"""
Minimal fix for TensorFlow/Keras 3 compatibility.
Only replace the specific calls that are failing.
"""

def fix_conv2d_only(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace tf.compat.v1.layers.conv2d with tf.keras.layers.Conv2D in functional style
    # But keep the same calling pattern to minimize disruption
    content = content.replace('tf.compat.v1.layers.conv2d(', 'tf.keras.layers.Conv2D(')
    content = content.replace('tf.compat.v1.layers.conv2d_transpose(', 'tf.keras.layers.Conv2DTranspose(')
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Applied minimal conv2d fix to {filename}")

if __name__ == "__main__":
    fix_conv2d_only("phasenet/model.py")
