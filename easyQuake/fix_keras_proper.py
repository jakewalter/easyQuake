#!/usr/bin/env python3
"""
Proper fix for TensorFlow/Keras 3 compatibility.
Convert TF1 layer calls to functional Keras calls.
"""

import re

def fix_keras_functional(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Fix Conv2D calls - convert from TF1 style to functional Keras style
    # Pattern: tf.keras.layers.Conv2D(net, filters=..., ...)
    # Should be: tf.keras.layers.Conv2D(filters=..., ...)(net)
    
    # Multi-line Conv2D pattern
    pattern = r'net = tf\.keras\.layers\.Conv2D\(net,\s*\n\s*filters=([^,]+),\s*\n\s*kernel_size=([^,]+),\s*\n\s*activation=([^,]+),\s*\n\s*([^)]*)\)'
    replacement = r'net = tf.keras.layers.Conv2D(\n                   filters=\1,\n                   kernel_size=\2,\n                   activation=\3,\n                   \4)(net)'
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Single line Conv2D pattern  
    pattern = r'net = tf\.keras\.layers\.Conv2D\(net,([^)]+)\)'
    replacement = r'net = tf.keras.layers.Conv2D(\1)(net)'
    content = re.sub(pattern, replacement, content)
    
    # Conv2DTranspose calls
    pattern = r'net = tf\.keras\.layers\.Conv2DTranspose\(net,\s*\n\s*filters=([^,]+),\s*\n\s*kernel_size=([^,]+),\s*\n\s*([^)]*)\)'
    replacement = r'net = tf.keras.layers.Conv2DTranspose(\n                         filters=\1,\n                         kernel_size=\2,\n                         \3)(net)'
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Applied proper Keras functional fix to {filename}")

if __name__ == "__main__":
    # First restore original
    import subprocess
    subprocess.run(['cp', 'phasenet/phasenet_original/phasenet/model.py', 'phasenet/model.py'])
    
    # Apply minimal conv2d replacement first
    with open("phasenet/model.py", 'r') as f:
        content = f.read()
    content = content.replace('tf.compat.v1.layers.conv2d(', 'tf.keras.layers.Conv2D(')
    content = content.replace('tf.compat.v1.layers.conv2d_transpose(', 'tf.keras.layers.Conv2DTranspose(')
    with open("phasenet/model.py", 'w') as f:
        f.write(content)
    
    # Now apply functional fixes
    fix_keras_functional("phasenet/model.py")
