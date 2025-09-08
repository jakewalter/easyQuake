#!/usr/bin/env python3
"""
Fix batch_normalization and dropout compatibility.
"""

import re

def fix_remaining_layers(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Fix batch_normalization calls
    content = content.replace('tf.compat.v1.layers.batch_normalization(', 'tf.keras.layers.BatchNormalization(')
    
    # Fix dropout calls  
    content = content.replace('tf.compat.v1.layers.dropout(', 'tf.keras.layers.Dropout(')
    
    # Fix the functional calling pattern for BatchNormalization
    # From: tf.keras.layers.BatchNormalization(net, training=is_training, name="...")
    # To: tf.keras.layers.BatchNormalization(name="...")(net, training=is_training)
    pattern = r'tf\.keras\.layers\.BatchNormalization\(net,\s*training=([^,]+),\s*name="([^"]+)"\)'
    replacement = r'tf.keras.layers.BatchNormalization(name="\2")(net, training=\1)'
    content = re.sub(pattern, replacement, content)
    
    # Fix the functional calling pattern for Dropout  
    # From: tf.keras.layers.Dropout(net, rate=rate, training=is_training, name="...")
    # To: tf.keras.layers.Dropout(rate=rate, name="...")(net, training=is_training)
    pattern = r'tf\.keras\.layers\.Dropout\(net,\s*rate=([^,]+),\s*training=([^,]+),\s*name="([^"]+)"\)'
    replacement = r'tf.keras.layers.Dropout(rate=\1, name="\3")(net, training=\2)'
    content = re.sub(pattern, replacement, content)
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Fixed remaining layer compatibility in {filename}")

if __name__ == "__main__":
    fix_remaining_layers("phasenet/model.py")
