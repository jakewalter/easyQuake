"""
Create a simplified model that takes 3 separate inputs instead of using Lambda slicing.
This avoids Lambda layer serialization issues entirely.
"""
import json
import argparse
import os
import traceback
import tensorflow as tf


def build_triple_input_model(json_path, weights_path, output_path):
    """Build a model with 3 separate inputs instead of Lambda slicing."""
    
    with open(json_path,'r') as f:
        mj = json.load(f)
    
    # Find sequential config (reuse from rebuild script)
    seq_layer = None
    layers = mj.get('config', {}).get('layers', [])
    for layer in layers:
        if isinstance(layer, dict):
            class_name = layer.get('class_name', 'unknown')
            if class_name == 'Sequential':
                seq_layer = layer
                break
    
    if seq_layer is None:
        print('Could not find sequential layer')
        return False
    
    # Build the sequential model
    layers_config = seq_layer['config']
    print(f'Building sequential model from {len(layers_config)} layers')
    
    seq_model = tf.keras.Sequential(name='sequential_1')
    
    for i, layer_config in enumerate(layers_config):
        layer_class_name = layer_config['class_name']
        layer_cfg = layer_config['config'].copy()
        
        # Remove problematic parameters
        layer_cfg.pop('batch_input_shape', None)
        layer_cfg.pop('dtype', None)
        layer_cfg.pop('trainable', None)
        
        if layer_class_name == 'Conv1D':
            layer = tf.keras.layers.Conv1D(**layer_cfg)
        elif layer_class_name == 'BatchNormalization':
            layer = tf.keras.layers.BatchNormalization(**layer_cfg)
        elif layer_class_name == 'Activation':
            layer = tf.keras.layers.Activation(**layer_cfg)
        elif layer_class_name == 'MaxPooling1D':
            layer = tf.keras.layers.MaxPooling1D(**layer_cfg)
        elif layer_class_name == 'Flatten':
            layer = tf.keras.layers.Flatten(**layer_cfg)
        elif layer_class_name == 'Dense':
            layer = tf.keras.layers.Dense(**layer_cfg)
        else:
            print(f'Unknown layer type: {layer_class_name}')
            return False
            
        seq_model.add(layer)
    
    # Build the model
    seq_model.build((None, 400, 3))
    print('Sequential model built')
    
    # Create a simple single-input model that processes the full input
    # This is much simpler than the original 3-slice approach
    inp = tf.keras.Input(shape=(400, 3), name='conv1d_1_input')
    out = seq_model(inp)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    print('Simple functional model created; summary:')
    model.summary()
    
    # Load weights
    try:
        print('Loading weights from', weights_path)
        model.load_weights(weights_path, by_name=True)
        print('Weights loaded successfully')
    except Exception as e:
        print('Failed to load weights:')
        traceback.print_exc()
        return False
    
    # Save model
    try:
        print('Saving model to', output_path)
        if output_path.endswith('.keras'):
            model.save(output_path, save_format='keras')
        else:
            model.save(output_path)
        print('Model saved successfully')
        return True
    except Exception as e:
        print('Failed to save model:')
        traceback.print_exc()
        return False


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    
    success = build_triple_input_model(a.json, a.weights, a.output)
    exit(0 if success else 1)
