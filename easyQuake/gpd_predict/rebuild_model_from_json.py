"""
Rebuild model from JSON by replacing marshalled Lambda functions with a local `get_slice` implementation.
This attempts to recreate the original functional graph:
  Input -> lambda_i (get_slice i) -> shared sequential_1 -> concatenate -> output

Usage:
  python gpd_predict/rebuild_model_from_json.py \\
    --json gpd_predict/model_pol.json \\
    --weights gpd_predict/model_pol_legacy_fixed.h5 \\
    --output gpd_predict/model_pol_legacy_fixed_rebuilt.h5

If successful, produce a rebuilt .h5 that can be converted to `.keras` with the main converter.
"""
import json
import argparse
import os
import traceback

import tensorflow as tf


def get_slice(x, i, parts):
    # Slice along the batch dimension into `parts` parts and return the i-th part.
    # Handles dynamic batch sizes; last part takes the remainder.
    shape = tf.shape(x)
    batch = shape[0]
    # floor division
    part = batch // parts
    start = i * part
    # last part takes remainder
    def last():
        size0 = batch - start
        return tf.slice(x, [start, 0, 0], tf.stack([size0, shape[1], shape[2]]))
    def middle():
        size0 = part
        return tf.slice(x, [start, 0, 0], tf.stack([size0, shape[1], shape[2]]))
    return tf.cond(tf.equal(i, parts - 1), last, middle)


def find_sequential_config(model_json):
    # model_json is the top-level JSON dict
    layers = model_json.get('config', {}).get('layers', [])
    print(f'Searching for sequential_1 in {len(layers)} layers')
    
    # Debug: print layer names and class names
    for i, layer in enumerate(layers):
        if isinstance(layer, dict):
            config = layer.get('config', {})
            class_name = layer.get('class_name', 'unknown')
            
            if isinstance(config, dict):
                name = config.get('name', 'unnamed')
                print(f'  Layer {i}: {class_name} name={name}')
                
                # Check if this is the sequential layer we want
                if class_name == 'Sequential' and name == 'sequential_1':
                    print(f'Found sequential_1 at layer {i}')
                    return config
            elif isinstance(config, list):
                print(f'  Layer {i}: {class_name} (config is list with {len(config)} items)')
                # For Sequential layers, the config might be structured differently
                # Let's return the whole layer dict and handle it in the caller
                if class_name == 'Sequential':
                    print(f'Found Sequential layer at {i}, returning full config')
                    return layer  # Return the whole layer, not just config
            else:
                print(f'  Layer {i}: {class_name} (config is {type(config).__name__})')
        else:
            print(f'  Layer {i}: not a dict, type={type(layer).__name__}')
    
    print('sequential_1 not found in top-level layers')
    return None


def build_model_from_json(json_path, weights_path, output_path, parts=3):
    with open(json_path,'r') as f:
        mj = json.load(f)
    seq_layer = find_sequential_config(mj)
    if seq_layer is None:
        print('Could not find sequential layer in JSON')
        return False
    
    # Handle different config formats
    if 'config' in seq_layer and isinstance(seq_layer['config'], dict):
        # Standard format: config is a dict
        seq_conf = seq_layer['config']
    else:
        # Alternative format: config is a list of layers
        seq_conf = seq_layer
    
    # Build the sequential model manually from the layer list
    try:
        # Extract the layers from the config
        if 'config' in seq_conf and isinstance(seq_conf['config'], list):
            layers_config = seq_conf['config']
        elif isinstance(seq_conf, dict) and 'config' in seq_conf:
            layers_config = seq_conf['config']
        else:
            print(f'Unexpected sequential config format: {type(seq_conf)}')
            return False
            
        print(f'Building sequential model from {len(layers_config)} layers')
        
        # Build Sequential model manually
        seq_model = tf.keras.Sequential(name='sequential_1')
        
        # Add each layer
        for i, layer_config in enumerate(layers_config):
            layer_class_name = layer_config['class_name']
            layer_cfg = layer_config['config'].copy()  # Make a copy to avoid modifying original
            
            # Remove common parameters that are not constructor arguments
            layer_cfg.pop('batch_input_shape', None)  # Only for InputLayer
            layer_cfg.pop('dtype', None)  # Handled automatically
            layer_cfg.pop('trainable', None)  # Can be set after creation
            
            print(f'  Adding layer {i}: {layer_class_name}')
            
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
        
        print('Built sequential submodel; summary:')
        # Build the model to get a summary
        seq_model.build((None, 400, 3))  # Build with expected input shape
        seq_model.summary()
        
    except Exception as e:
        print('Failed to build sequential submodel:')
        traceback.print_exc()
        return False

    # Build functional model: Input -> 3 slices -> apply seq_model -> concat
    input_shape = None
    # find input layer config
    for layer in mj.get('config',{}).get('layers',[]):
        if layer.get('class_name') == 'InputLayer':
            cfg = layer.get('config',{})
            batch_input_shape = cfg.get('batch_input_shape')
            if batch_input_shape:
                # ignore batch dim
                input_shape = tuple(batch_input_shape[1:])
                break
    if input_shape is None:
        print('Could not determine input shape from JSON')
        return False

    inp = tf.keras.Input(shape=input_shape, name='conv1d_1_input')
    slices = []
    for i in range(parts):
        # use a lambda layer that captures i
        lam = tf.keras.layers.Lambda(lambda x, idx=i: get_slice(x, idx, parts), output_shape=input_shape, name=f'lambda_{i+1}')(inp)
        slices.append(lam)

    # Apply the same seq_model to each slice. seq_model is a Model; calling it will reuse weights.
    seq_outputs = [seq_model(s) for s in slices]

    # Concatenate along axis=0 (as in the original JSON)
    try:
        concat = tf.keras.layers.Concatenate(axis=0, name='activation_7')(seq_outputs)
    except Exception:
        # If axis=0 fails, try axis=-1 as fallback
        concat = tf.keras.layers.Concatenate(axis=-1, name='activation_7')(seq_outputs)

    model = tf.keras.Model(inputs=inp, outputs=concat)
    print('Rebuilt functional model; summary:')
    model.summary()

    # Attempt to load weights by name
    try:
        print('Attempting to load weights by_name=True from', weights_path)
        model.load_weights(weights_path, by_name=True)
        print('Weights loaded (by_name=True).')
    except Exception:
        print('by_name load failed; trying direct load...')
        try:
            model.load_weights(weights_path)
            print('Weights loaded (direct).')
        except Exception:
            print('Failed to load weights:')
            traceback.print_exc()
            return False

    # Save rebuilt model to output_path
    try:
        print('Saving rebuilt model to', output_path)
        
        # Save as .keras if the output path has that extension
        if output_path.endswith('.keras'):
            model.save(output_path, save_format='keras')
        else:
            model.save(output_path)
            
        print('Saved. exists:', os.path.exists(output_path), 'size:', os.path.getsize(output_path) if os.path.exists(output_path) else 'N/A')
        return True
    except Exception:
        print('Failed to save rebuilt model:')
        traceback.print_exc()
        return False


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    ok = build_model_from_json(a.json, a.weights, a.output)
    if not ok:
        print('Rebuild failed')
        exit(2)
    else:
        print('Rebuild succeeded')
