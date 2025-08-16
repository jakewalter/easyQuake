#!/usr/bin/env python
"""
Attempt to load legacy JSON + HDF5 weights using standalone Keras (2.x).
This script tries several strategies and falls back to manual layer-wise weight
assignment if needed.

Run in a Keras 2.x environment (standalone `keras`, not `tf.keras`):

conda create -n keras2 python=3.7 -y
conda activate keras2
pip install keras==2.3.1 tensorflow==1.15 h5py==2.10.0

Usage:
    python keras2_direct_loader.py model_pol.json model_pol_best.hdf5 model_pol_legacy.h5

"""
import sys
import json
import h5py
import numpy as np

from keras.models import model_from_json, load_model


def find_group_recursive(h5group, name):
    """Search recursively for a subgroup named `name` and return it or None."""
    if name in h5group:
        return h5group[name]
    for key in h5group.keys():
        try:
            subgroup = h5group[key]
            if isinstance(subgroup, h5py.Group):
                found = find_group_recursive(subgroup, name)
                if found is not None:
                    return found
        except Exception:
            continue
    return None


def manual_load_weights(model, weights_path):
    f = h5py.File(weights_path, 'r')
    try:
        if 'model_weights' in f:
            mw = f['model_weights']
        else:
            mw = f

        for layer in model.layers:
            lname = layer.name
            g = find_group_recursive(mw, lname)
            if g is None:
                print(f"No weights group found for layer '{lname}'")
                continue
            # Keras stores weight names in attribute 'weight_names'
            weight_names = [n.decode('utf8') for n in g.attrs.get('weight_names', [])]
            if not weight_names:
                print(f"No weight names found for layer '{lname}' (group exists). Skipping.")
                continue
            weights = []
            for wn in weight_names:
                # wn may be like 'dense/kernel:0' and stored under that name
                try:
                    weights.append(g[wn][:])
                except Exception as e:
                    # sometimes weights are stored under a different subgroup
                    # try stripping path and searching
                    short = wn.split('/')[-1]
                    if short in g:
                        weights.append(g[short][:])
                    else:
                        print(f"Could not read weight '{wn}' for layer '{lname}': {e}")
                        weights.append(None)
            # Filter out None and compare lengths
            weights = [w for w in weights if w is not None]
            if not weights:
                print(f"No usable weights for layer '{lname}'")
                continue
            try:
                layer.set_weights(weights)
                print(f"Set weights for layer '{lname}' (num arrays: {len(weights)})")
            except Exception as e:
                print(f"Failed to set weights for layer '{lname}': {e}")
    finally:
        f.close()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python keras2_direct_loader.py <model.json> <weights.hdf5> <out_model.h5>")
        sys.exit(1)

    json_path, weights_path, out_h5 = sys.argv[1:4]

    print('\nLoading JSON architecture from:', json_path)
    with open(json_path, 'r') as jf:
        json_str = jf.read()

    try:
        model = model_from_json(json_str)
        print('Model built from JSON. Layers:', len(model.layers))
        for i, l in enumerate(model.layers):
            print(i, l.name, type(l).__name__)
    except Exception as e:
        print('Failed to build model from JSON:', e)
        # try to load weights file as a full model
        try:
            m = load_model(weights_path)
            print('Loaded weights file as full model with load_model; saving to', out_h5)
            m.save(out_h5)
            print('Saved:', out_h5)
            sys.exit(0)
        except Exception as e2:
            print('Also could not load weights file as full model:', e2)
            raise

    # Try direct weight loading
    try:
        print('\nTrying model.load_weights(weights_path)')
        model.load_weights(weights_path)
        print('Direct load_weights succeeded.')
    except Exception as e1:
        print('Direct load_weights failed:', e1)
        try:
            print('\nTrying model.load_weights(weights_path, by_name=True)')
            model.load_weights(weights_path, by_name=True)
            print('Load by_name succeeded.')
        except Exception as e2:
            print('Load by_name failed:', e2)
            print('\nAttempting manual layer-wise load from HDF5...')
            manual_load_weights(model, weights_path)

    # Save combined model to HDF5
    try:
        print('\nSaving assembled model to:', out_h5)
        model.save(out_h5)
        print('Saved assembled model to', out_h5)
    except Exception as e:
        print('Failed saving assembled model:', e)
        sys.exit(1)

    print('Done.')
