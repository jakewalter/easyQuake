#!/usr/bin/env python3
"""
Convert legacy EqT_model.h5 (Keras HDF5, older format) into a sanitized
TF2/Keras compatible .keras file by extracting model_config JSON, removing
legacy 'trainable' keys, reconstructing via model_from_json using the
repository's custom layers, loading weights, and saving in modern format.

Run from repository root:
python3 EQTransformer/convert_eqt_model.py /path/to/EqT_model.h5
"""
import sys, os, json, tempfile

p = sys.argv[1] if len(sys.argv) > 1 else 'EQTransformer/EqT_model.h5'
if not os.path.exists(p):
    print('Model file not found:', p); sys.exit(2)

import h5py
from tensorflow.keras.models import model_from_json

# import custom layers from repo
from easyQuake.EQTransformer.EqT_utils import SeqSelfAttention, FeedForward, LayerNormalization, f1
from tensorflow.keras.layers import SpatialDropout1D as KSpatialDropout1D
from tensorflow.keras.optimizers import Adam as KAdam

# include known custom classes and common Keras classes referenced in old configs
custom_objects = {
    'SeqSelfAttention': SeqSelfAttention,
    'FeedForward': FeedForward,
    'LayerNormalization': LayerNormalization,
    'f1': f1,
    'SpatialDropout1D': KSpatialDropout1D,
    'Adam': KAdam,
}

# helper to recursively strip 'trainable'
def strip_trainable(obj):
    modified = False
    if isinstance(obj, dict):
        if obj.get('class_name') == 'SpatialDropout1D' and 'config' in obj:
            if 'trainable' in obj['config']:
                obj['config'].pop('trainable', None)
                modified = True
        for k, v in list(obj.items()):
            sub_mod = strip_trainable(v)
            modified = modified or sub_mod
    elif isinstance(obj, list):
        for item in obj:
            sub_mod = strip_trainable(item)
            modified = modified or sub_mod
    return modified
#!/usr/bin/env python3
"""Convert a legacy Keras HDF5 model to a modern .keras file.

This script looks for JSON model_config blobs inside the HDF5, sanitizes
legacy keys (e.g. 'trainable'), reconstructs the model using the repo's
custom layers, loads weights, and saves a modern Keras file.

Usage:
    python3 EQTransformer/convert_eqt_model.py /path/to/EqT_model.h5
"""

import sys
import os
import json
import tempfile

if len(sys.argv) < 2:
    print('Usage: convert_eqt_model.py /path/to/EqT_model.h5')
    sys.exit(2)

p = sys.argv[1]
if not os.path.exists(p):
    print('Model file not found:', p)
    sys.exit(2)

import h5py
from tensorflow.keras.models import model_from_json

# import project-specific custom layers
from easyQuake.EQTransformer.EqT_utils import SeqSelfAttention, FeedForward, LayerNormalization, f1
from tensorflow.keras.layers import SpatialDropout1D as KSpatialDropout1D
from tensorflow.keras.optimizers import Adam as KAdam

# base custom objects
custom_objects = {
    'SeqSelfAttention': SeqSelfAttention,
    'FeedForward': FeedForward,
    'LayerNormalization': LayerNormalization,
    'f1': f1,
    'SpatialDropout1D': KSpatialDropout1D,
    'Adam': KAdam,
}

# also map common Keras layers that legacy configs may reference
from tensorflow.keras import layers as _kl
for name in ('LSTM', 'GRU', 'SimpleRNN', 'Bidirectional', 'TimeDistributed', 'Masking', 'Dropout'):
    custom_objects.setdefault(name, getattr(_kl, name))


def strip_trainable(obj):
    """Recursively remove legacy 'trainable' keys from a JSON-like structure."""
    modified = False
    if isinstance(obj, dict):
        obj.pop('trainable', None)
        for k, v in list(obj.items()):
            if strip_trainable(v):
                modified = True
    elif isinstance(obj, list):
        for item in obj:
            if strip_trainable(item):
                modified = True
    return modified


def normalize_optim_keys(obj):
    """Recursively rename legacy optimizer keys (lr -> learning_rate)."""
    if isinstance(obj, dict):
        if 'lr' in obj and 'learning_rate' not in obj:
            obj['learning_rate'] = obj.pop('lr')
        for v in obj.values():
            normalize_optim_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            normalize_optim_keys(item)


def find_json_candidates(h5path):
    cands = []
    with h5py.File(h5path, 'r') as f:
        # top-level attrs
        for k, v in f.attrs.items():
            try:
                s = v.decode() if isinstance(v, (bytes, bytearray)) else v
                if isinstance(s, str) and 'class_name' in s:
                    cands.append(s)
            except Exception:
                pass

        # model_config dataset or attr
        try:
            if 'model_config' in f.attrs:
                mc = f.attrs['model_config']
                s = mc.decode() if isinstance(mc, (bytes, bytearray)) else mc
                if isinstance(s, str):
                    cands.append(s)
        except Exception:
            pass

        if 'model_config' in f:
            try:
                v = f['model_config'][()]
                s = v.decode() if isinstance(v, (bytes, bytearray)) else v
                if isinstance(s, str):
                    cands.append(s)
            except Exception:
                pass

        # recursively search datasets
        def walk(g):
            for name, obj in g.items():
                if isinstance(obj, h5py.Dataset):
                    try:
                        v = obj[()]
                        if isinstance(v, (bytes, bytearray, str)):
                            s = v.decode() if isinstance(v, (bytes, bytearray)) else v
                            if isinstance(s, str) and 'class_name' in s:
                                cands.append(s)
                    except Exception:
                        pass
                else:
                    walk(obj)

        walk(f)
    return cands


def try_reconstruct(h5path, custom_objects):
    candidates = find_json_candidates(h5path)
    if not candidates:
        raise RuntimeError('No model_config JSON found in HDF5')

    last_err = None
    for cand in candidates:
        try:
            j = json.loads(cand)
        except Exception as e:
            last_err = e
            continue

        # normalize optimizer keys and remove training_config
        try:
            normalize_optim_keys(j)
        except Exception:
            pass
        j.pop('training_config', None)

        # extract model spec
        model_spec = j.get('model_config', j) if isinstance(j, dict) else j
        # drill down if further nested
        while isinstance(model_spec, dict) and 'model_config' in model_spec:
            model_spec = model_spec['model_config']

        # ensure we have a full Model JSON
        if isinstance(model_spec, dict) and 'class_name' not in model_spec:
            model_spec = {'class_name': 'Model', 'config': model_spec}

        # sanitize
        try:
            strip_trainable(model_spec)
            json_str = json.dumps(model_spec)
            model = model_from_json(json_str, custom_objects=custom_objects)
        except Exception as e:
            last_err = e
            continue

        # load weights
        try:
            model.load_weights(h5path)
        except Exception:
            try:
                with h5py.File(h5path, 'r') as src:
                    if 'model_weights' not in src:
                        raise RuntimeError('model_weights group not found')
                    tmpf = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
                    tmpf.close()
                    with h5py.File(tmpf.name, 'w') as wf:
                        src.copy('model_weights', wf)
                try:
                    model.load_weights(tmpf.name)
                finally:
                    try:
                        os.unlink(tmpf.name)
                    except Exception:
                        pass
            except Exception as e:
                last_err = e
                continue

        return model

    raise RuntimeError('Could not reconstruct model from HDF5') from last_err


def main():
    try:
        model = try_reconstruct(p, custom_objects=custom_objects)
    except Exception as e:
        print('Failed to reconstruct model; last error:', repr(e))
        sys.exit(4)

    out = os.path.splitext(p)[0] + '.keras'
    model.save(out)
    print('Saved converted model to', out)


if __name__ == '__main__':
    main()
