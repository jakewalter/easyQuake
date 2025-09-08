#!/usr/bin/env python3
"""
Standalone converter: reconstruct EqT model from HDF5 JSON + weights and save .keras.
This avoids importing the edited `mseed_predictor.py` so it is robust to the file's current state.

Usage:
    python3 tools/convert_standalone.py --h5 EQTransformer/EqT_model.sanitized.h5
"""
import os, sys, json, argparse, h5py, tempfile, importlib.util, importlib.machinery

parser = argparse.ArgumentParser()
parser.add_argument('--h5', required=True)
parser.add_argument('--out', default=None)
args = parser.parse_args()

h5path = args.h5
outpath = args.out or os.path.splitext(h5path)[0] + '.keras'

# load EqT_utils by path
repo_dir = os.path.dirname(os.path.dirname(__file__))
eqt_utils_path = os.path.join(repo_dir, 'EQTransformer', 'EqT_utils.py')
if not os.path.exists(eqt_utils_path):
    print('EqT_utils.py not found at', eqt_utils_path)
    sys.exit(2)
spec = importlib.util.spec_from_loader('eqt_utils', importlib.machinery.SourceFileLoader('eqt_utils', eqt_utils_path))
eq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eq)
SeqSelfAttention = eq.SeqSelfAttention
FeedForward = eq.FeedForward
LayerNormalization = eq.LayerNormalization
f1 = eq.f1

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as _layers

# shim SpatialDropout1D to accept legacy kwargs
try:
    class SpatialDropout1D(tf.keras.layers.SpatialDropout1D):
        def __init__(self, rate, noise_shape=None, seed=None, dtype=None, **kwargs):
            kwargs.pop('trainable', None)
            kwargs.pop('noise_shape', None)
            kwargs.pop('seed', None)
            kwargs.pop('dtype', None)
            super().__init__(rate=rate, **kwargs)

        @classmethod
        def from_config(cls, config):
            cfg = dict(config)
            cfg.pop('trainable', None)
            cfg.pop('noise_shape', None)
            cfg.pop('seed', None)
            cfg.pop('dtype', None)
            return super().from_config(cfg)
except Exception:
    pass

try:
    _layers.SpatialDropout1D = SpatialDropout1D
    tf.keras.utils.get_custom_objects()['SpatialDropout1D'] = SpatialDropout1D
except Exception:
    pass

# Provide legacy alias for keras.engine.InputSpec if missing
try:
    if not hasattr(tf.keras, 'engine'):
        class _EngineShim:
            pass
        tf.keras.engine = _EngineShim()
        tf.keras.engine.InputSpec = tf.keras.layers.InputSpec
except Exception:
    pass

custom_objects = {
    'SeqSelfAttention': SeqSelfAttention,
    'FeedForward': FeedForward,
    'LayerNormalization': LayerNormalization,
    'f1': f1,
    'SpatialDropout1D': _layers.SpatialDropout1D,
}
for name in ('LSTM','GRU','SimpleRNN','Bidirectional','TimeDistributed','Masking','Dropout','Add'):
    custom_objects.setdefault(name, getattr(_layers, name))

def convert(h5path, outpath=None):
    # find JSON candidates
    candidates = []
    with h5py.File(h5path, 'r') as f:
    if 'model_config' in f.attrs:
        mc = f.attrs['model_config']
        candidates.append(mc.decode() if isinstance(mc, (bytes, bytearray)) else mc)
    if 'model_config' in f:
        try:
            mc = f['model_config'][()]
            candidates.append(mc.decode() if isinstance(mc, (bytes, bytearray)) else mc)
        except Exception:
            pass

    def walk(g):
        for name, obj in g.items():
            try:
                if isinstance(obj, h5py.Dataset):
                    try:
                        v = obj[()]
                        if isinstance(v, (bytes, bytearray, str)):
                            s = v.decode() if isinstance(v, (bytes, bytearray)) else v
                            if 'class_name' in s and 'model_config' in s[:5000]:
                                candidates.append(s)
                    except Exception:
                        pass
                else:
                    walk(obj)
            except Exception:
                pass

    walk(f)

if not candidates:
    print('No model config JSON found in HDF5')
    sys.exit(3)

last_err = None
for cand in candidates:
    try:
        j = json.loads(cand)
    except Exception:
        continue

    cfg = j['config'] if isinstance(j, dict) and j.get('class_name') == 'Model' else j

    def strip_trainable(obj):
        if isinstance(obj, dict):
            obj.pop('trainable', None)
            for v in obj.values():
                strip_trainable(v)
        elif isinstance(obj, list):
            for it in obj:
                strip_trainable(it)

    strip_trainable(cfg)

    try:
        model = Model.from_config(cfg, custom_objects=custom_objects)
    except Exception as e:
        last_err = e
        continue

    try:
        model.load_weights(h5path)
    except Exception:
        try:
            with h5py.File(h5path, 'r') as f:
                if 'model_weights' in f:
                    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
                    tmp.close()
                    try:
                        with h5py.File(tmp.name, 'w') as wf:
                            f.copy('model_weights', wf)
                        model.load_weights(tmp.name)
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass
                else:
                    raise
        except Exception as e:
            last_err = e
            continue

    model.save(outpath)
    print('Saved .keras model to', outpath)
    return outpath

    # nothing worked
    raise RuntimeError(f'Conversion failed; last error: {last_err}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    try:
        out = convert(args.h5, args.out)
        print('Conversion succeeded ->', out)
        sys.exit(0)
    except Exception as e:
        print('Conversion failed:', e)
        sys.exit(4)
