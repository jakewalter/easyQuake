#!/usr/bin/env python3
"""
Sanitize legacy Keras HDF5 model JSON blobs by removing keys that break
modern Keras deserialization (for example, 'trainable').

This script does NOT require TensorFlow. It creates a copy of the HDF5 file
and removes/rewrites any attributes or small datasets that contain JSON-like
text and a 'class_name' or 'model_config' token.

Usage:
    python3 tools/sanitize_eqt_h5.py --h5 EQTransformer/EqT_model.h5

Output: by default writes EQTransformer/EqT_model.sanitized.h5 next to input.
"""
import argparse
import os
import shutil
import json
import tempfile
import h5py


def is_text_like(obj):
    return isinstance(obj, (bytes, bytearray, str))


def try_parse_json(s):
    try:
        if isinstance(s, (bytes, bytearray)):
            s2 = s.decode('utf-8')
        else:
            s2 = s
        return json.loads(s2)
    except Exception:
        return None


def strip_trainable(obj):
    if isinstance(obj, dict):
        obj.pop('trainable', None)
        for k, v in list(obj.items()):
            strip_trainable(v)
    elif isinstance(obj, list):
        for it in obj:
            strip_trainable(it)


def sanitize_h5(inpath, outpath=None, max_dataset_size=500000):
    if outpath is None:
        base, ext = os.path.splitext(inpath)
        outpath = base + '.sanitized' + ext
    shutil.copyfile(inpath, outpath)

    with h5py.File(outpath, 'r+') as f:
        # sanitize attributes on root and groups
        def san_attr(obj):
            for key, val in list(obj.attrs.items()):
                if is_text_like(val):
                    parsed = try_parse_json(val)
                    if parsed is not None and (('class_name' in parsed) or ('model_config' in str(val)[:5000])):
                        try:
                            # mutate and write back
                            if isinstance(parsed, dict) and parsed.get('class_name') == 'Model':
                                cfg = parsed['config']
                            else:
                                cfg = parsed
                            strip_trainable(cfg)
                            new = json.dumps(parsed, separators=(',', ':')).encode('utf-8')
                            obj.attrs.modify(key, new)
                        except Exception:
                            pass

        def walk_group(g):
            san_attr(g)
            for name, child in list(g.items()):
                try:
                    if isinstance(child, h5py.Dataset):
                        # try to read small textual datasets
                        try:
                            v = child[()]
                        except Exception:
                            continue
                        if is_text_like(v):
                            parsed = try_parse_json(v)
                            if parsed is not None and (('class_name' in parsed) or ('model_config' in str(v)[:5000])):
                                try:
                                    if isinstance(parsed, dict) and parsed.get('class_name') == 'Model':
                                        cfg = parsed['config']
                                    else:
                                        cfg = parsed
                                    strip_trainable(cfg)
                                    new = json.dumps(parsed, separators=(',', ':')).encode('utf-8')
                                    # overwrite dataset with new bytes
                                    del g[name]
                                    g.create_dataset(name, data=new)
                                except Exception:
                                    pass
                        else:
                            # skip large binary datasets (weights)
                            pass
                    else:
                        walk_group(child)
                except Exception:
                    pass

        walk_group(f)

    return outpath


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True, help='Path to input HDF5 model')
    p.add_argument('--out', default=None, help='Output sanitized HDF5 path')
    args = p.parse_args()

    out = sanitize_h5(args.h5, args.out)
    print('Sanitized file written to:', out)


if __name__ == '__main__':
    main()
