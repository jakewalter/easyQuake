#!/usr/bin/env python3
"""
Convert legacy EqT_model.h5 to .keras using the EQTransformer converter.
Run this script in the project's environment where TensorFlow is installed.

Usage:
    python3 tools/convert_eqt_model.py \
        --h5 /path/to/EqT_model.h5 \
        [--out /path/to/EqT_model.keras]

If run without args it will try the default path inside the repo.
"""
import argparse
import os
import importlib.util
import importlib.machinery
import traceback

DEFAULT_H5 = os.path.join(os.path.dirname(__file__), '..', 'EQTransformer', 'EqT_model.h5')
DEFAULT_MSEED = os.path.join(os.path.dirname(__file__), '..', 'EQTransformer', 'mseed_predictor.py')


def load_mseed_predictor(path):
    spec = importlib.util.spec_from_loader('mp_mseed', importlib.machinery.SourceFileLoader('mp_mseed', path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default=os.path.abspath(DEFAULT_H5), help='Path to legacy EqT_model.h5')
    p.add_argument('--mseed', default=os.path.abspath(DEFAULT_MSEED), help='Path to mseed_predictor.py')
    p.add_argument('--out', default=None, help='Output .keras path (optional)')
    args = p.parse_args()

    print('Using mseed_predictor:', args.mseed)
    print('Input HDF5:', args.h5)
    if args.out:
        print('Output file:', args.out)

    try:
        mp = load_mseed_predictor(args.mseed)
    except Exception:
        print('Failed to load mseed_predictor module:')
        traceback.print_exc()
        return 2

    if not hasattr(mp, 'convert_h5_to_keras'):
        print('mseed_predictor does not expose convert_h5_to_keras; available attrs:')
        print(sorted([k for k in dir(mp) if not k.startswith('__')])[:200])
        return 3

    try:
        out = mp.convert_h5_to_keras(args.h5, outpath=args.out)
        print('Conversion succeeded ->', out)
        return 0
    except Exception:
        print('Conversion failed:')
        traceback.print_exc()
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
