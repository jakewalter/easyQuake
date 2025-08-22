#!/usr/bin/env python3
"""
Helper: ensure test dayfile has three-component mseed entry and run GPD in verbose mode.
"""
import os
import glob
import sys
import importlib.util

DAYFILE = "/home/jwalter/easyQuake/tests/test_project_20250820_105331/20240101/dayfile.in"
OUTFILE = "/home/jwalter/easyQuake/tests/test_project_20250820_105331/20240101/gpd_picks.out"
GPD_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gpd_predict', 'gpd_predict.py'))
GPD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gpd_predict'))

print('DAYFILE =>', DAYFILE)
print('GPD script =>', GPD_PY)
print('OUTFILE =>', OUTFILE)

# Ensure tests mseed files exist
if not os.path.exists(DAYFILE) or os.path.getsize(DAYFILE) == 0:
    print('Dayfile missing or empty; searching for O2.WILZ mseed components under /home/jwalter/easyQuake/tests')
    N = glob.glob('/home/jwalter/easyQuake/tests/**/O2.WILZ.*N*.mseed', recursive=True)
    E = glob.glob('/home/jwalter/easyQuake/tests/**/O2.WILZ.*E*.mseed', recursive=True)
    Z = glob.glob('/home/jwalter/easyQuake/tests/**/O2.WILZ.*Z*.mseed', recursive=True)
    print('Found counts N/E/Z:', len(N), len(E), len(Z))
    if N and E and Z:
        # pick first match from each
        n = N[0]
        e = E[0]
        z = Z[0]
        print('Using:', n, e, z)
        os.makedirs(os.path.dirname(DAYFILE), exist_ok=True)
        with open(DAYFILE, 'w') as f:
            f.write(f"{n} {e} {z}\n")
        print('Wrote dayfile:', DAYFILE)
    else:
        print('Could not find all components; aborting.\nN matches (first 5):', N[:5], '\nE matches:', E[:5], '\nZ matches:', Z[:5])
        sys.exit(0)
else:
    print('Dayfile exists and non-empty; not overwriting.')

# Remove outfile if exists
if os.path.exists(OUTFILE):
    try:
        os.remove(OUTFILE)
    except Exception:
        pass

# Import process_dayfile from the workspace gpd_predict script via spec loader
spec = importlib.util.spec_from_file_location('gpd_predict_cmd', GPD_PY)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print('Failed to load gpd_predict module:', e)
    raise

if hasattr(mod, 'process_dayfile'):
    try:
        print('Calling process_dayfile(...) in verbose mode')
        mod.process_dayfile(DAYFILE, OUTFILE, base_dir=GPD_DIR, verbose=True, plot=False)
    except Exception as e:
        print('process_dayfile raised an exception:', e)
        import traceback
        traceback.print_exc()
else:
    print('gpd_predict module does not expose process_dayfile; aborting')

print('Done. Output file path:', OUTFILE)
if os.path.exists(OUTFILE):
    print('Output size:', os.path.getsize(OUTFILE))
    with open(OUTFILE) as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            print(line.strip())
