#!/usr/bin/env python3
"""Benchmark helper for gpd_predict: model load, predict, inline cached calls, and CLI calls.

Run this from the repository root: python gpd_predict/benchmark_gpd.py
"""
import time
import os
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL = os.path.join('gpd_predict', 'model_pol_new.keras')
DAYFILE = os.path.join('gpd_predict', 'testdata', 'dayfile.in')
OUT = os.path.join('gpd_predict', 'testdata', 'out_bench.txt')


def prepare_testdata():
    try:
        from obspy import Trace, Stream, UTCDateTime
        import numpy as np
    except Exception as e:
        print('missing obspy or numpy; create testdata requires them:', e)
        return False
    base = os.path.join('gpd_predict', 'testdata')
    if not os.path.exists(base):
        os.makedirs(base)
        sr = 100.0
        n = 1200
        t0 = UTCDateTime()
        for name, chan in [('N.mseed', 'HHN'), ('E.mseed', 'HHE'), ('Z.mseed', 'HHZ')]:
            tr = Trace()
            tr.stats.sampling_rate = sr
            tr.stats.starttime = t0
            tr.stats.network = 'XX'
            tr.stats.station = 'TEST'
            tr.stats.channel = chan
            tr.data = (np.sin(np.linspace(0, 50, n)) + 0.01 * np.random.randn(n)).astype('float32')
            st = Stream([tr])
            st.write(os.path.join(base, name), format='MSEED')
        with open(os.path.join(base, 'dayfile.in'), 'w') as f:
            f.write(os.path.join(base, 'N.mseed') + ' ' + os.path.join(base, 'E.mseed') + ' ' + os.path.join(base, 'Z.mseed') + '\n')
    return True


def time_model_load():
    try:
        import tensorflow as tf
        try:
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        except Exception:
            pass
        from tensorflow.keras.models import load_model
        t0 = time.time()
        load_model(MODEL)
        return time.time() - t0
    except Exception as e:
        return f'error: {e}'


def time_predict():
    try:
        from tensorflow.keras.models import load_model
        import numpy as np
        m = load_model(MODEL)
        x = np.random.randn(100, 400, 3).astype('float32')
        t0 = time.time()
        m.predict(x, batch_size=100, verbose=0)
        return time.time() - t0
    except Exception as e:
        return f'error: {e}'


def time_inline_calls():
    try:
        from gpd_predict import process_dayfile
        t0 = time.time()
        process_dayfile(DAYFILE, OUT, base_dir='gpd_predict', verbose=False, plot=False)
        t1 = time.time()
        process_dayfile(DAYFILE, OUT, base_dir='gpd_predict', verbose=False, plot=False)
        t2 = time.time()
        return (t1 - t0, t2 - t1)
    except Exception as e:
        return ('error', str(e))


def time_cli_calls():
    cmd = ['python', os.path.join('gpd_predict', 'gpd_predict.py'), '-V', '-P', '-I', DAYFILE, '-O', OUT, '-F', 'gpd_predict']
    try:
        t0 = time.time()
        subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        t1 = time.time()
        subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        t2 = time.time()
        return (t1 - t0, t2 - t1)
    except Exception as e:
        return ('error', str(e))


def main():
    print('cwd', os.getcwd())
    ok = prepare_testdata()
    print('prepare_testdata:', ok)
    print('model_load_time:', time_model_load())
    print('single_predict_time:', time_predict())
    inline = time_inline_calls()
    print('inline_first, inline_second:', inline)
    cli = time_cli_calls()
    print('cli_first, cli_second:', cli)


if __name__ == '__main__':
    main()
