set -eu
PYTHON=python
MODEL=gpd_predict/model_pol_new.keras
DAYFILE=gpd_predict/testdata/dayfile.in
OUT=gpd_predict/testdata/out_bench.txt

# ensure testdata exists (create simple dayfile if missing)
python - <<'PY'
from obspy import Trace, Stream, UTCDateTime
import numpy as np, os
base='gpd_predict/testdata'
if not os.path.exists(base):
    os.makedirs(base)
    sr=100.0
    n=1200
    t0=UTCDateTime()
    for name, chan in [('N.mseed','HHN'),('E.mseed','HHE'),('Z.mseed','HHZ')]:
        tr=Trace()
        tr.stats.sampling_rate=sr
        tr.stats.starttime=t0
        tr.stats.network='XX'; tr.stats.station='TEST'; tr.stats.channel=chan
        tr.data = (np.sin(np.linspace(0,50,n)) + 0.01*np.random.randn(n)).astype('float32')
        st=Stream([tr])
        st.write(base+'/'+name, format='MSEED')
    with open(base+'/dayfile.in','w') as f:
        f.write(base+'/N.mseed '+base+'/E.mseed '+base+'/Z.mseed\n')
print('prepared testdayfile')
PY

echo '1) measure model load time'
 - <<'PY'
import time
try:
    import tensorflow as tf
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    except Exception:
        pass
    from tensorflow.keras.models import load_model
    t0=time.time()
    load_model('')
    print('load_time', time.time()-t0)
except Exception as e:
    print('load_time_error', e)
PY

echo '2) measure single predict time (batch of 100 windows)'
 - <<'PY'
import time, numpy as np
try:
    from tensorflow.keras.models import load_model
    m=load_model('')
    x=np.random.randn(100,400,3).astype('float32')
    t0=time.time()
    m.predict(x, batch_size=100, verbose=0)
    print('predict_time', time.time()-t0)
except Exception as e:
    print('predict_time_error', e)
PY

echo '3) inline: call process_dayfile twice (first=load+predict, second=reuse)'
 - <<'PY'
import time
try:
    from gpd_predict import process_dayfile
    t0=time.time()
    process_dayfile('','', base_dir='gpd_predict', verbose=False, plot=False)
    print('inline_first', time.time()-t0)
    t0=time.time()
    process_dayfile('','', base_dir='gpd_predict', verbose=False, plot=False)
    print('inline_second', time.time()-t0)
except Exception as e:
    print('inline_error', e)
PY

echo '4) separate process: invoke CLI twice'
PY_START=1755554052.588173547
python gpd_predict/gpd_predict.py -V -P -I  -O  -F gpd_predict || true
PY_END=1755554052.592658747
echo cli_first 
PY_START=1755554052.626352241
python gpd_predict/gpd_predict.py -V -P -I  -O  -F gpd_predict || true
PY_END=1755554052.627937894
echo cli_second 
