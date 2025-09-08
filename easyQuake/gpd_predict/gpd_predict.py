#! /usr/bin/env python3

# Automatic picking of seismic waves using Generalized Phase Detection 
# See http://scedc.caltech.edu/research-tools/deeplearning.html for more info
#
# Ross et al. (2018), Generalized Seismic Phase Detection with Deep Learning,
#                     Bull. Seismol. Soc. Am., doi:10.1785/0120180080
#                                              
# Author: Zachary E. Ross (2018)                
# Contact: zross@gps.caltech.edu                        
# Website: http://www.seismolab.caltech.edu/ross_z.html         

import argparse as ap
import os
import numpy as np
import obspy.core as oc
import pylab as plt
import sys
#####################
# Hyperparameters
min_proba = 0.994 # Minimum softmax probability for phase detection (balanced threshold for practical use)
freq_min = 3.0
freq_max = 20.0
filter_data = True
decimate_data = True # If false, assumes data is already 100 Hz samprate
n_shift = 10 # Number of samples to shift the sliding window at a time
n_gpu = 1 # Number of GPUs to use (if any)
#####################
batch_size = 1000*3

half_dur = 2.00
only_dt = 0.01
n_win = int(half_dur/only_dt)
n_feat = 2*n_win

#-------------------------------------------------------------

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


def main():
    parser = ap.ArgumentParser(
        prog='gpd_predict.py',
        description='Automatic picking of seismic waves using'
                    'Generalized Phase Detection')
    parser.add_argument(
        '-I',
        type=str,
        default=None,
        help='Input file')
    parser.add_argument(
        '-O',
        type=str,
        default=None,
        help='Output file')
    parser.add_argument(
        '-P',
        default=True,
        action='store_false',
        help='Suppress plotting output')
    parser.add_argument(
        '-V',
        default=False,
        action='store_true',
        help='verbose')
    parser.add_argument(
        '-F',
        type=str,
        default=None,
        help='path where GPD lives')
    args = parser.parse_args()

    plot = args.P
    # Delegate to reusable function
    process_dayfile(args.I, args.O, base_dir=args.F, verbose=args.V, plot=plot)


def process_dayfile(infile, outfile, base_dir=None, verbose=False, plot=False):
    """Process a dayfile (infile) and write picks to outfile.

    This function mirrors the behavior of the CLI main(), but is callable
    programmatically and will cache model(s) per process if called multiple times.
    """
    # Reading in input file
    fdir = []
    evid = []
    staid = []
    with open(infile) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)

    # module-level cache (per-process) to avoid reloading the model repeatedly
    global _CACHED_MODEL
    try:
        _CACHED_MODEL
    except NameError:
        _CACHED_MODEL = None

    # load model using TensorFlow/Keras (import lazily to avoid heavy startup at module import)
    try:
        import tensorflow as tf
        # configure GPUs for growth to avoid full allocation
        try:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            [tf.config.experimental.set_memory_growth(physical_devices[i], True) for i in range(0,len(physical_devices))]
        except Exception:
            pass
    except Exception:
        # Let import errors bubble when model loading is attempted
        raise

    import keras
    model = _CACHED_MODEL
    base_dir = base_dir if base_dir else os.path.dirname(__file__)
    
    # Try calibrated models first
    gpd_calibrated_path = os.path.join(base_dir, 'model_pol_gpd_calibrated_F80.h5')
    properly_converted_path = os.path.join(base_dir, 'model_pol_properly_converted.keras')
    fixed_path = os.path.join(base_dir, 'model_pol_fixed.h5')
    keras_path = os.path.join(base_dir, 'model_pol_new.keras')
    h5_path = os.path.join(base_dir, 'model_pol_legacy.h5')

    # Try models in order of preference
    if model is None:
        # 1. Try optimized converted model first (from original JSON, single-branch)
        optimized_converted_path = os.path.join(base_dir, 'model_pol_optimized_converted.keras')
        if os.path.isfile(optimized_converted_path):
            try:
                model = keras.models.load_model(optimized_converted_path)
                print(f"Loaded optimized converted model from: {optimized_converted_path}")
            except Exception as e_opt:
                print(f"Failed to load optimized converted model ({optimized_converted_path}): {e_opt}")
        
        # 2. Try final converted model as backup
        final_converted_path = os.path.join(base_dir, 'model_pol_final_converted.keras')
        if model is None and os.path.isfile(final_converted_path):
            try:
                model = keras.models.load_model(final_converted_path)
                print(f"Loaded final converted model from: {final_converted_path}")
            except Exception as e_final:
                print(f"Failed to load final converted model ({final_converted_path}): {e_final}")
        
        # 3. Try GPD-calibrated model as backup
        if model is None and os.path.isfile(gpd_calibrated_path):
            try:
                model = keras.models.load_model(gpd_calibrated_path)
                print(f"Loaded GPD-calibrated model from: {gpd_calibrated_path}")
            except Exception as e_gpd:
                print(f"Failed to load GPD-calibrated model ({gpd_calibrated_path}): {e_gpd}")
        
        # 4. Try fixed temperature-corrected model as backup
        if model is None and os.path.isfile(fixed_path):
            try:
                model = keras.models.load_model(fixed_path)
                print(f"Loaded temperature-corrected model from: {fixed_path}")
            except Exception as e_fixed:
                print(f"Failed to load fixed model ({fixed_path}): {e_fixed}")
        
        # 3. Try .keras if calibrated models not available or failed
        if model is None:
            try:
                model = keras.models.load_model(keras_path)
                print(f"Loaded model from: {keras_path}")
            except Exception as e_keras:
                print(f"Failed to load .keras model ({keras_path}): {e_keras}")
                
        # 3. Try legacy HDF5 fallback
        if model is None and os.path.isfile(h5_path):
            try:
                model = keras.models.load_model(h5_path)
                print(f"Loaded legacy HDF5 model from: {h5_path}")
            except Exception as e_h5:
                print(f"Failed to load .h5 model ({h5_path}): {e_h5}")
                
        if model is None:
            raise RuntimeError("Failed to load any GPD model variant")
        
        _CACHED_MODEL = model

    if n_gpu > 1:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=n_gpu)

    ofile = open(outfile, 'w')

    for i in range(nsta):
        try:
            fname = fdir[i][0].split("/")
            if not os.path.isfile(fdir[i][0]):
                print("%s doesn't exist, skipping" % fdir[i][0])
                continue
            if not os.path.isfile(fdir[i][1]):
                print("%s doesn't exist, skipping" % fdir[i][1])
                continue
            if not os.path.isfile(fdir[i][2]):
                print("%s doesn't exist, skipping" % fdir[i][2])
                continue
            st = oc.Stream()
            st += oc.read(fdir[i][0])  # N
            st += oc.read(fdir[i][1])  # E
            st += oc.read(fdir[i][2])  # Z
            
            # Sort traces by channel (Z, N, E) - GPD expects this order
            st.sort(['channel'])
            
            # Filter and process (using exact same preprocessing as test_gpd_direct.py)
            if filter_data:
                st.filter('highpass', freq=freq_min, corners=2, zerophase=True)
                st.filter('lowpass', freq=freq_max, corners=2, zerophase=True)
            st.detrend('demean')
            st.detrend('linear')
            
            # Resample to 100 Hz if needed
            if decimate_data:
                for tr in st:
                    if tr.stats.sampling_rate != 100.0:
                        tr.resample(100.0)
            
            st.merge(fill_value='interpolate')
            print(st)
            for tr in st:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled()


            chan = st[0].stats.channel
            sr = st[0].stats.sampling_rate

            dt = st[0].stats.delta
            net = st[0].stats.network
            sta = st[0].stats.station
            chan = st[0].stats.channel
            latest_start = np.max([x.stats.starttime for x in st])
            earliest_stop = np.min([x.stats.endtime for x in st])
            if (earliest_stop>latest_start):
                st.trim(latest_start, earliest_stop)
                if verbose:
                    print("Reshaping data matrix for sliding window")
                tt = (np.arange(0, st[0].data.size, n_shift) + n_win) * dt
                tt_i = np.arange(0, st[0].data.size, n_shift) + n_feat
                #tr_win = np.zeros((tt.size, n_feat, 3))
                sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
                sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
                sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)
                tr_win = np.zeros((sliding_N.shape[0], n_feat, 3))
                tr_win[:,:,0] = sliding_N
                tr_win[:,:,1] = sliding_E
                tr_win[:,:,2] = sliding_Z
                tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
                tt = tt[:tr_win.shape[0]]
                tt_i = tt_i[:tr_win.shape[0]]

                if verbose:
                    ts = model.predict(tr_win, verbose=True, batch_size=batch_size)
                else:
                    ts = model.predict(tr_win, verbose=False, batch_size=batch_size)

                prob_S = ts[:,1]
                prob_P = ts[:,0]
                prob_N = ts[:,2]

                from obspy.signal.trigger import trigger_onset
                trigs = trigger_onset(prob_P, min_proba, 0.1)
                print(f"DEBUG: P-wave max probability: {np.max(prob_P):.4f}, min_proba threshold: {min_proba}")
                print(f"DEBUG: Number of P triggers found: {len(trigs)}")
                p_picks = []
                s_picks = []
                for trig in trigs:
                    if trig[1] == trig[0]:
                        continue
                    pick = np.argmax(ts[trig[0]:trig[1], 0])+trig[0]
                    stamp_pick = st[0].stats.starttime + tt[pick]
                    chan_pick = st[0].stats.channel[0:2]+'Z'
                    p_picks.append(stamp_pick)
                    ofile.write("%s %s %s P %s\n" % (net, sta, chan_pick, stamp_pick.isoformat()))

                trigs = trigger_onset(prob_S, min_proba, 0.1)
                print(f"DEBUG: S-wave max probability: {np.max(prob_S):.4f}")
                print(f"DEBUG: Number of S triggers found: {len(trigs)}")
                for trig in trigs:
                    if trig[1] == trig[0]:
                        continue
                    pick = np.argmax(ts[trig[0]:trig[1], 1])+trig[0]
                    stamp_pick = st[0].stats.starttime + tt[pick]
                    chan_pick_s = st[0].stats.channel[0:2]+'E'
                    s_picks.append(stamp_pick)
                    ofile.write("%s %s %s S %s\n" % (net, sta, chan_pick_s, stamp_pick.isoformat()))

                if plot:
                    fig = plt.figure(figsize=(8, 12))
                    ax = []
                    ax.append(fig.add_subplot(4,1,1))
                    ax.append(fig.add_subplot(4,1,2,sharex=ax[0],sharey=ax[0]))
                    ax.append(fig.add_subplot(4,1,3,sharex=ax[0],sharey=ax[0]))
                    ax.append(fig.add_subplot(4,1,4,sharex=ax[0]))
                    for i in range(3):
                        ax[i].plot(np.arange(st[i].data.size)*dt, st[i].data, c='k', \
                                   lw=0.5)
                    ax[3].plot(tt, ts[:,0], c='r', lw=0.5)
                    ax[3].plot(tt, ts[:,1], c='b', lw=0.5)
                    for p_pick in p_picks:
                        for i in range(3):
                            ax[i].axvline(p_pick-st[0].stats.starttime, c='r', lw=0.5)
                    for s_pick in s_picks:
                        for i in range(3):
                            ax[i].axvline(s_pick-st[0].stats.starttime, c='b', lw=0.5)
                    plt.tight_layout()
                    plt.show()
        except Exception as e:
            print(f"DEBUG: Exception in GPD processing: {e}")
            import traceback
            traceback.print_exc()
    ofile.close()


if __name__ == "__main__":
    main()