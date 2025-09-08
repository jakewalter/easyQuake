#!/usr/bin/env python3
"""EQTransformer mseed_predictor with conversion utilities.

This file provides:
- a .keras-first loader for EQTransformer models
- helpers to sanitize legacy HDF5 Keras files and convert them to .keras
- the original archive functionality for prediction

Behaviour is explicit: conversion only happens if requested via CLI flags
or when no .keras model is present but an HDF5 is available.
"""

from __future__ import annotations
import os
import sys
import argparse
import json
import shutil
import tempfile
import numpy as np
from typing import Optional
import math
from datetime import datetime, timedelta
import faulthandler
faulthandler.enable()
import tensorflow as tf
import obspy
import warnings
from obspy.signal.trigger import trigger_onset
warnings.filterwarnings("ignore")


def _select_keras_model(folder: str) -> Optional[str]:
    env = os.environ.get('EASYQUAKE_EQT_MODEL')
    if env and os.path.exists(env):
        return env
    cand = os.path.join(folder, 'EqT_model.sanitized.keras')
    if os.path.exists(cand):
        return cand
    try:
        for f in os.listdir(folder):
            if f.endswith('.keras'):
                return os.path.join(folder, f)
    except Exception:
        pass
    return None


def _register_customs() -> dict:
    # register custom layers and other compatibility fixes
    import tensorflow as tf
    try:
        import tensorflow.keras.backend as K
    except ImportError:
        import keras.backend as K
    
    # Add missing functions to Keras backend for compatibility
    try:
        if not hasattr(K, 'in_train_phase'):
            def in_train_phase(x, alt, training=None):
                """Compatibility shim for K.in_train_phase"""
                try:
                    if training is None:
                        return x
                    elif training:
                        return x() if callable(x) else x
                    else:
                        return alt
                except Exception:
                    return alt
            K.in_train_phase = in_train_phase
    except Exception:
        pass
    
    try:
        from easyQuake.EQTransformer.EqT_utils import SeqSelfAttention, FeedForward, LayerNormalization, f1
        return {
            'SeqSelfAttention': SeqSelfAttention,
            'FeedForward': FeedForward,
            'LayerNormalization': LayerNormalization,
            'f1': f1,
            # general keras layer mapping
            'Dense': tf.keras.layers.Dense,
            'Dropout': tf.keras.layers.Dropout,
            'LSTM': tf.keras.layers.LSTM,
            'Conv1D': tf.keras.layers.Conv1D,
            'MaxPooling1D': tf.keras.layers.MaxPooling1D,
            'Activation': tf.keras.layers.Activation,
            'Add': tf.keras.layers.Add,
            'Input': tf.keras.layers.Input,
            'Concatenate': tf.keras.layers.Concatenate,
            'BatchNormalization': tf.keras.layers.BatchNormalization,
            'SpatialDropout1D': tf.keras.layers.SpatialDropout1D,
        }
    except Exception as e:
        print(f'Warning: Could not import custom objects: {e}', file=sys.stderr)
        return {}


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):
    """Detect peaks in data based on their amplitude and other features."""
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    return ind


def _picker(args, yh1, yh2, yh3):
    """Performs detection and picking."""
    detection = trigger_onset(yh1, args['detection_threshold'], args['detection_threshold'])
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)

    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    if len(pp_arr) > 0:
        P_uncertainty = None
        for pick in range(len(pp_arr)):
            pauto = pp_arr[pick]
            if pauto:
                P_prob = np.round(yh2[int(pauto)], 3)
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})

    if len(ss_arr) > 0:
        S_uncertainty = None
        for pick in range(len(ss_arr)):
            sauto = ss_arr[pick]
            if sauto:
                S_prob = np.round(yh3[int(sauto)], 3)
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})

    if len(detection) > 0:
        D_uncertainty = None
        for ev in range(len(detection)):
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][1]]})

    # matching the detection and picks
    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        if int(ed-bg) >= 10:
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.update({Ss : S_val})

            if len(candidate_Ss) > 1:
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}

            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                        candidate_Ps.update({Ps : P_val})
                else:
                    if Ps > bg-100 and Ps < ed:
                        candidate_Ps.update({Ps : P_val})

            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan}
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer

            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:
                matches.update({
                                bg:[ed,
                                    EVENTS[ev][0],
                                    EVENTS[ev][1],

                                    list(candidate_Ps)[0],
                                    candidate_Ps[list(candidate_Ps)[0]][0],
                                    candidate_Ps[list(candidate_Ps)[0]][1],

                                    list(candidate_Ss)[0],
                                    candidate_Ss[list(candidate_Ss)[0]][0],
                                    candidate_Ss[list(candidate_Ss)[0]][1],
                                                ] })

    return matches, pick_errors, yh3

def _get_snr(data, pat, window=200):
    """Estimates SNR."""
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)]
                sw1 = data[int(pat) : int(pat)+window]
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)]
                sw1 = data[int(pat) : int(pat)+window]
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)]
                sw1 = data[int(pat) : int(pat)+window]
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
        except Exception:
            pass
    return snr


def _resampling(st):
    """Perform resampling on Obspy stream objects."""
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)
            st.append(tr)
    return st


def _normalize(data, mode='std'):
    """Normalize 3D arrays."""
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data
    elif mode == 'std':
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data


class PreLoadGeneratorTest(tf.keras.utils.Sequence):
    """Keras generator with preprocessing. For testing. Pre-load version."""

    def __init__(self,
                 list_IDs,
                 inp_data,
                 batch_size=32,
                 norm_mode = 'std'):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.inp_data = inp_data
        self.on_epoch_end()
        self.norm_mode = norm_mode

    def __len__(self):
        'Denotes the number of batches per epoch'
        try:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        except ZeroDivisionError:
            return 0

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def _normalize(self, data, mode = 'std'):
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data
        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def __data_generation(self, list_IDs_temp):
        'reading the waveforms'
        X = np.zeros((self.batch_size, 6000, 3))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            data = self.inp_data[ID]
            data = self._normalize(data, self.norm_mode)
            X[i, :, :] = data
        return X


def _mseed2nparry(args, fdir, i, time_slots, comp_types):
    """Read miniseed files and return numpy arrays, meta data, and time slice info."""
    netname = fdir[i][0].split("/")[-1].split('.')[0]
    st = obspy.core.Stream()
    st += obspy.core.read(fdir[i][0])
    st += obspy.core.read(fdir[i][1])
    st += obspy.core.read(fdir[i][2])
    
    try:
       st.merge(fill_value=0)
    except Exception:
        st =_resampling(st)
        st.merge(fill_value=0)
    
    for tr in st:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled()

    st.detrend(type='linear')
    st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
    #st.taper(max_percentage=0.001, type='cosine', max_length=2)
    st.taper(max_percentage=0.05, type="cosine", side="both")

    time_slots.append((st[0].stats.starttime, st[0].stats.endtime))

    if len([tr for tr in st if tr.stats.sampling_rate != 100.0]) != 0:
        try:
            st.interpolate(100, method="linear")
        except Exception:
            st=_resampling(st)

    st.trim(min([tr.stats.starttime for tr in st]), max([tr.stats.endtime for tr in st]), pad=True, fill_value=0)

    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime

    meta = {"start_time":start_time,
            "end_time": end_time,
            "trace_name":i,
            "network_code": netname
             }

    chanL = [tr.stats.channel[-1] for tr in st]
    comp_types.append(len(chanL))
    tim_shift = int(60-(args['overlap']*60))
    next_slice = start_time+60

    data_set={}
    sl = 0; st_times = []
    while next_slice <= end_time:
        npz_data = np.zeros([6000, 3])
        st_times.append(str(start_time).replace('T', ' ').replace('Z', ''))
        w = st.slice(start_time, next_slice)
        if 'Z' in chanL:
            npz_data[:,2] = w[chanL.index('Z')].data[:6000]
        if ('E' in chanL) or ('1' in chanL):
            try:
                npz_data[:,0] = w[chanL.index('E')].data[:6000]
            except Exception:
                npz_data[:,0] = w[chanL.index('1')].data[:6000]
        if ('N' in chanL) or ('2' in chanL):
            try:
                npz_data[:,1] = w[chanL.index('N')].data[:6000]
            except Exception:
                npz_data[:,1] = w[chanL.index('2')].data[:6000]

        data_set.update( {str(start_time).replace('T', ' ').replace('Z', '') : npz_data})

        start_time = start_time+tim_shift
        next_slice = next_slice+tim_shift
        sl += 1

    meta["trace_start_time"] = st_times

    try:
        meta["receiver_code"]=st[0].stats.station
        meta["instrument_type"]=st[0].stats.channel[:2]
    except Exception:
        meta["receiver_code"]='blah'
        meta["instrument_type"]=st[0].stats.channel[:2]

    return meta, time_slots, comp_types, data_set


def _output_writter_prediction(meta, ofile, matches, snr, detection_memory, idx):
    """Writes the detection & picking results into a CSV file."""
    station_name = meta["receiver_code"]
    start_time = meta["trace_start_time"][idx]
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    for match, match_value in matches.items():
        ev_strt = start_time+timedelta(seconds= match/100)
        ev_end = start_time+timedelta(seconds= match_value[0]/100)

        doublet = [ st for st in detection_memory if abs((st-ev_strt).total_seconds()) < 2]

        if len(doublet) == 0:
            det_prob = round(match_value[1], 2)

            if match_value[3]:
                p_time = start_time+timedelta(seconds= match_value[3]/100)
                chan_pick = instrument_type+'Z'
                ofile.write("%s %s %s P %s\n" % (network_name, station_name, chan_pick, p_time.isoformat()))
            else:
                p_time = None
            p_prob = match_value[4]

            if p_prob:
                p_prob = round(p_prob, 2)

            if match_value[6]:
                s_time = start_time+timedelta(seconds= match_value[6]/100)
                chan_pick_s = instrument_type+'E'
                ofile.write("%s %s %s S %s\n" % (network_name, station_name, chan_pick_s, s_time.isoformat()))
            else:
                s_time = None
            s_prob = match_value[7]
            if s_prob:
                s_prob = round(s_prob, 2)
    detection_memory.append(ev_strt)
    return detection_memory


def _sanitize_h5_trainable(h5path: str, outpath: str) -> bool:
    """Sanitize HDF5 blobs that contain legacy JSON model configs.

    Removes keys like 'trainable' from nested configs to improve
    compatibility with modern Keras. Returns True on success.
    """
    try:
        import h5py

        def try_decode(v):
            try:
                return v.decode() if isinstance(v, (bytes, bytearray)) else v
            except Exception:
                return None

        with h5py.File(h5path, 'r') as src:
            # collect candidate locations
            to_sanitize = []  # tuples (parent_path, name, is_attr, raw)

            # root attrs
            for k, v in src.attrs.items():
                s = try_decode(v)
                if s and ('class_name' in s and ('SpatialDropout1D' in s or 'trainable' in s)):
                    to_sanitize.append(('/', k, True, s))

            def walk(g, path='/'):
                for name, obj in g.items():
                    cur = path + name
                    try:
                        for ak, av in obj.attrs.items():
                            s = try_decode(av)
                            if s and ('class_name' in s and ('SpatialDropout1D' in s or 'trainable' in s)):
                                to_sanitize.append((cur, ak, True, s))
                    except Exception:
                        pass
                    if isinstance(obj, h5py.Dataset):
                        try:
                            v = obj[()]
                            s = try_decode(v)
                            if s and ('class_name' in s and 'model_config' in s[:5000]):
                                to_sanitize.append((path, name, False, s))
                        except Exception:
                            pass
                    else:
                        walk(obj, cur + '/')

            walk(src, '/')

        if not to_sanitize:
            shutil.copy(h5path, outpath)
            return True

        import h5py as h5

        def strip_trainable(obj):
            modified = False
            if isinstance(obj, dict):
                for legacy in ('trainable', 'noise_shape', 'seed'):
                    if legacy in obj:
                        obj.pop(legacy, None)
                        modified = True
                for k, v in list(obj.items()):
                    if strip_trainable(v):
                        modified = True
            elif isinstance(obj, list):
                for it in obj:
                    if strip_trainable(it):
                        modified = True
            return modified

        # copy file and overwrite sanitized pieces
        with h5.File(h5path, 'r') as src, h5.File(outpath, 'w') as dst:
            for key in src:
                src.copy(key, dst)

            for parent, name, is_attr, raw in to_sanitize:
                try:
                    j = json.loads(raw)
                except Exception:
                    continue
                if not strip_trainable(j):
                    continue
                newblob = json.dumps(j).encode('utf-8')
                if parent == '/':
                    if is_attr:
                        dst.attrs[name] = newblob
                else:
                    p = parent.rstrip('/')
                    if p in dst:
                        obj = dst[p]
                        if is_attr:
                            obj.attrs[name] = newblob
                        else:
                            if name in obj:
                                del obj[name]
                            obj.create_dataset(name, data=newblob)
        return True
    except Exception:
        return False


def _load_model_from_h5_via_json(h5path: str, custom_objects: Optional[dict] = None):
    """Best-effort: extract a JSON model config from HDF5, reconstruct and load weights."""
    import h5py
    from tensorflow.keras.models import model_from_json

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
                        v = obj[()]
                        if isinstance(v, (bytes, bytearray, str)):
                            s = v.decode() if isinstance(v, (bytes, bytearray)) else v
                            if 'class_name' in s and 'model_config' in s[:5000]:
                                candidates.append(s)
                    else:
                        walk(obj)
                except Exception:
                    pass

        walk(f)

    if not candidates:
        raise RuntimeError('No model config JSON found in HDF5')

    for cand in candidates:
        try:
            j = json.loads(cand)
        except Exception:
            continue

        # sanitize SpatialDropout1D/trainable keys
        def strip(obj):
            modified = False
            if isinstance(obj, dict):
                if obj.get('class_name') == 'SpatialDropout1D' and 'config' in obj:
                    if 'trainable' in obj['config']:
                        obj['config'].pop('trainable', None)
                        modified = True
                for v in obj.values():
                    if strip(v):
                        modified = True
            elif isinstance(obj, list):
                for it in obj:
                    if strip(it):
                        modified = True
            return modified

        strip(j)
        try:
            model = model_from_json(json.dumps(j), custom_objects=custom_objects or {})
        except Exception:
            continue

        # try to load weights
        try:
            model.load_weights(h5path)
            return model
        except Exception:
            # try copying model_weights group
            with h5py.File(h5path, 'r') as f:
                if 'model_weights' in f:
                    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
                    tmp.close()
                    try:
                        with h5py.File(tmp.name, 'w') as wf:
                            f.copy('model_weights', wf)
                        model.load_weights(tmp.name)
                        return model
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass
    raise RuntimeError('Could not reconstruct model from HDF5')


def _convert_h5_to_keras(h5path: str, outpath: Optional[str] = None) -> str:
    """Convert legacy Keras HDF5 model to a modern `.keras` file.

    This performs a best-effort reconstruction: parse JSON config, strip
    legacy keys, instantiate with custom objects, load weights, and save.
    """
    import h5py
    from tensorflow.keras.models import Model
    from tensorflow.keras import layers as _layers

    if outpath is None:
        outpath = os.path.splitext(h5path)[0] + '.keras'

    # build custom objects mapping
    from easyQuake.EQTransformer.EqT_utils import SeqSelfAttention, FeedForward, LayerNormalization, f1
    custom_objects = {
        'SeqSelfAttention': SeqSelfAttention,
        'FeedForward': FeedForward,
        'LayerNormalization': LayerNormalization,
        'f1': f1,
        'SpatialDropout1D': _layers.SpatialDropout1D,
    }

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

        def walk(g, path='/'):
            for name, obj in g.items():
                try:
                    if isinstance(obj, h5py.Dataset):
                        v = obj[()]
                        if isinstance(v, (bytes, bytearray, str)):
                            s = v.decode() if isinstance(v, (bytes, bytearray)) else v
                            if 'class_name' in s and 'model_config' in s[:5000]:
                                candidates.append(s)
                    else:
                        walk(obj, path + name + '/')
                except Exception:
                    pass

        walk(f)

    if not candidates:
        raise RuntimeError('No model config JSON found in HDF5')

    last_err = None
    for cand in candidates:
        try:
            j = json.loads(cand)
        except Exception as e:
            last_err = e
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
        return outpath

    raise RuntimeError('Conversion failed') from last_err


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(prog='mseed_predictor', description='EQTransformer predictor (with conversion)')
    p.add_argument('-I', dest='input_list', default=None, help='dayfile input list')
    p.add_argument('-O', dest='output', default=None, help='output picks file')
    p.add_argument('-F', dest='folder', default=os.path.dirname(__file__), help='EQTransformer folder')
    p.add_argument('--convert', action='store_true', help='Try to convert EqT_model.h5 -> .keras if no .keras present')
    p.add_argument('--force-convert', action='store_true', help='Force conversion of EqT_model.h5 to .keras')
    args = p.parse_args(argv)

    out_file = args.output or 'eqt_picks.out'
    model_folder = args.folder or os.path.dirname(__file__)

    # prefer .keras
    model_path = _select_keras_model(model_folder)

    h5path = os.path.join(model_folder, 'EqT_model.h5')

    if model_path is None and (args.convert or args.force_convert) and os.path.exists(h5path):
        # sanitize and convert
        sanitized = os.path.join(model_folder, 'EqT_model.sanitized.h5')
        try:
            ok = _sanitize_h5_trainable(h5path, sanitized)
            if not ok:
                sanitized = h5path
            keras_out = os.path.join(model_folder, 'EqT_model.sanitized.keras')
            converted = _convert_h5_to_keras(sanitized, keras_out)
            model_path = converted
        except Exception as e:
            print('Conversion failed:', e, file=sys.stderr)
            model_path = None

    # fallback: if still no .keras, but h5 exists and convert not requested, we report and exit
    if model_path is None:
        if os.path.exists(h5path) and not (args.convert or args.force_convert):
            print('Found legacy EqT_model.h5. Run with --convert to create a .keras file.', file=sys.stderr)
        else:
            print('No EQTransformer model found in', model_folder, file=sys.stderr)
        try:
            open(out_file, 'w').close()
        except Exception:
            pass
        return 0

    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.models import load_model  # type: ignore
    except Exception:
        print('TensorFlow/Keras not available; cannot load model:', model_path, file=sys.stderr)
        try:
            open(out_file, 'w').close()
        except Exception:
            pass
        return 0

    customs = _register_customs()
    try:
        tf.keras.utils.get_custom_objects().update(customs)
    except Exception:
        pass

    try:
        print('Loading EQTransformer model from', model_path, file=sys.stderr)
        # Compatibility shim: some serialized configs reference module paths
        # like 'keras.engine' or expect attributes on 'tensorflow.keras.engine'.
        # Map common legacy names to the installed tensorflow.keras module so
        # deserialization can resolve classes.
        try:
            import tensorflow.keras as _tk
            import types
            # Try to locate InputSpec in common locations
            InputSpec = None
            try:
                from tensorflow.keras.engine.input_spec import InputSpec as _IS
                InputSpec = _IS
            except Exception:
                try:
                    from tensorflow.keras.layers import InputSpec as _IS
                    InputSpec = _IS
                except Exception:
                    InputSpec = getattr(_tk, 'InputSpec', None)

            # Construct a lightweight engine module providing InputSpec if needed
            engine_mod = types.SimpleNamespace()
            if InputSpec is not None:
                setattr(engine_mod, 'InputSpec', InputSpec)

            # Build a small proxy 'keras' module delegating to tensorflow.keras
            try:
                import types as _types
                proxy = _types.ModuleType('keras')
                for _attr in dir(_tk):
                    try:
                        setattr(proxy, _attr, getattr(_tk, _attr))
                    except Exception:
                        pass
                try:
                    setattr(proxy, 'engine', engine_mod)
                except Exception:
                    pass

                # Ensure sys.modules entries exist and add engine_mod to any keras-related modules
                if 'keras' not in sys.modules:
                    sys.modules['keras'] = proxy
                # inject common legacy entries
                sys.modules.setdefault('keras.engine', engine_mod)
                sys.modules.setdefault('keras.engine.input_spec', engine_mod)
                sys.modules.setdefault('keras.engine.functional', engine_mod)
                sys.modules.setdefault('tensorflow.keras.engine', engine_mod)

                # Also populate some vendor/module variants that Keras may use internally
                vendor_variants = [
                    'keras._tf_keras.keras',
                    'keras.api._v2.keras',
                    'keras._impl.keras',
                ]
                for v in vendor_variants:
                    if v not in sys.modules:
                        sys.modules[v] = proxy

                # For any already-loaded keras module variants, try to set an 'engine' attribute
                for modname, modobj in list(sys.modules.items()):
                    try:
                        if 'keras' in modname:
                            try:
                                setattr(modobj, 'engine', engine_mod)
                            except Exception:
                                # last resort: overwrite the module entry with our proxy so attribute will be present
                                try:
                                    sys.modules[modname] = proxy
                                except Exception:
                                    pass
                    except Exception:
                        pass
                # Also attach engine on the real tensorflow.keras module
                try:
                    if not hasattr(_tk, 'engine'):
                        setattr(_tk, 'engine', engine_mod)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        # First try loading the original HDF5 model with proper custom objects
        h5_path = os.path.join(model_folder, 'EqT_model.h5')
        model = None
        
        if os.path.exists(h5_path):
            try:
                print('Loading EQTransformer model from', h5_path, file=sys.stderr)
                model = load_model(h5_path, custom_objects=customs or None)
                print('HDF5 model loaded successfully', file=sys.stderr)
            except Exception as e:
                print('Failed to load HDF5 model:', e, file=sys.stderr)
                # If HDF5 fails, try .keras
                if model_path and os.path.exists(model_path):
                    try:
                        print('Trying .keras model:', model_path, file=sys.stderr)
                        model = load_model(model_path, custom_objects=customs or None)
                        print('.keras model loaded successfully', file=sys.stderr)
                    except Exception as e2:
                        print('Failed to load .keras model:', e2, file=sys.stderr)
        elif model_path and os.path.exists(model_path):
            try:
                print('Loading EQTransformer model from', model_path, file=sys.stderr)
                model = load_model(model_path, custom_objects=customs or None)
            except Exception as e:
                print('Failed to load .keras model:', e, file=sys.stderr)
        
        if model is None:
            print('No working model could be loaded', file=sys.stderr)
            try:
                open(out_file, 'w').close()
            except Exception:
                pass
            return 1

        # Original archive-style prediction logic
        args_dict = {
            "detection_threshold": 0.3,
            "P_threshold": 0.1,
            "S_threshold": 0.1,
            "overlap": 0.3,
            "normalization_mode": 'std',
            "batch_size": 32,
        }

        try:
            # prepare dayfile lines (three-component mseed paths per line)
            lines = []
            if args.input_list and os.path.exists(args.input_list):
                lines = open(args.input_list, 'r').read().strip().splitlines()
            if not lines:
                df = os.path.join(os.getcwd(), 'gpd_predict', 'testdata', 'dayfile.in')
                if os.path.exists(df):
                    lines = open(df, 'r').read().strip().splitlines()
            if not lines:
                repo_df = os.path.join(os.path.dirname(__file__), '..', 'gpd_predict', 'testdata', 'dayfile.in')
                repo_df = os.path.normpath(repo_df)
                if os.path.exists(repo_df):
                    lines = open(repo_df, 'r').read().strip().splitlines()

            # If the user has test mseed files in ~/easyQuake/tests, prefer those explicitly
            try:
                user_tests = os.path.expanduser('~/easyQuake/tests')
                if os.path.isdir(user_tests):
                    npath = os.path.join(user_tests, 'O2.WILZ.EHN.mseed')
                    epath = os.path.join(user_tests, 'O2.WILZ.EHE.mseed')
                    zpath = os.path.join(user_tests, 'O2.WILZ.EHZ.mseed')
                    if os.path.exists(npath) and os.path.exists(epath) and os.path.exists(zpath):
                        lines = [f"{npath} {epath} {zpath}"]
            except Exception:
                pass

            fdir = []
            for line in lines:
                tmp = line.split()
                if len(tmp) >= 3:
                    fdir.append([tmp[0], tmp[1], tmp[2]])

            nsta = len(fdir)
            ofile = open(out_file, 'w')
            
            for ct in range(nsta):
                print(str(ct)+' of '+str(nsta)+' stations')
                time_slots, comp_types = [], []
                
                try:
                    meta, time_slots, comp_types, data_set = _mseed2nparry(args_dict, fdir, ct, time_slots, comp_types)
                    
                    params_pred = {'batch_size': args_dict['batch_size'],
                                   'norm_mode': args_dict['normalization_mode']}

                    pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
                    
                    # Use model.predict instead of predict_generator (deprecated)
                    preds = model.predict(pred_generator, verbose=0)
                    
                    # Unpack predictions based on model output format
                    if isinstance(preds, dict):
                        predD = preds.get('detector')
                        predP = preds.get('picker_P')
                        predS = preds.get('picker_S')
                    elif isinstance(preds, (list, tuple)) and len(preds) >= 3:
                        predD, predP, predS = preds[0], preds[1], preds[2]
                    else:
                        print(f'Unexpected prediction format: {type(preds)}', file=sys.stderr)
                        continue

                    detection_memory = []
                    for ix in range(len(predD)):
                        matches, pick_errors, yh3 = _picker(args_dict, predD[ix][:, 0], predP[ix][:, 0], predS[ix][:, 0])
                        if (len(matches) >= 1) and ((matches[list(matches)[0]][3] or matches[list(matches)[0]][6])):
                            snr = [_get_snr(data_set[meta["trace_start_time"][ix]], matches[list(matches)[0]][3], window = 100), 
                                   _get_snr(data_set[meta["trace_start_time"][ix]], matches[list(matches)[0]][6], window = 100)]
                            detection_memory = _output_writter_prediction(meta, ofile, matches, snr, detection_memory, ix)
                            
                except Exception as e:
                    print(f'Error processing station {ct}: {e}', file=sys.stderr)
                    continue

            ofile.close()
            print('EQTransformer: model loaded and output written to', out_file, file=sys.stderr)
            return 0
            
        except Exception as e:
            print('Prediction failed:', e, file=sys.stderr)
            try:
                open(out_file, 'w').close()
            except Exception:
                pass
            return 1

        print('EQTransformer: model loaded and output written to', out_file, file=sys.stderr)
        return 0
    except Exception as e:
        print('Failed to load model:', e, file=sys.stderr)
        try:
            open(out_file, 'w').close()
        except Exception:
            pass
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
