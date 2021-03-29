#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feb 23 2021 version:
https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/mseed_predictor.py

Remove plotting, as not needed
"""

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import csv
import keras
import time
from os import listdir
import os
import platform
import shutil
from tqdm import tqdm
from datetime import datetime, timedelta
import contextlib
import sys
import warnings
from obspy import read
from os.path import join
import json
import pickle
import faulthandler; faulthandler.enable()
import obspy
import logging
import argparse as ap

from obspy.signal.trigger import trigger_onset
from EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
        


    


            


def _get_snr(data, pat, window=200):
    
    """ 
    
    Estimates SNR.
    
    Parameters
    ----------
    data : numpy array
        3 component data.    
        
    pat: positive integer
        Sample point where a specific phase arrives. 
        
    window: positive integer, default=200
        The length of the window for calculating the SNR (in the sample).         
        
    Returns
   --------   
    snr : {float, None}
       Estimated SNR in db. 
       
        
    """      
       
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)    
        except Exception:
            pass
    return snr 



def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    
    """

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

    """ 
    
    Performs detection and picking.
    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
        Detection probabilities. 
        
    yh2 : 1D array
        P arrival probabilities.  
        
    yh3 : 1D array
        S arrival probabilities. 
        
   
    Returns
    -------    
    matches : dic
        Contains the information for the detected and picked event.            
        
    matches : dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
        
    yh3 : 1D array             
        normalized S_probability                              
                
    """                
             
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
    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []
        
        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a,x] for x in l2[b:e]])
            
        best_pair = None
        for pr in ans: 
            ds = pr[1]-pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds           
        return best_pair


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



def _resampling(st):
    'perform resampling on Obspy stream objects'
    
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
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



def _normalize(data, mode = 'max'):  
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """  
       
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



class PreLoadGeneratorTest(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. Pre-load version.
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32.
        Batch size.
            
    n_channels: int, default=3.
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'                
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    
    """  
    
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
            print("Your data duration in mseed file is too short! Try either longer files or reducing batch_size. ")
        
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def _normalize(self, data, mode = 'max'):          
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
        'readint the waveforms' 
        X = np.zeros((self.batch_size, 6000, 3))           
        # Generate data
        for i, ID in enumerate(list_IDs_temp):            
            data = self.inp_data[ID]
            data = self._normalize(data, self.norm_mode)                            
            X[i, :, :] = data                                                           
                           
        return X
    
    
    

def _mseed2nparry(args, fdir, i, time_slots, comp_types):
    ' read miniseed files and from a list of string names and returns 3 dictionaries of numpy arrays, meta data, and time slice info'
    
    netname = fdir[i][0].split("/")[-1].split('.')[0]
    st = obspy.core.Stream()
    st += obspy.core.read(fdir[i][0])
    st += obspy.core.read(fdir[i][1])
    st += obspy.core.read(fdir[i][2])
    #st.resample(100)
    try:
       st.merge(fill_value=0)                     
    except Exception:
        st =_resampling(st)
        st.merge(fill_value=0) 
    #print(st)
    for tr in st:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled()
    
    st.detrend(type='linear')
    st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
    st.taper(max_percentage=0.001, type='cosine', max_length=2) 
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




detection_threshold=0.3           
P_threshold=0.1
S_threshold=0.1
loss_weights=[0.03, 0.40, 0.58]
loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
normalization_mode='std'
batch_size=500          
overlap = 0.3
gpuid=None
gpu_limit=None
overwrite=False


#try:
#    f = open('setup.py')
#    for li, l in enumerate(f):
#        if li == 8:
#            EQT_VERSION = l.split('"')[1]
#except Exception:
#    EQT_VERSION = "0.1.59"
#    

#os.system(fullpath1+" -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
def _output_writter_prediction(meta, ofile, matches, snr, detection_memory, idx):
    
    """ 
    
    Writes the detection & picking results into a CSV file.
    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.
    predict_writer: obj
        For writing out the detection/picking results in the CSV file. 
       
    csvPr: obj
        For writing out the detection/picking results in the CSV file.  
    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    #station_lat = meta["receiver_latitude"]
    #station_lon = meta["receiver_longitude"]
    #station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    #station_name = "{:<4}".format(station_name).replace(" ","")
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
            
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
                        












if __name__ == "__main__":
    parser = ap.ArgumentParser(
        prog='mseed_predictor.py',
        description='Automatic picking EQTransformer')
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
        '-F',
        type=str,
        default=None,
        help='path where EQTransformer lives')
    
    args1 = parser.parse_args()
    input_model=args1.F+"/EqT_model.h5"
        
 
    args = {
    "I": args1.I,
    "O": args1.O,
    "input_model": input_model,
    "detection_threshold": detection_threshold,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "loss_weights": loss_weights,     
    "loss_types": loss_types,
    "normalization_mode": normalization_mode,
    "overlap": overlap,
    "batch_size": batch_size,    
    "gpuid": gpuid,
    "gpu_limit": gpu_limit 
    }        
        
    if args['gpuid']:     
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args['gpuid'])
        tf.Session(config=tf.ConfigProto(log_device_placement=True))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
        K.tensorflow_backend.set_session(tf.Session(config=config))          

    model = load_model(args['input_model'], 
                       custom_objects={'SeqSelfAttention': SeqSelfAttention, 
                                       'FeedForward': FeedForward,
                                       'LayerNormalization': LayerNormalization, 
                                       'f1': f1                                                                            
                                        })              
    model.compile(loss = args['loss_types'],
                  loss_weights = args['loss_weights'],           
                  optimizer = Adam(lr = 0.001),
                  metrics = [f1])
    # print('*** Loading is complete!', flush=True)  


        
    fdir = []
    evid = []
    staid = []
    with open(args["I"]) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)
    
    #station_list = sorted(set(station_list))
    
    data_track = dict()
    ofile = open(args["O"], 'w')
    # print(f"######### There are files for {len(station_list)} stations in {args['input_dir']} directory. #########", flush=True)
    for ct in range(nsta):

            
        plt_n = 0            
        #csvPr_gen = open('X_prediction_results.csv', 'w')          

        # print(f'========= Started working on {st}, {ct+1} out of {len(station_list)} ...', flush=True)        
        
        start_Predicting = time.time()       
        
        #file_list = [join(st, ev) for ev in listdir(args["input_dir"]+"/"+st) if ev.split("/")[-1].split(".")[-1].lower() == "mseed"]; 
        
        #mon = [ev.split('__')[1]+'__'+ev.split('__')[2] for ev in file_list ];
#        uni_list = list(set(mon))
#        uni_list.sort()  
          
        time_slots, comp_types = [], []

        meta, time_slots, comp_types, data_set = _mseed2nparry(args, fdir, ct, time_slots, comp_types)
        print(str(ct)+' of '+str(nsta)+' stations')
        params_pred = {'batch_size': args['batch_size'],
                       'norm_mode': args['normalization_mode']}  
            
        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)

        predD, predP, predS = model.predict_generator(pred_generator)
        
        detection_memory = []
        for ix in range(len(predD)):
            matches, pick_errors, yh3 =  _picker(args, predD[ix][:, 0], predP[ix][:, 0], predS[ix][:, 0])        
            if (len(matches) >= 1) and ((matches[list(matches)[0]][3] or matches[list(matches)[0]][6])):
                snr = [_get_snr(data_set[meta["trace_start_time"][ix]], matches[list(matches)[0]][3], window = 100), _get_snr(data_set[meta["trace_start_time"][ix]], matches[list(matches)[0]][6], window = 100)]
                pre_write = len(detection_memory)
                detection_memory=_output_writter_prediction(meta, ofile, matches, snr, detection_memory, ix)
                post_write = len(detection_memory)
                                                       
        end_Predicting = time.time() 
        data_track[ct]=[time_slots, comp_types] 
        delta = (end_Predicting - start_Predicting) 
        hour = int(delta / 3600)
        delta -= hour * 3600
        minute = int(delta / 60)
        delta -= minute * 60
        seconds = delta     
                        
#        dd = pd.read_csv('X_prediction_results.csv')
#        print(f'\n', flush=True)
        # print(' *** Finished the prediction in: {} hours and {} minutes and {} seconds.'.format(hour, minute, round(seconds, 2)), flush=True)         
        # print(' *** Detected: '+str(len(dd))+' events.', flush=True)
        # print(' *** Wrote the results into --> " ' + str(save_dir)+' "', flush=True)
        
    ofile.close()
    with open('time_tracks.pkl', 'wb') as f:
        pickle.dump(data_track, f, pickle.HIGHEST_PROTOCOL)

       
        

