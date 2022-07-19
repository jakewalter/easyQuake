#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:44:47 2022

@author: jwalter
"""
import obspy
import os
import shutil
from obspy import read,UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset, classic_sta_lta, plot_trigger
# how to pick phases using STA/LTA method? Please refer to
# https://docs.obspy.org/tutorial/code_snippets/trigger_tutorial.html
import numpy as np


    # infile = dir1+'/dayfile.in'
    # outfile = dir1+'/gpd_picks.out'
    
def recSTALTAPy_h(a, b, nsta, nlta):
    """
    Recursive STA/LTA written in Python.
    .. note::
        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.recSTALTA` in this module!
    :type a: NumPy ndarray
    :param a: Seismic Trace
    :type nsta: Int
    :param nsta: Length of short time average window in samples
    :type nlta: Int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy ndarray
    :return: Characteristic function of recursive STA/LTA
    .. seealso:: [Withers1998]_ (p. 98) and [Trnkoczy2012]_
    """
    try:
        a = a.tolist()
    except:
        pass

    try:
        b = b.tolist()
    except:
        pass
    ndat = len(a)
    # compute the short time average (STA) and long time average (LTA)
    csta = 1. / nsta
    clta = 1. / nlta
    sta = 0.
    lta = 1e-99  # avoid zero devision
    charfct = [0.0] * len(a)
    icsta = 1 - csta
    iclta = 1 - clta
    for i in range(1, ndat):
        sq = a[i] ** 2 + b[i] ** 2
        sta = csta * sq + icsta * sta
        lta = clta * sq + iclta * lta
        charfct[i] = sta / lta
        if i < nlta:
            charfct[i] = 0.
    return np.array(charfct)



def trigger_p_s(fdir, i, outfilea):
    f = open(outfilea+str(i),'w')

    stn = read(fdir[i][0])
    stn.merge(method=1)
    for tr in stn:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled()
    ste = read(fdir[i][1])
    ste.merge(method=1)
    for tr in ste:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled()
    stz = read(fdir[i][2])
    stz.merge(method=1)
    for tr in stz:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled()
    ste.detrend('demean') 
    ste.detrend('linear') 
    stn.detrend('demean') 
    stn.detrend('linear') 
    ste.filter(type="bandpass",freqmin=2.0,freqmax=15.0,zerophase=False)
    stn.filter(type="bandpass",freqmin=2.0,freqmax=15.0,zerophase=False)
    trz = stz[0]  
    tre = ste[0]
    trn = stn[0]

    trz.detrend('demean') 
    trz.detrend('linear') 
    trz.filter(type="bandpass",freqmin=2.0,freqmax=24.0,zerophase=False)
    df = trz.stats.sampling_rate
    #tstart = trz.stats.starttime - UTCDateTime(int(dirname[0:4]),int(dirname[4:6]),int(dirname[6:8])) 
    dfe = tre.stats.sampling_rate
    #tstarte = tre.stats.starttime - UTCDateTime(int(dirname[0:4]),int(dirname[4:6]),int(dirname[6:8]))
    latest_start = np.max([tre.stats.starttime,trn.stats.starttime])
    earliest_stop = np.min([tre.stats.endtime,trn.stats.endtime])
    if (earliest_stop>latest_start):
        trn.trim(latest_start, earliest_stop)
        tre.trim(latest_start, earliest_stop)
    cfte = recSTALTAPy_h(tre.data, trn.data, int(0.2 * df), int(2.5 * df))
    on_ofe = trigger_onset(cfte, 4.0, 2.0)
    cft = recursive_sta_lta(trz.data, int(0.1 * df), int(2.5 * df))
    on_of = trigger_onset(cft, 6.0, 2.0)
    
    i = 0
    while(i<len(on_ofe)):
        trig_one = on_ofe[i,0]
        trig_ofe = on_ofe[i,1]
        trig_offe = int(trig_ofe + (trig_ofe - trig_one)*4.0)
        if trig_offe >= tre.stats.npts - 1:
            break
        if max(cft[trig_one:trig_ofe]) > 6.0:
            f.write("%s %s %s S %s\n" % (tre.stats.network, tre.stats.station, tre.stats.channel, (tre.stats.starttime+trig_one/dfe).isoformat()))
        i=i+1
        
    i = 0
    while(i<len(on_of)):
        trig_on = on_of[i,0]
        trig_of = on_of[i,1]
        trig_off = int(trig_of + (trig_of - trig_on)*4.0 + 3*df) 
        if trig_off >= trz.stats.npts - 1:
            break
        if max(cft[trig_on:trig_of]) > 10.0:
            f.write("%s %s %s P %s\n" % (trz.stats.network, trz.stats.station, trz.stats.channel, (trz.stats.starttime+trig_on/df).isoformat()))
        i=i+1

    # except:
    #     print('no station, skip the station')
    f.close()

if __name__ == "__main__":
    main()
