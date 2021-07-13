#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
set of functions to drive EasyQuake
"""
#import sys
#sys.path.append("/home/jwalter/syncpython")
from .phasepapy import fbpicker
pathgpd = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/gpd_predict'
pathEQT = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/EQTransformer'
pathhyp = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/hyp2000'

from .phasepapy import tables1D, assoc1D
from .phasepapy import tt_stations_1D
import os
st = os.stat(pathgpd+'/gpd_predict.py')
st1 = os.stat(pathEQT+'/mseed_predictor.py')
import stat

os.chmod(pathgpd+'/gpd_predict.py', st.st_mode | stat.S_IEXEC)
os.chmod(pathEQT+'/mseed_predictor.py', st1.st_mode | stat.S_IEXEC)

import os
from obspy import UTCDateTime
from obspy import Inventory, read_inventory
from obspy.clients.fdsn import Client
from obspy import read
import datetime
import numpy as np
import glob
#import sys

import obspy.taup as taup
from obspy import geodetics
from obspy.clients.fdsn.mass_downloader import CircularDomain, RectangularDomain, Restrictions, MassDownloader
from obspy.core.event.base import WaveformStreamID
from sqlalchemy.orm import *
from sqlalchemy import create_engine
import pandas as pd
import sqlite3
from sqlite3 import Error
from obspy.geodetics import gps2dist_azimuth

import pylab as plt
import re
from datetime import datetime
#from mpl_toolkits.basemap import Basemap


from obspy import Stream
from obspy.core.event import Catalog, Event, Magnitude, Origin, Pick, StationMagnitude, Amplitude, Arrival, OriginUncertainty, OriginQuality, ResourceIdentifier
#from obspy.signal.invsim import simulate_seismometer as seis_sim
fmtP = "%4s%1sP%1s%1i %15s"
fmtS = "%12s%1sS%1s%1i\n"



fmt = "%6s%02i%05.2f%1s%03i%05.2f%1s%4i\n"


#min_proba = 0.993 # Minimum softmax probability for phase detection
## try 0.992 if you have the computing power
#freq_min = 3.0
#freq_max = 20.0
#filter_data = True
#decimate_data = True # If false, assumes data is already 100 Hz samprate
#n_shift = 10 # Number of samples to shift the sliding window at a time
#n_gpu = 1 # Number of GPUs to use (if any)
######################
#batch_size = 1000*3
#
#half_dur = 2.00
#only_dt = 0.01
#n_win = int(half_dur/only_dt)
#n_feat = 2*n_win


from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

class SCNL():
    """ This class is copied from PhasePaPy"""
    def __init__(self,input=None):
        if not isinstance(input, SCNL):
            self.station=None
            self.channel=None
            self.network=None
            self.location=None
        if type(input) is str:
            self.parse_scnlstr(input)
        if type(input) is list:
            if len(input)==4:
                self.station,self.channel,self.network,self.location=input
            if len(input)==3:
                self.station,self.channel,self.network=input


def download_mseed(dirname=None, project_folder=None, single_date=None, minlat=None, maxlat=None, minlon=None, maxlon=None, dense=False):
    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
    stopping = starting + 86430
    starttime = starting
    endtime = stopping
    #domain = CircularDomain(-90,0,minradius=0.0, maxradius=30.0)
    domain = RectangularDomain(minlatitude=minlat, maxlatitude=maxlat,minlongitude=minlon, maxlongitude=maxlon)
    #domain = RectangularDomain(minlatitude=-90, maxlatitude=-60,minlongitude=-180, maxlongitude=180)
    if dense:
        restrictions = Restrictions(starttime=starttime, endtime=endtime,reject_channels_with_gaps=False,minimum_length=0,minimum_interstation_distance_in_m=1, channel_priorities=["HH[ZNE12]", "BH[ZNE12]","EH[ZNE12]","SH[ZNE12]","HN[ZNE12]","EN[ZNE12]"])
    else:
        restrictions = Restrictions(starttime=starttime, endtime=endtime,reject_channels_with_gaps=False,minimum_length=0,minimum_interstation_distance_in_m=5000, channel_priorities=["HH[ZNE12]", "BH[ZNE12]","EH[ZNE12]","SH[ZNE12]","HN[ZNE12]","EN[ZNE12]"])
    mseed1 = project_folder+'/'+dirname
    if not os.path.exists(mseed1):
        os.makedirs(mseed1) #domain = CircularDomain(-90,0,minradius=0.0, maxradius=30.0)
    #original1 = project_folder+'/*.[BH]??__'+dirname+'*'
    #os.system("mv %s %s" % (original1,mseed1))
    mdl = MassDownloader()
    mdl.download(domain, restrictions, threads_per_client=4, mseed_storage=mseed1,stationxml_storage=mseed1)

def download_mseed_event(dirname=None, project_folder=None, starting=None, stopping = None, minlat=None, maxlat=None, minlon=None, maxlon=None, maxrad=None):
    starttime = starting
    endtime = stopping
    #domain = CircularDomain(lat1,lon1,minradius=0.0, maxradius=maxrad)
    domain = RectangularDomain(minlatitude=minlat, maxlatitude=maxlat,minlongitude=minlon, maxlongitude=maxlon)
    #domain = RectangularDomain(minlatitude=-90, maxlatitude=-60,minlongitude=-180, maxlongitude=180)
    restrictions = Restrictions(starttime=starttime, endtime=endtime,chunklength_in_sec=86400,reject_channels_with_gaps=False,minimum_length=0,minimum_interstation_distance_in_m=5000, channel_priorities=["HH[ZNE12]", "BH[ZNE12]","EH[ZNE12]","SH[ZNE12]","HN[ZNE12]","EN[ZNE12]"])
    mseed1 = project_folder+'/'+dirname
    if not os.path.exists(mseed1):
        os.makedirs(mseed1) #domain = CircularDomain(-90,0,minradius=0.0, maxradius=30.0)
    #original1 = project_folder+'/*.[BH]??__'+dirname+'*'
    #os.system("mv %s %s" % (original1,mseed1))
    mdl = MassDownloader()
    mdl.download(domain, restrictions, threads_per_client=4, mseed_storage=mseed1,stationxml_storage=mseed1)

def download_mseed_event_radial(dirname=None, project_folder=None, starting=None, stopping = None, lat1=None, lon1=None, maxrad=None):
    starttime = starting
    endtime = stopping
    domain = CircularDomain(lat1,lon1,minradius=0.0, maxradius=maxrad)
    #domain = RectangularDomain(minlatitude=minlat, maxlatitude=maxlat,minlongitude=minlon, maxlongitude=maxlon)
    #domain = RectangularDomain(minlatitude=-90, maxlatitude=-60,minlongitude=-180, maxlongitude=180)
    restrictions = Restrictions(starttime=starttime, endtime=endtime,chunklength_in_sec=86400,reject_channels_with_gaps=False,minimum_length=0,minimum_interstation_distance_in_m=1000, channel_priorities=["HH[ZNE12]", "BH[ZNE12]","EH[ZNE12]","SH[ZNE12]","HN[ZNE12]","EN[ZNE12]"])
    mseed1 = project_folder+'/'+dirname
    if not os.path.exists(mseed1):
        os.makedirs(mseed1) #domain = CircularDomain(-90,0,minradius=0.0, maxradius=30.0)
    #original1 = project_folder+'/*.[BH]??__'+dirname+'*'
    #os.system("mv %s %s" % (original1,mseed1))
    mdl = MassDownloader()
    mdl.download(domain, restrictions, threads_per_client=4, mseed_storage=mseed1,stationxml_storage=mseed1)

def process_local_sac():
    print('Local sac files')



def build_tt_tables(lat1=None,long1=None,maxrad=None,starting=None, stopping=None, channel_codes=['EH','BH','HH','HN'],db=None,maxdist=500.,source_depth=5., delta_distance=1):
    """
    """
    # Create a connection to an sqlalchemy database
    tt_engine=create_engine(db,echo=False, connect_args={'check_same_thread': False})
    tt_stations_1D.BaseTT1D.metadata.create_all(tt_engine)
    TTSession=sessionmaker(bind=tt_engine)
    tt_session=TTSession()
    fdsnclient=Client()
    inv=fdsnclient.get_stations(starttime=starting,endtime=stopping,latitude=lat1,longitude=long1,maxradius=maxrad,channel='*HZ',level='channel')
    # Get inventory
    for net in inv:
        network=net.code
        for sta in net:
            loccodes=[]
            for ch in sta:
                for cc in channel_codes:
                  if re.match(cc,ch.code):
                    if not ch.location_code in loccodes:
                      loccodes.append(ch.location_code)
            for loc in loccodes:
                print(sta.code,network,loc,sta.latitude,sta.longitude,sta.elevation)
                station=tt_stations_1D.Station1D(sta.code,network,loc,sta.latitude,sta.longitude,sta.elevation)
                tt_session.add(station)
            tt_session.commit()

    # Now we have to build our traveltime lookup tables
    # We will use IASP91 here but obspy.taup does let you build your own model
    velmod=taup.TauPyModel(model='iasp91')
    #delta_distance=1. # km for spacing tt calculations
    distance_km=np.arange(0,maxdist+delta_distance,delta_distance)
    for d_km in distance_km:
        d_deg=geodetics.kilometer2degrees(d_km)
        ptimes=[]
        stimes=[]
        p_arrivals=velmod.get_travel_times(source_depth_in_km=source_depth,
          distance_in_degree=d_deg,phase_list=['P','p'])
        for p in p_arrivals:
            ptimes.append(p.time)
        s_arrivals=velmod.get_travel_times(source_depth_in_km=source_depth,
            distance_in_degree=d_deg,phase_list=['S','s'])
        for s in s_arrivals:
            stimes.append(s.time)
        tt_entry=tt_stations_1D.TTtable1D(d_km,d_deg,np.min(ptimes),np.min(stimes),np.min(stimes)-np.min(ptimes))
        tt_session.add(tt_entry)
        tt_session.commit() # Probably faster to do the commit outside of loop but oh well
    tt_session.close()
    return inv

def build_tt_tables_local_directory(dirname=None,project_folder=None,channel_codes=['EH','BH','HH','HN'],db=None,maxdist=800.,source_depth=5.,delta_distance=1):
    """
    """
    # Create a connection to an sqlalchemy database
    tt_engine=create_engine(db,echo=False, connect_args={'check_same_thread': False})
    tt_stations_1D.BaseTT1D.metadata.create_all(tt_engine)
    TTSession=sessionmaker(bind=tt_engine)
    tt_session=TTSession()
    inv = Inventory()
    dir1a = glob.glob(project_folder+'/'+dirname+'/*xml')
    for file1 in dir1a:
        inv1a = read_inventory(file1)
        inv.networks.extend(inv1a)
    for net in inv:
        network=net.code
        for sta in net:
            loccodes=[]
            for ch in sta:
                for cc in channel_codes:
                  if re.match(cc,ch.code):
                    if not ch.location_code in loccodes:
                      loccodes.append(ch.location_code)
            for loc in loccodes:
                print(sta.code,network,loc,sta.latitude,sta.longitude,sta.elevation)
                station=tt_stations_1D.Station1D(sta.code,network,loc,sta.latitude,sta.longitude,sta.elevation)
                tt_session.add(station)
            tt_session.commit()

    # Now we have to build our traveltime lookup tables
    # We will use IASP91 here but obspy.taup does let you build your own model
    velmod=taup.TauPyModel(model='iasp91')
    #delta_distance=1. # km for spacing tt calculations
    distance_km=np.arange(0,maxdist+delta_distance,delta_distance)
    for d_km in distance_km:
        d_deg=geodetics.kilometer2degrees(d_km)
        ptimes=[]
        stimes=[]
        p_arrivals=velmod.get_travel_times(source_depth_in_km=source_depth,
          distance_in_degree=d_deg,phase_list=['P','p'])
        for p in p_arrivals:
            ptimes.append(p.time)
        s_arrivals=velmod.get_travel_times(source_depth_in_km=source_depth,
            distance_in_degree=d_deg,phase_list=['S','s'])
        for s in s_arrivals:
            stimes.append(s.time)
        tt_entry=tt_stations_1D.TTtable1D(d_km,d_deg,np.min(ptimes),np.min(stimes),np.min(stimes)-np.min(ptimes))
        tt_session.add(tt_entry)
        tt_session.commit() # Probably faster to do the commit outside of loop but oh well
    tt_session.close()
    return inv



def build_tt_tables_local_directory_ant(dirname=None,project_folder=None,channel_codes=['EH','BH','HH'],db=None,maxdist=800.,source_depth=5.,delta_distance=1):
    """
    """
    # Create a connection to an sqlalchemy database
    tt_engine=create_engine(db,echo=False, connect_args={'check_same_thread': False})
    tt_stations_1D.BaseTT1D.metadata.create_all(tt_engine)
    TTSession=sessionmaker(bind=tt_engine)
    tt_session=TTSession()
    inv = Inventory()
    dir1a = glob.glob(project_folder+'/'+dirname+'/*xml')
    m = Basemap(projection='spstere',boundinglat=-60,lon_0=180,resolution='i')

    for file1 in dir1a:
        inv1a = read_inventory(file1)
        inv.networks.extend(inv1a)
    for net in inv:
        network=net.code
        for sta in net:
            loccodes=[]
            for ch in sta:
                for cc in channel_codes:
                  if re.match(cc,ch.code):
                    if not ch.location_code in loccodes:
                      loccodes.append(ch.location_code)
            for loc in loccodes:
                print(sta.code,network,loc,sta.latitude,sta.longitude,sta.elevation)
                x,y = m(sta.longitude,sta.latitude)

                station=tt_stations_1D.Station1D(sta.code,network,loc,y,x,sta.elevation)
                tt_session.add(station)
            tt_session.commit()

    # Now we have to build our traveltime lookup tables
    # We will use IASP91 here but obspy.taup does let you build your own model
    velmod=taup.TauPyModel(model='iasp91')
    #delta_distance=1. # km for spacing tt calculations
    distance_km=np.arange(0,maxdist+delta_distance,delta_distance)
    for d_km in distance_km:
        d_deg=geodetics.kilometer2degrees(d_km)
        ptimes=[]
        stimes=[]
        p_arrivals=velmod.get_travel_times(source_depth_in_km=source_depth,
          distance_in_degree=d_deg,phase_list=['P','p'])
        for p in p_arrivals:
            ptimes.append(p.time)
        s_arrivals=velmod.get_travel_times(source_depth_in_km=source_depth,
            distance_in_degree=d_deg,phase_list=['S','s'])
        for s in s_arrivals:
            stimes.append(s.time)
        tt_entry=tt_stations_1D.TTtable1D(d_km,d_deg,np.min(ptimes),np.min(stimes),np.min(stimes)-np.min(ptimes))
        tt_session.add(tt_entry)
        tt_session.commit() # Probably faster to do the commit outside of loop but oh well
    tt_session.close()
    return inv

def fb_pick(dbengine=None,picker=None,fileinput=None):
    fdir = []
    engine_assoc=dbengine
    with open(fileinput) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
        nsta = len(fdir)

        for i in range(nsta):
            Session=sessionmaker(bind=engine_assoc)
            dbsession=Session()
            st = Stream()
            st += read(fdir[i][0])
            st += read(fdir[i][1])
            st += read(fdir[i][2])
            st.merge(fill_value='interpolate')
            #print(st)
            for tr in st:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled()
            st.detrend(type='linear')
            for tr in st:
                print(tr)
                scnl,picks,polarity,snr,uncert=picker.picks(tr)
                t_create=datetime.utcnow()
                for i in range(len(picks)):
                    new_pick=tables1D.Pick(scnl,picks[i].datetime,polarity[i],snr[i],uncert[i],t_create)
                    dbsession.add(new_pick)

def gpd_pick_add(dbsession=None,fileinput=None,inventory=None):
    filepath = fileinput
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            try:
                line = fp.readline()
                #print(line)
                cnt += 1
                if len(line.split())>4:
                    sta1 = line.split()[1]
                    chan1 = line.split()[2]
                    #print(sta1,chan1)
                    #scnl.station = sta1
                    net1 = line.split()[0]
                    scnl = SCNL([sta1,chan1,net1])
                    #print(scnl.channel)
                    type1 = line.split()[3]
                    scnl.phase = type1
                    #print(scnl.phase)
                    time1 = UTCDateTime(line.split()[4]).datetime
                else:
                    sta1 = line.split()[0]
                    chan1 = line.split()[1]
                    #print(sta1,chan1)
                    #scnl.station = sta1
                    #net1 = line.split()[0]
                    try:
                        net1 = inventory.select(station=sta1)[0].code
                    except:
                        net1 = 'OK'
                        pass
                    scnl = SCNL([sta1,chan1,net1])
                    #print(scnl.channel)
                    type1 = line.split()[2]
                    scnl.phase = type1
                    #print(scnl.phase)
                    time1 = UTCDateTime(line.split()[3]).datetime
                t_create=datetime.utcnow()
                new_pick=tables1D.Pick(scnl,time1,'',10,0.1,t_create)
                #tables1D.Pick.phase=type1
                dbsession.add(new_pick) # Add pick i to the database
                dbsession.commit() #
            except:
                pass

# def gpd_pick_add(dbsession=None,fileinput=None):
#   filepath = fileinput
#   with open(filepath) as fp:
#     line = fp.readline()
#     cnt = 1
#     while line:
#       try:
#         print("Line {}: {}".format(cnt, line.strip()))
#         line = fp.readline()
#         cnt += 1
#         sta1 = line.split()[1]
#         chan1 = line.split()[2]
#         #print(sta1,chan1)
#         #scnl.station = sta1
#         net1 = line.split()[0]
#         scnl = SCNL([sta1,chan1,'OK'])
#         #print(scnl.channel)
#         type1 = line.split()[3]
#         scnl.phase = type1
#         time1 = UTCDateTime(line.split()[4]).datetime
#         t_create=datetime.utcnow()

#         new_pick=tables1D.Pick(scnl,time1,'',10,0.1,t_create)
#         dbsession.add(new_pick) # Add pick i to the database
#         dbsession.commit() #
#       except:
#         pass

def get_chan1(stationfile):
    if len(list(filter(None, stationfile.split('/')[-1].split('.'))))==5:
        comp = list(filter(None, stationfile.split('/')[-1].split('.')))[3][2]
    else:
        comp = list(filter(None, stationfile.split('/')[-1].split('.')))[2][2]
    return comp

def get_chan3(stationfile):
    if len(list(filter(None, stationfile.split('/')[-1].split('.'))))==5:
        comp3 = list(filter(None, stationfile.split('/')[-1].split('.')))[3][0:3]
    else:
        comp3 = list(filter(None, stationfile.split('/')[-1].split('.')))[2][0:3]
    return comp3

def detection_continuous(dirname=None, project_folder=None, project_code=None, local=True, machine=True, machine_picker=None, single_date=None, latitude=None, longitude=None, max_radius=None):
#    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
#    stopping = starting + 86430
    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
    stopping = starting + 86430
    dir1 = project_folder+'/'+dirname
    #print(single_date.strftime("%Y%m%d"))
    #print(dir1+'/1dassociator_'+project_code+'.db')
    if os.path.exists(dir1+'/1dassociator_'+project_code+'.db'):
        os.remove(dir1+'/1dassociator_'+project_code+'.db')
    db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
#    if os.path.exists(dir1+'/tt_ex_1D_'+project_code+'.db'):
#        os.remove(dir1+'/tt_ex_1D_'+project_code+'.db')
#    db_tt='sqlite:///'+dir1+'/tt_ex_1D_'+project_code+'.db' # Traveltime database44.448,longitude=-115.136
#    print(db_tt)
#    if local:
#        inventory = build_tt_tables_local_directory(dirname=dirname,project_folder=project_folder,channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5.)
#    else:
#        inventory = build_tt_tables(lat1=latitude,long1=longitude,maxrad=max_radius,starting=starting, stopping=stopping, channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5.)
    engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
    tables1D.Base.metadata.create_all(engine_assoc)
    Session=sessionmaker(bind=engine_assoc)
    session=Session()
    filelist = glob.glob(dir1+'/*mseed') or glob.glob(dir1+'/*SAC')
    stations = set()
    for file1 in filelist:
        station = file1.split('.')[1]
        net = file1.split('.')[0].split('/')[-1]
        netsta = net+'.'+station
        print(file1.split('.')[1])
        stations.add(netsta)
    #### create infile
    day_strings = []
    for stationin in stations:
        station3 = glob.glob(dir1+'/*'+stationin+'.*mseed') or glob.glob(dir1+'/*'+stationin+'.*SAC')
        station3a = [None,None,None]
        if len(station3)>3:
            #print(station3)
            ind1 = np.empty((len(station3),1))
            ind1[:] = np.nan
            for idxs, station1 in enumerate(station3):
                if get_chan3(station1) == 'HHZ':
                    ind1[idxs] = 2
                elif get_chan3(station1) == 'HHN' or get_chan3(station1) == 'HH1':
                    ind1[idxs] = 0
                elif get_chan3(station1) == 'HHE' or get_chan3(station1) == 'HH2':
                    ind1[idxs] = 1
                #print(idxs)
                #if ind1:
                #    station3a[ind1] = station1
            #ind2 = np.argwhere(~np.isnan(ind1))[:,0]
            for idxsa, ind2a in enumerate(ind1):
                if ~np.isnan(ind2a[0]):
                    #print(ind2a)
                    #print(station3a)
                    station3a[int(ind2a[0])] = station3[idxsa]
        else:
            for station1 in station3:
                if get_chan1(station1)  == 'Z':
                    ind1 = 2
                elif get_chan1(station1)  == 'N' or get_chan1(station1) == '1':
                    ind1 = 0
                elif get_chan1(station1)  == 'E' or get_chan1(station1) == '2':
                    ind1 = 1
                #print(ind1)
                station3a[ind1] = station1
        if any(elem is None for elem in station3a):
            continue
        day_strings.append((station3a[0]+' '+station3a[1]+' '+station3a[2]))

    day_string = "\n".join(day_strings)

    with open(dir1+'/dayfile.in', "w") as open_file:
        open_file.write(day_string)
    infile = dir1+'/dayfile.in'
    outfile = dir1+'/gpd_picks.out'
    #gpd_predict.py -V -P -I infile -O outflie
    #os.system("gpd_predict.py -V -P -I %s -O %s")%(infile, outfile)
    #gpd_predict(inputfile=infile,outputfile=outfile)
    fileinassociate = outfile

    if local:
        inv = Inventory()
        dir1a = glob.glob(project_folder+'/'+dirname+'/*xml')
        for file1 in dir1a:
            inv1a = read_inventory(file1)
            inv.networks.extend(inv1a)
    else:
        fdsnclient=Client()
        inv=fdsnclient.get_stations(starttime=starting,endtime=stopping,latitude=latitude,longitude=longitude,maxradius=max_radius,channel='*HZ',level='channel')

    if machine == True and machine_picker is None:
        machine_picker = 'GPD'
    if machine == True and machine_picker == 'GPD':
        fullpath1 = pathgpd+'/gpd_predict.py'
        os.system(fullpath1+" -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        gpd_pick_add(dbsession=session,fileinput=fileinassociate,inventory=inv)
    elif machine == True and machine_picker == 'EQTransformer':
        fullpath2 = pathEQT+'/mseed_predictor.py'
        os.system(fullpath2+" -I %s -O %s -F %s" % (infile, outfile, pathEQT))
        gpd_pick_add(dbsession=session,fileinput=fileinassociate,inventory=inv)
    else:
        picker = fbpicker.FBPicker(t_long = 5, freqmin = 1, mode = 'rms', t_ma = 20, nsigma = 7, t_up = 0.7, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
        fb_pick(dbengine=engine_assoc,picker=picker,fileinput=infile)

def association_continuous(dirname=None, project_folder=None, project_code=None, maxdist = None, maxkm=None, single_date=None, local=True, nsta_declare=4, delta_distance=1, latitude=None, longitude=None, max_radius=None):
    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
    stopping = starting + 86430

    dir1 = project_folder+'/'+dirname
    print(single_date.strftime("%Y%m%d"))
    #print(dir1+'/1dassociator_'+project_code+'.db')
#    if os.path.exists(dir1+'/1dassociator_'+project_code+'.db'):
#        os.remove(dir1+'/1dassociator_'+project_code+'.db')
#    db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
    if os.path.exists(dir1+'/tt_ex_1D_'+project_code+'.db'):
        os.remove(dir1+'/tt_ex_1D_'+project_code+'.db')
    db_tt='sqlite:///'+dir1+'/tt_ex_1D_'+project_code+'.db' # Traveltime database44.448,longitude=-115.136
    print(db_tt)
    if local:
        inventory = build_tt_tables_local_directory(dirname=dirname,project_folder=project_folder,channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5., delta_distance=delta_distance)
    else:
        inventory = build_tt_tables(lat1=latitude,long1=longitude,maxrad=max_radius,starting=starting, stopping=stopping, channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5., delta_distance=delta_distance)
    inventory.write(dir1+'/dailyinventory.xml',format="STATIONXML")
    if not os.path.exists(dir1+'/1dassociator_'+project_code+'.db'):
        db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
        engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
        tables1D.Base.metadata.create_all(engine_assoc)
        Session=sessionmaker(bind=engine_assoc)
        session=Session()
        gpd_pick_add(dbsession=session,fileinput=dir1+'/gpd_picks.out',inventory=inventory)

    db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
    assocXX=assoc1D.LocalAssociator(db_assoc, db_tt, max_km = maxkm, aggregation = 1, aggr_norm = 'L2', cutoff_outlier = 10, assoc_ot_uncert = 3, nsta_declare = nsta_declare, loc_uncert_thresh = 0.2)
    print("aggregate")
    t0=datetime.utcnow()
      # Identify candidate events (Pick Aggregation)
    assocXX.id_candidate_events()
    t1=datetime.utcnow()
    print('Took '+str(t1-t0))
    print("associate")
      # Associate events
    assocXX.associate_candidates()
    t2=datetime.utcnow()
    print('Took '+str(t2-t1))
      # Add singles stations to events
    try:
        assocXX.single_phase()
    except:
        pass






def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file, check_same_thread = False)
        return conn
    except Error as e:
        print(e)

    return None



def hypo_station(project_folder=None, project_code=None, catalog_year=None, year=None):
    hypo71_string_sta = ""
    station_strings = []
    f1 = open(project_folder+'/'+'sta','w')
    #f2 = open(project_folder+'/'+'station.dat', 'w')
    #for stas in temp:
    if catalog_year:
        files = sorted(glob.glob(project_folder+'/'+str(year)+'*/tt*'+project_code+'.db'))
    else:
        files = sorted(glob.glob(project_folder+'/*/tt*'+project_code+'.db')) or glob.glob(project_folder+'/tt*'+project_code+'.db')
    #print(files)
    stas1 = pd.DataFrame()
    for dfilesta in files:
        conn1 = create_connection(dfilesta)
        with conn1:
            cur1 = conn1.cursor()
            cur1.execute("SELECT * FROM stations")

            rows = cur1.fetchall()

            for row in rows:
                #print(row[0],row[1])
                #(row[0])
                df4 = pd.DataFrame()
                df4 = pd.DataFrame({'station': row[1], 'net':row[2],'latitude':row[4],'longitude':row[5],'elevation':row[6]}, index=[0])
                stas1=stas1.append(df4)
        conn1.close()
    stas1 = stas1.drop_duplicates()
    stas1 = stas1.reset_index(drop=True)
    print(stas1)
    for idx1 in stas1.index:
        stas = stas1.iloc[idx1]
        print(stas)
#                temp = stas1[stas1['station'].str.contains(sta_used)]
#                stas = temp.iloc[0]



        if len(stas['station'])>4:
            sta = stas['station'][1:]
        else:
            sta = stas['station']
        lon = stas['longitude']
        lon_deg = int(abs(lon))
        lon_min = (abs(lon) - abs(lon_deg)) * 60.
        lat = stas['latitude']
        lat_deg = int(abs(lat))
        lat_min = (abs(lat) - abs(lat_deg)) * 60.
        hem_NS = 'N'
        hem_EW = 'E'
        if lat < 0:
            hem_NS = 'S'
        if lon < 0:
            hem_EW = 'W'
        # hypo 71 format uses elevation in meters not kilometers
        ele = stas['elevation']
        hypo71_string_sta += fmt % (sta, lat_deg, lat_min, hem_NS,
                                lon_deg, lon_min, hem_EW, ele)
        station_strings.append("%s %.6f %.6f %i" % (sta, stas['latitude'], stas['longitude'], stas['elevation']))


                #print(hypo71_string_sta)
    station_string = "\n".join(station_strings)
    with open(project_folder+'/'+'station.dat', "w") as open_file:
        open_file.write(station_string)
    f1.write(str(hypo71_string_sta))
    f1.close()





def select_all_associated(conn, f0):
    """
    Query all rows in the associated table
    :param conn: the Connection object
    :return:
    """
    cur1 = conn.cursor()
    cur1.execute("SELECT * FROM associated")
    stalistall = set()
    rows = cur1.fetchall()
    dfs1 = pd.DataFrame()
    cat1 = Catalog()
    for rownum, row in enumerate(rows):
        #print(row)
        #(row[0])
        df4 = pd.DataFrame()
        df4 = pd.DataFrame({'Time': row[1], 'Lat':row[3],'Long':row[4]}, index=[0])
        dfs1=dfs1.append(df4)
        origin = Origin()
        origin.latitude = row[3]
        origin.longitude = row[4]
        origin.depth = 5000
        origin.time = row[1]
        origin.arrivals = []
        strday = row[1][0:4]+row[1][5:7]+row[1][8:10]
        cur1.execute('SELECT * FROM picks_modified WHERE assoc_id IN (?)',[int(row[0])])
        picks1a = sorted(cur1.fetchall())
        stas = []
        event = Event()
        evid = 'smi:local/Event/'+strday+str(rownum+1).zfill(3)
        orid = 'smi:local/Origin/'+strday+str(rownum+1).zfill(3)
        event.resource_id = ResourceIdentifier(id=evid)
        origin.resource_id = ResourceIdentifier(id=orid)
        event.resource_id = ResourceIdentifier(id='smi:local/Event/'+strday+str(rownum).zfill(3))
        origin.resource_id = ResourceIdentifier(id='smi:local/Origin/'+strday+str(rownum).zfill(3)+'_1')
        for pick1 in picks1a:

            #print(pick1)
            stream_id = WaveformStreamID(network_code=pick1[3], station_code=pick1[1], location_code="", channel_code=pick1[2])
            p = Pick()
            p.time = pick1[5]
            p.phase_hint = pick1[6]
            p.waveform_id = stream_id
            p.evaluation_mode = 'automatic'
            pres_id = 'smi:local/Pick/'+strday+'/'+str(pick1[0])
            #res_id = ResourceIdentifier(prefix='Pick')
            #res_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
            p.resource_id = ResourceIdentifier(id=pres_id)
            #print(p)

            a = Arrival()
            #a.time = pick1[5]
            a.phase = pick1[6]
            a.pick_id = p.resource_id
            ares_id = 'smi:local/Arrival/'+strday+'/'+str(pick1[0])
            #res_id = ResourceIdentifier(prefix='Pick')
            #res_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
            a.resource_id = ResourceIdentifier(id=ares_id)
            a.time_weight = 1.0
            #print(a)

            #origin.picks.append(p)
            sta1 = pick1[1]
            stas.append(sta1)
            stalistall.add(sta1)
            origin.arrivals.append(a)
            event.picks.append(p)
        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id
        cat1.append(event)
            #print(stalistall)
        stalist = list(set(stas))
        for states in stalist:
            hypo71_string = ""
            numP = -9
            numS = -9
            #print(states)
            for num, line in enumerate(picks1a):
                if states in line and 'P' in line:
                    numP = num
                if states in line and 'S' in line:
                    numS = num
#            if numP > -1 and numS < -1:
#                #print('just P'+str(numP))
#            if numP < -1 and numS > -1:
#                #print('just S')
#            if numP > -1 and numS > -1:
#                print('both'+str(numP)+' '+str(numS))
            if len(states)>4:
                sta = states[1:]
            else:
                sta = states
            if numP > -1:
                pick = picks1a[numP]



                t = UTCDateTime(pick[5])
                hundredth = int(round(t.microsecond / 1e4))
                if hundredth == 100:
                    t_p = t + 1
                    hundredth = 0
                else:
                    t_p = t
                date = t_p.strftime("%y%m%d%H%M%S") + ".%02d" % hundredth
                onset = 'I'
                polarity = '?'
                weight = 1
                #print(sta,onset,polarity,weight,date)
                hypo71_string += fmtP % (sta, onset, polarity, weight, date)
                #f0.write(str(hypo71_string))

                #print(hypo71_string)
                if numP > -1 and numS > -1:
                    pick = picks1a[numS]
                    #t = UTCDateTime(pick[5])
                    t2 = UTCDateTime(pick[5])
                    # if the S time's absolute minute is higher than that of the
                    # P pick, we have to add 60 to the S second count for the
                    # hypo 2000 output file
                    # +60 %60 is necessary if t.min = 57, t2.min = 2 e.g.
                    mindiff = (t2.minute - t.minute + 60) % 60
                    abs_sec = t2.second + (mindiff * 60)
                    hundredth = int(round(t2.microsecond / 1e4))
                    if hundredth == 100:
                        abs_sec += 1
                        hundredth = 0
                    date2 = "%s.%02d" % (abs_sec, hundredth)
                    hundredth = int(round(t.microsecond / 1e4))
                    if hundredth == 100:
                        t_p = t + 1
                        hundredth = 0
                    else:
                        t_p = t
                    date = t_p.strftime("%y%m%d%H%M%S") + ".%02d" % hundredth
                    onset = 'I'
                    polarity = '?'
                    weight = 1
                    #print(sta,onset,polarity,weight,date)
                    hypo71_string += fmtS % (date2, onset, polarity,weight)

                else:
                    hypo71_string += "\n"
            f0.write(str(hypo71_string))
                #print(str(hypo71_string))
                #os.system(fullpath1+" -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        f0.write("\n")
            #f1.write("\n")



    return dfs1, stalistall, cat1, f0

def combine_associated(project_folder=None, project_code=None, catalog_year=False, year=None, hypoflag=False, eventmode=False):
    #files = sorted(glob.glob('/data/tx/ContWaveform/*/1dass*'++'.db'))
    #files = [f for f in os.listdir(dirdata) if os.path.isfile(os.path.join(dirdata, f))]
    #dir1 = project_folder+'/'+dirname

    hypo_station(project_folder, project_code)
    files = sorted(glob.glob(project_folder+'/*/1dass*'+project_code+'.db'))
    if catalog_year:
        files = sorted(glob.glob(project_folder+'/'+str(year)+'*/1dass*'+project_code+'.db'))
    if eventmode:
        files = sorted(glob.glob(project_folder+'/1dass*'+project_code+'.db'))
    f0 = open(project_folder+'/pha_'+project_code,'w')
    dfs2 = pd.DataFrame()
    stalistall1 = []
    cat = Catalog()
    for dfile in files:
        # create a database connection
        print(dfile)
        conn = create_connection(dfile)

        with conn:
            #print("1. Query task by priority:")
            #select_task_by_priority(conn,1)

            #print('Day '+dfile[-6:-3])
            #try:

            dfs1,stalistall,cat1,f0 = select_all_associated(conn, f0)
            cat.extend(cat1)
            for stas1 in stalistall:
                if stas1 not in stalistall1:
                    stalistall1.append(stas1)
            dfs2 = dfs2.append(dfs1)
#            except:
#                pass
        conn.close()
    f0.close()
    if catalog_year:
        cat.write(project_folder+'/'+project_code+'_'+str(year)+'_cat.xml',format="QUAKEML")
    else:
        if not eventmode:
            cat.write(project_folder+'/'+project_code+'_cat.xml',format="QUAKEML")
    return cat, dfs2





#
#      index0=int(round((picks[i]-self.tr.stats.starttime)/dt,0))
#      index=index0
#
#      # roll forward index+=1
#      while True:
#        if index>=self.stats.npts-1-2:
#          break
#        elif (self.tr[index+1]-self.tr[index])*(self.tr[index+2]-self.tr[index+1])>0:
#          index+=1
#        else:
#          break
#
#      # notice index+1, rolling stop one point before extreme, compare with std to avoid very small
#      if self.tr[index+1] - self.tr[index0] > 0 and abs(self.tr[index+1] - self.tr[index0]) > self.picker.pol_coeff * np.std(self.tr[index0 - self.picker.pol_len: index0]):
#        polarity='C'
#      elif self.tr[index+1] - self.tr[index0] < 0 and abs(self.tr[index+1] - self.tr[index0]) > self.picker.pol_coeff * np.std(self.tr[index0 - self.picker.pol_len: index0]):
#        polarity='D'
#      else:
#        polarity=''
def polarity(tr,pickP=None):
    dt=tr.stats.delta
    #t = np.arange(0, tr.stats.npts/tr.stats.sampling_rate, dt)
    index0=int(round((pickP-tr.stats.starttime)/dt,0))
    index=index0
    pol_coeff = 5
    pol_len = 5
    polarity = 'undecidable'
    while True:
        if index>=tr.stats.npts-1-2:
            break
        elif (tr[index+1]-tr[index])*(tr[index+2]-tr[index+1])>0:
            index+=1
        else:
            break
        if tr[index+1] - tr[index0] > 0 and abs(tr[index+1] - tr[index0]) > pol_coeff * np.std(tr[index0 - pol_len: index0]):
            polarity='positive'
        elif tr[index+1] - tr[index0] < 0 and abs(tr[index+1] - tr[index0]) > pol_coeff * np.std(tr[index0 - pol_len: index0]):
            polarity='negative'
        else:
            polarity='undecidable'
    return polarity


def magnitude_quakeml(cat=None, project_folder=None,plot_event=False,eventmode=False):

    paz_wa = {'sensitivity': 2080, 'zeros': [0j], 'gain': 1,'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

    print('Computing magnitudes')
    client = Client()

    for event in cat:
        origin = event.origins[0]
        print(origin)
        event_lat = origin.latitude
        event_lon = origin.longitude
        strday = str(origin.time.year).zfill(2)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)
        if eventmode:
            strday = str(project_folder.split('/')[-1])
        #    strday = str(origin.time.year).zfill(2)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)
        print(strday)
        strdaytime = strday+str(origin.time.hour).zfill(2)+str(origin.time.minute).zfill(2)[0]
        mags = []
        mags_iaspei = []

        st2 = Stream()
        for idx1, pick in enumerate(event.picks):
            if pick.phase_hint == 'S':
                ### make Amplitude
                try:
                    try:
                        st3 = read(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.*.'+pick.waveform_id.channel_code[0:2]+'*mseed',debug_headers=True)
                        #print(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed')
                    except:
                        try:
                            st3 = read(project_folder+'/'+strday+'*/*.'+pick.waveform_id.station_code+'*SAC',debug_headers=True)
                        except:
                            #st3 = read(project_folder+'/scratch/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed',debug_headers=True)
                            try:
                                st3 = read(project_folder+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed',debug_headers=True)
                            except:
                                st3 = read(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed',debug_headers=True)
                                pass
    #                    try:
    #                        st3 = read(project_folder+'/'+strday+'*/*.'+pick.waveform_id.station_code+'*SAC',debug_headers=True)
    #                    except:
    #                        st3 = read(project_folder+'/'+strdaytime+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed',debug_headers=True)
                        pass

        #            pazs = glob.glob('/data/tx/ContWaveform/'+strday+'/SACPZ.'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*[EN12]')
                    #st = read(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*[EN12]*.SAC',debug_headers=True)
                    try:
                        st3.merge(fill_value='interpolate')
                        print(st3)
                        for tr in st3:
                            if isinstance(tr.data, np.ma.masked_array):
                                tr.data = tr.data.filled()
                        st = st3.select(channel='[EHB]H[EN12]')
                        for tr in st3:
                            inventory_local = glob.glob(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.xml')
                            if len(inventory_local)>0:
                                inv = read_inventory(inventory_local[0])
                            else:
                                try:
                                    inv0 = read_inventory(project_folder+'/'+strday+'*/dailyinventory.xml')
                                    inv = inv0.select(network=pick.waveform_id.network_code, station=pick.waveform_id.station_code, time=origin.time)
                                except:
                                    print('Getting response from DMC')
                                    starttime = UTCDateTime(origin.time-10)
                                    endtime = UTCDateTime(origin.time+10)
                                    inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel=tr.stats.channel,level="response")
                                    pass
                                    #                    paz = [x for x in pazs if tr.stats.channel in x]
        #                    attach_paz(tr, paz[0])
                            #inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel=tr.stats.channel,level="response")
                            tr.stats.network = inv[0].code
                            tr.stats.location = inv[0][0][0].location_code
                            pre_filt = (0.05, 0.06, 30.0, 35.0)
                            tr.trim(pick.time-30, pick.time+120)


                            #tr.demean()
                            tr.detrend()
                            tr.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt, zero_mean=True)
                            #tr.data = seis_sim(tr.data, tr.stats.sampling_rate,paz_remove=None, paz_simulate=paz_wa, water_level=10)
                            tr.simulate(paz_simulate=paz_wa, water_level=10)


                            #tr = tr.filter('bandpass', freqmin=fminbp, freqmax=fmaxbp, zerophase=True)
                        #st.trim(pick.time-5,pick.time+10)
                        tr1 = st3.select(channel='[EHB]HZ')[0]

                        sta_lat = inv[0][0].latitude
                        sta_lon = inv[0][0].longitude
                        epi_dist, az, baz = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
                        epi_dist = epi_dist / 1000
                        tr1.stats.distance = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)[0]
                        tr1.trim(pick.time-20,pick.time+60)
                        st2 += tr1
                        st.trim(pick.time-1,pick.time+5)
                        ampls = (max(abs(st[0].data)), max(abs(st[1].data)))
                        for idx2,ampl in enumerate(ampls):

                            amp = Amplitude()
                            res_id = 'smi:local/Amplitude/'+strday+'/'+str(10*idx2+idx1)
                            #res_id = ResourceIdentifier(prefix='Amplitude')
                            #res_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
                            amp.resource_id = ResourceIdentifier(id=res_id)
                            amp.pick_id = pick.resource_id
                            amp.waveform_id = pick.waveform_id
                            amp.type = 'ML'
                            amp.generic_amplitude = ampl
                            amp.evaluation_mode = 'automatic'

                            if epi_dist < 60:
                                a = 0.018
                                b = 2.17
                            else:
                                a = 0.0038
                                b = 3.02
                            ml = np.log10(ampl * 1000) + a * epi_dist + b

                            ml_iaspei = np.log10(ampl*1e6)+1.11*np.log10(epi_dist) + 0.00189*epi_dist - 2.09
                            print(ml, ml_iaspei)

                            if epi_dist < 160:
                                mags.append(ml)
                                mags_iaspei.append(ml_iaspei)
                                #### make StationMagnitude
                                stamag = StationMagnitude()
                                res_id = 'smi:local/StationMagnitude/'+strday+'/'+str(10*idx2+idx1)
                                #res_id = ResourceIdentifier(prefix='StationMagnitude')
                                #res_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
                                stamag.resource_id = ResourceIdentifier(id=res_id)
                                stamag.origin_id = origin.resource_id
                                stamag.waveform_id = pick.waveform_id
                                stamag.mag = ml
                                stamag.station_magnitude_type = 'ML'
                                stamag.amplitude_id = amp.resource_id
                            ## add them to the event
                            event.station_magnitudes.append(stamag)
                            event.amplitudes.append(amp)
                    except:
                        print('Something went wrong here')
                        pass
                except:
                    pass

        for pick in event.picks:
            if pick.phase_hint == 'P':
                tr = st2.select(station=pick.waveform_id.station_code)
                try:
                    tr = tr[0]
                    pol = polarity(tr,pick.time)
                    pick.polarity = pol
                    print(pol)
                except:
                    pass





        netmag = np.median(mags_iaspei)
        try:
            m = Magnitude()
            m.mag = netmag
            m.mag_errors = {"uncertainty": np.std(mags_iaspei)}
            m.magnitude_type = 'ML'
            m.origin_id = origin.resource_id
            meth_id = 'smi:local/median'
            m.method_id = meth_id
            m.station_count = len(mags_iaspei)
            m_id = 'smi:local/Magnitude/'+strday+'/'+str(idx1)
            #m_id = ResourceIdentifier(prefix='StationMagnitude')
            #m_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
            m.resource_id = m_id
            event.magnitudes.append(m)



            event.preferred_magnitude_id = m.resource_id




            if plot_event:
                dir1a = glob.glob(project_folder+'/'+strday+'*')
                filename = dir1a[0]+'/'+strdaytime
                fig = plt.figure()
                st2.filter('highpass', freq=.1, zerophase=True)
                st2.plot(type='section', scale=2,plot_dx=100e3, recordlength=50,
                    time_down=True, linewidth=.25, grid_linewidth=.25, show=False,
                    outfile=filename,fig=fig)
                plt.close()
        except:
            print('Magnitude failed')
            pass
    if not eventmode:
        cat.write(project_folder+'/cat.xml',format="QUAKEML")
    return cat



def single_event_xml(catalog=None,project_folder=None, format="QUAKEML"):
    xmlspath = project_folder+'/'+format.lower()
    if not os.path.exists(xmlspath):
        os.makedirs(xmlspath)
    for ev in catalog:
        filename = str(ev.resource_id).split('/')[-1] + ".xml"
        ev.write(xmlspath+'/'+filename, format=format)


def cut_event_waveforms():
    for event in cat:
        origin = event.origins[0]
        print(origin)
        event_lat = origin.latitude
        event_lon = origin.longitude
        strday = str(origin.time.year).zfill(2)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)
        if eventmode:
            strday = str(project_folder.split('/')[-1])
        #    strday = str(origin.time.year).zfill(2)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)
        print(strday)
        strdaytime = strday+str(origin.time.hour).zfill(2)+str(origin.time.minute).zfill(2)[0]


#        st2 = Stream()
#
#        for idx1, pick in enumerate(event.picks):
#            if pick.phase_hint == 'S':
#                try:
#                    try:
#                        st3 = read(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.*.'+pick.waveform_id.channel_code[0:2]+'*mseed',debug_headers=True)
#                        #print(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed')
#                    except:
#                        try:
#                            st3 = read(project_folder+'/'+strday+'*/*.'+pick.waveform_id.station_code+'*SAC',debug_headers=True)
#                        except:
#                            try:
#                                st3 = read(project_folder+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed',debug_headers=True)
#                            except:
#                                st3 = read(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*mseed',debug_headers=True)
#                                pass
#                        pass
#








def detection_association_event(project_folder=None, project_code=None, maxdist = None, maxkm=None, local=True, machine=True, approxorigintime=None, downloadwaveforms=True, delta_distance=1, latitude=None, longitude=None, max_radius=None):
    approxotime = UTCDateTime(approxorigintime)
    dirname = str(approxotime.year)+str(approxotime.month).zfill(2)+str(approxotime.day).zfill(2)+str(approxotime.hour).zfill(2)+str(approxotime.minute).zfill(2)+str(approxotime.second).zfill(2)
    #starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0') -
    starting = approxotime - 60
    stopping = approxotime + 120
    dir1 = project_folder+'/'+dirname
    print(dir1)
    if downloadwaveforms:
        download_mseed_event_radial(dirname=dirname, project_folder=project_folder, starting=starting, stopping = stopping, lat1=latitude, lon1=longitude, maxrad=max_radius)
    #print(single_date.strftime("%Y%m%d"))
    #print(dir1+'/1dassociator_'+project_code+'.db')
    if os.path.exists(dir1+'/1dassociator_'+project_code+'.db'):
        os.remove(dir1+'/1dassociator_'+project_code+'.db')
    db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
    engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
    tables1D.Base.metadata.create_all(engine_assoc)
    Session=sessionmaker(bind=engine_assoc)
    session=Session()
    filelist = glob.glob(dir1+'/*mseed') or glob.glob(dir1+'/*SAC')
    stations = set()
    for file1 in filelist:
        station = file1.split('.')[1]
        net = file1.split('.')[0].split('/')[-1]
        netsta = net+'.'+station
        print(file1.split('.')[1])
        stations.add(netsta)
    #### create infile
    day_strings = []
    for stationin in stations:
        station3 = glob.glob(dir1+'/*'+stationin+'.*mseed') or glob.glob(dir1+'/*'+stationin+'.*SAC')
        station3a = [None,None,None]
        if len(station3)>3:
            #print(station3)
            ind1 = np.empty((len(station3),1))
            ind1[:] = np.nan
            for idxs, station1 in enumerate(station3):
                if get_chan3(station1) == 'HHZ':
                    ind1[idxs] = 2
                elif get_chan3(station1) == 'HHN' or get_chan3(station1) == 'HH1':
                    ind1[idxs] = 0
                elif get_chan3(station1) == 'HHE' or get_chan3(station1) == 'HH2':
                    ind1[idxs] = 1
                #print(idxs)
                #if ind1:
                #    station3a[ind1] = station1
            #ind2 = np.argwhere(~np.isnan(ind1))[:,0]
            for idxsa, ind2a in enumerate(ind1):
                if ~np.isnan(ind2a[0]):
                    #print(ind2a)
                    #print(station3a)
                    station3a[int(ind2a[0])] = station3[idxsa]
        else:
            for station1 in station3:
                if get_chan1(station1)  == 'Z':
                    ind1 = 2
                elif get_chan1(station1)  == 'N' or get_chan1(station1) == '1':
                    ind1 = 0
                elif get_chan1(station1)  == 'E' or get_chan1(station1) == '2':
                    ind1 = 1
                #print(ind1)
                station3a[ind1] = station1
        if any(elem is None for elem in station3a):
            continue
        day_strings.append((station3a[0]+' '+station3a[1]+' '+station3a[2]))

    day_string = "\n".join(day_strings)

    with open(dir1+'/dayfile.in', "w") as open_file:
        open_file.write(day_string)
    infile = dir1+'/dayfile.in'
    outfile = dir1+'/gpd_picks.out'
    fileinassociate = outfile

    if local:
        inv = Inventory()
        dir1a = glob.glob(project_folder+'/'+dirname+'/*xml')
        for file1 in dir1a:
            inv1a = read_inventory(file1)
            inv.networks.extend(inv1a)
    else:
        fdsnclient=Client()
        inv=fdsnclient.get_stations(starttime=starting,endtime=stopping,latitude=latitude,longitude=longitude,maxradius=max_radius,channel='*HZ',level='channel')
    if machine:
        fullpath1 = pathgpd+'/gpd_predict.py'
        os.system(fullpath1+" -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        gpd_pick_add(dbsession=session,fileinput=fileinassociate,inventory=inv)
    else:
        picker = fbpicker.FBPicker(t_long = 5, freqmin = 1, mode = 'rms', t_ma = 20, nsigma = 7, t_up = 0.7, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
        fb_pick(dbengine=engine_assoc,picker=picker,fileinput=infile)    # starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
    # stopping = starting + 86430
    session.close()


#    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
#    stopping = starting + 86430

    dir1 = project_folder+'/'+dirname

    if os.path.exists(dir1+'/tt_ex_1D_'+project_code+'.db'):
        os.remove(dir1+'/tt_ex_1D_'+project_code+'.db')
    db_tt='sqlite:///'+dir1+'/tt_ex_1D_'+project_code+'.db' # Traveltime database44.448,longitude=-115.136
    print(db_tt)
    if local:
        inventory = build_tt_tables_local_directory(dirname=dirname,project_folder=project_folder,channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5., delta_distance=delta_distance)
    else:
        inventory = build_tt_tables(lat1=latitude,long1=longitude,maxrad=max_radius,starting=starting, stopping=stopping, channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5., delta_distance=delta_distance)
    inventory.write(dir1+'/dailyinventory.xml',format="STATIONXML")
    if not os.path.exists(dir1+'/1dassociator_'+project_code+'.db'):
        db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
        engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
        tables1D.Base.metadata.create_all(engine_assoc)
        Session=sessionmaker(bind=engine_assoc)
        session=Session()
        gpd_pick_add(dbsession=session,fileinput=dir1+'/gpd_picks.out',inventory=inventory)
        session.close()

    db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
    assocXX=assoc1D.LocalAssociator(db_assoc, db_tt, max_km = maxkm, aggregation = 1, aggr_norm = 'L2', cutoff_outlier = 10, assoc_ot_uncert = 3, nsta_declare = 4, loc_uncert_thresh = 0.2)
    print("aggregate")
    t0=datetime.utcnow()
      # Identify candidate events (Pick Aggregation)
    assocXX.id_candidate_events()
    t1=datetime.utcnow()
    print('Took '+str(t1-t0))
    print("associate")
      # Associate events
    assocXX.associate_candidates()
    t2=datetime.utcnow()
    print('Took '+str(t2-t1))
      # Add singles stations to events
    try:
        assocXX.single_phase()
    except:
        pass
    engine_assoc.dispose()
    cat, dfs = combine_associated(project_folder=dir1, project_code=project_code, eventmode=True)
    cat = magnitude_quakeml(cat=cat, project_folder=dir1,plot_event=False, eventmode=True)
    #cat.write('catalog_idaho.xml',format='QUAKEML')
    #single_event_xml(cat,dir1,"QUAKEML")
    for idx1, ev in enumerate(cat):
        filename = dirname+'_'+str(idx1) + ".xml"
        ev.write(project_folder+'/'+filename, format='QUAKEML')


def simple_cat_df(cat=None, uncertainty=False):
    times = []
    lats = []
    lons = []
    deps = []
    magnitudes = []
    magnitudestype = []
    resourceid = []
    rms = []
    az_gap = []
    hor_err = []
    vert_err = []
    n_arr = []

    for event in cat:
        if len(event.origins) != 0:
            origin1 = event.preferred_origin() or event.origins[0]
            times.append(origin1.time.datetime)
            lats.append(origin1.latitude)
            lons.append(origin1.longitude)
            deps.append(origin1.depth)
            if event.preferred_magnitude() is not None:
                magnitudes.append(event.preferred_magnitude().mag)
                magnitudestype.append(event.preferred_magnitude().magnitude_type)
            else:
                if len(event.magnitudes)>0:
                    magnitudes.append(event.magnitudes[0].mag)
                    magnitudestype.append(event.magnitudes[0].magnitude_type)
                else:
                    magnitudes.append(np.nan)
                    magnitudestype.append(np.nan)
            resourceid.append(event.resource_id)
        if uncertainty is True:
            rms.append(origin1.quality.standard_error)
            az_gap.append(origin1.quality.azimuthal_gap)
            hor_err.append(origin1.origin_uncertainty.horizontal_uncertainty)
            vert_err.append(origin1.depth_errors.uncertainty)
            n_arr.append(len(origin1.arrivals))

    if uncertainty is True:
        catdf1 = pd.DataFrame({'origintime':times,'latitude':lats,'longitude':lons, 'depth':deps,'magnitude':magnitudes,'type':magnitudestype,'horizontal_error':hor_err,'vertical_error':vert_err,'num_arrivals':n_arr,'rms':rms, 'azimuthal_gap':az_gap ,'id':resourceid})
        catdf1 = catdf1.sort_values(by='origintime',ascending=True)
        catdf1 = catdf1.reset_index(drop=True)
    else:
        catdf1 = pd.DataFrame({'origintime':times,'latitude':lats,'longitude':lons, 'depth':deps,'magnitude':magnitudes,'type':magnitudestype,'id':resourceid})
        catdf1 = catdf1.sort_values(by='origintime',ascending=True)
        catdf1 = catdf1.reset_index(drop=True)
    return catdf1

def catdf_narrowbounds(catdf=None,lat_a=None,lat_b=None,lon_a=None,lon_b=None):
    catdf = catdf[(catdf['latitude']>lat_a) & (catdf['latitude']<lat_b) & (catdf['longitude']>lon_a) & (catdf['longitude']<lon_b)]
    return catdf

#
#
def quakeml_to_hypodd(cat=None, download_station_metadata=True, project_folder=None, project_code=None):
    #catdf = simple_cat_df(cat)
    phase_dat_file = project_folder+'/'+project_code+'.pha'
    #for idx0, t0 in enumerate(catdf.index):
    #stations = []
    stations = set()

    event_strings = []
    for idx1, event in enumerate(cat):
        #evo = event.preferred_origin().time
        evid = idx1
        #otime = UTCDateTime(evo)
        origin = event.preferred_origin() or event.origins[0]
        try:
            mag1 = event.preferred_magnitude() or event.magnitudes[0]
            magpref = mag1.mag
        except:
            magpref = 0
            continue
        #print(magpref)


        depth_error = 0
        longitude_error = 0
        latitude_error = 0


        string = "# {year} {month} {day} {hour} {minute} " + \
            "{second:.6f} {latitude:.6f} {longitude:.6f} " + \
            "{depth:.4f} {magnitude:.6f} {horizontal_error:.6f} " + \
            "{depth_error:.6f} {travel_time_residual:.6f} {event_id}"
        event_string = string.format(year=origin.time.year,
                             month=origin.time.month,
                             day=origin.time.day,
                             hour=origin.time.hour,
                             minute=origin.time.minute,
                             # Seconds + microseconds
                             second=float(origin.time.second) +
                             (origin.time.microsecond / 1e6),
                             latitude=origin.latitude,
                             longitude=origin.longitude,
                             # QuakeML depth is in meters. Convert to km.
                             depth=origin.depth / 1000.0,
                             magnitude=magpref,
                             horizontal_error=max(
                                 [latitude_error,
                                  longitude_error]),
                             depth_error= depth_error,
                             travel_time_residual=0,
                             event_id=idx1)
        event_strings.append(event_string)
        for _i, arrv in enumerate(origin.arrivals):
            pick = arrv.pick_id.get_referred_object()
            stations.add(pick.waveform_id.network_code+'.'+pick.waveform_id.station_code)
            #print(pick.polarity)
            # Only P and S phases currently supported by HypoDD.
            if pick.phase_hint is not None:
                if pick.phase_hint.upper() != "P" and pick.phase_hint.upper() != "S":
                    continue
                string = "{station_id} {travel_time:.6f} {weight:.2f} {phase}"
                travel_time = pick.time - origin.time
                # Simple check to assure no negative travel times are used.
                if travel_time < 0:
                    msg = "Negative absolute travel time. " + \
                        "{phase} phase pick for event {event_id} at " + \
                        "station {station_id} will not be used."
                    msg = msg.format(
                        phase=pick.phase_hint,
                        event_id=evid,
                        station_id=pick.waveform_id.station_code)
                    print(msg)
                    continue
                phase_weighting = lambda sta_id, ph_type, time, uncertainty: 1.0
                weight = phase_weighting(pick.waveform_id.station_code, pick.phase_hint.upper(),
                                         pick.time,
                                         arrv.time_residual)
                pick_string = string.format(
                    station_id=pick.waveform_id.station_code,
                    travel_time=travel_time,
                    weight=weight,
                    phase=pick.phase_hint.upper())
                event_strings.append(pick_string)


        event_string = "\n".join(event_strings)
#        except:
#            print('Some error occurred????', evo)
#            pass
        # Write the phase.dat file.
        with open(phase_dat_file, "w") as open_file:
            open_file.write(event_string)

    if download_station_metadata:
        station_dat_file = project_folder+'/'+project_code+'station.dat'
        print('Downloading station location metadata')
        client = Client()
        station_strings = []
        for sta in stations:
            print(sta)
            net, sta1 = sta.split('.')
            #sta1 = sta

            try:
                inva = client.get_stations(starttime=cat[0].preferred_origin().time, endtime=cat[-1].preferred_origin().time, network='*', station=sta1, level='station')
            except:
                inva = None
                pass
            if inva is not None:
                dists = []
                for idxn, net in enumerate(inva):
                    dists.append(gps2dist_azimuth(net[0].latitude, net[0].longitude, cat[-1].preferred_origin().latitude, cat[-1].preferred_origin().longitude)[0])

                station_lat = inva[np.where(dists==np.min(dists))[0][0]][0].latitude
                station_lon = inva[np.where(dists==np.min(dists))[0][0]][0].longitude
                station_elev = inva[np.where(dists==np.min(dists))[0][0]][0].elevation
                station_strings.append("%s %.6f %.6f %i" % (sta1,  station_lat,  station_lon,  station_elev))
            #inva1[0][0].latitude
        station_string = "\n".join(set(station_strings))
        with open(station_dat_file, "w") as open_file:
            open_file.write(station_string)


#    station_dat_file = project_folder+'/'+'station.dat'
#
#    #station_strings = []
#    #for key, value in self.stations.iteritems():
#    #    station_strings.append("%s %.6f %.6f %i" % (key, value["latitude"],
#    #        value["longitude"], value["elevation"]))
#    #station_string = "\n".join(station_strings)
#    #with open(station_dat_file, "w") as open_file:
#    #    open_file.write(station_string)
#    #self.log("Created station.dat input file.")
#
#
#    starttime = UTCDateTime("2010-01-01T00:00:00.000")
#    endtime = UTCDateTime("2022-01-01T00:00:00.000")
#    #line = [(-98.15, 35.88),(-98.05, 35.8)] # Cushing area
#
#    client = Client('IRIS')
#    inva = client.get_stations(starttime=starttime, endtime=endtime,network="*", loc="*", channel="*",minlatitude=minlat, maxlatitude=maxlat,minlongitude=minlon, maxlongitude=maxlon,level="station")
#    station_strings = []
#    for sta in stations:
#        print(sta)
#        inva1 = inva.select(station=sta)
#        if len(inva1.networks) > 0:
#            station_strings.append("%s %.6f %.6f %i" % (sta, inva1[0][0].latitude, inva1[0][0].longitude, inva1[0][0].elevation))
#        #inva1[0][0].latitude
#    station_string = "\n".join(station_strings)
#    with open(station_dat_file, "w") as open_file:
#        open_file.write(station_string)




def plot_hypodd_catalog(file=None):
    catdfr = pd.read_csv(file,delimiter=r"\s+")
    catdfr = catdfr.dropna()
    catdfr = catdfr.reset_index(drop=True)
    #rutc = np.zeros((len(catdfr.index),1))
    rutc = []
    for i in range(0,len(catdfr.index)):
        rutc.append(UTCDateTime(int(catdfr.iloc[i,10]),int(catdfr.iloc[i,11]),int(catdfr.iloc[i,12]),int(catdfr.iloc[i,13]),int(catdfr.iloc[i,14]),catdfr.iloc[i,15]))

    catdfr['rutc'] = rutc
    catdfr.sort_values(by=['rutc'], inplace=True)
    catdfr = catdfr.reset_index(drop=True)





    from mpl_toolkits.basemap import Basemap
    # 1. Draw the map background
    #fig = plt.figure(figsize=(8, 8))
    lat0 = np.median(catdfr.iloc[:,1].values)
    lon0 = np.median(catdfr.iloc[:,2].values)
    m = Basemap(projection='lcc', resolution='h',
                lat_0=lat0, lon_0=lon0,
                width=1E6, height=.6E6)
    #m.shadedrelief()
    #m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    #m.drawcounties(color='gray')
    m.drawstates(color='gray')

    # 2. scatter city data, with color reflecting population
    # and size reflecting area
    m.scatter(catdfr.iloc[:,2].values,catdfr.iloc[:,1].values,s=catdfr.iloc[:,16].values**3*8,c=catdfr.index,marker='o',alpha=0.5,latlon=True)

    #m.scatter(catdfo.iloc[:,2].values,catdfo.iloc[:,1].values,s=catdfo.iloc[:,16].values**3*10,c=catdfo.index,marker='o',alpha=0.5,latlon=True)



    cbar = plt.colorbar()
    N_TICKS=8
    indexes = [catdfr['rutc'].iloc[i].strftime('%Y-%m-%d') for i in np.linspace(0,catdfr.shape[0]-1,N_TICKS).astype(int)]

    #indexes = [catdfr.index[i].strftime('%Y-%m-%d') for i in np.linspace(0,catdfr.shape[0]-1,N_TICKS).astype(int)]
    cbar.ax.set_yticklabels(indexes)
    plt.savefig('hypoDDmap.png')
    plt.show()







def locate_hyp2000(cat=None, project_folder=None, vel_model=None):
    for idx1, event in enumerate(cat):
        origin = event.preferred_origin() or event.origins[0]
        stas = []
        picks1a = []
        for _i, arrv in enumerate(origin.arrivals):
            pick = arrv.pick_id.get_referred_object()
            stas.append(pick.waveform_id.station_code)
            picks1a.append(pick.waveform_id.station_code + ' '+pick.phase_hint+' '+str(pick.time))
            #print(stalistall)
        stalist = list(set(stas))
        hypo71_string = ""
        for states in stalist:

            numP = -9
            numS = -9
            #print(states)
            for num, line in enumerate(picks1a):
                if states in line and 'P' in line:
                    numP = num
                if states in line and 'S' in line:
                    numS = num
            if len(states)>4:
                sta = states[1:]
            else:
                sta = states
            if numP > -1:
                pick = picks1a[numP]
                t = UTCDateTime(pick.split(' ')[-1])
                hundredth = int(round(t.microsecond / 1e4))
                if hundredth == 100:
                    t_p = t + 1
                    hundredth = 0
                else:
                    t_p = t
                date1 = t_p.strftime("%y%m%d%H%M%S") + ".%02d" % hundredth
                onset = 'I'
                polarity = '?'
                weight = 1
                #print(sta,onset,polarity,weight,date)
                hypo71_string += fmtP % (sta, onset, polarity, weight, date1)
                #f0.write(str(hypo71_string))

                #print(hypo71_string)
                if numP > -1 and numS > -1:
                    pick = picks1a[numS]
                    #t = UTCDateTime(pick[5])
                    t2 = UTCDateTime(pick.split(' ')[-1])
                    # if the S time's absolute minute is higher than that of the
                    # P pick, we have to add 60 to the S second count for the
                    # hypo 2000 output file
                    # +60 %60 is necessary if t.min = 57, t2.min = 2 e.g.
                    mindiff = (t2.minute - t.minute + 60) % 60
                    abs_sec = t2.second + (mindiff * 60)
                    hundredth = int(round(t2.microsecond / 1e4))
                    if hundredth == 100:
                        abs_sec += 1
                        hundredth = 0
                    date2 = "%s.%02d" % (abs_sec, hundredth)
                    hundredth = int(round(t.microsecond / 1e4))
                    if hundredth == 100:
                        t_p = t + 1
                        hundredth = 0
                    else:
                        t_p = t
                    #date = t_p.strftime("%y%m%d%H%M%S") + ".%02d" % hundredth
                    onset = 'I'
                    polarity = '?'
                    weight = 1
                    #print(sta,onset,polarity,weight,date)
                    hypo71_string += fmtS % (date2, onset, polarity,weight)

                else:
                    hypo71_string += "\n"

        try:
            if os.path.exists(project_folder+'/out.sum'):
                os.system('rm '+project_folder+'/out.sum')
            if vel_model is None:
                velmodel = pathhyp+'/standard.crh'
                os.system("cp %s %s" % (velmodel,project_folder))
                os.system("cat %s/run.hyp | hyp2000" % (project_folder))
                vel_model = 'standard.crh'
                #os.system("mv %s %s" % (original1,mseed1))
        except:
            pass
        fcur = open(project_folder+'/pha','w')
        fcur.write(str(hypo71_string))
        fcur.close()

        frun = open(project_folder+'/run.hyp','w')
        frun.write("crh 1 "+vel_model)
        frun.write("\n")
        frun.write('h71 3 2 2')
        frun.write("\n")
        frun.write("sta '"+project_folder+"/sta'")
        frun.write("\n")
        frun.write("phs '"+project_folder+"/pha'")
        frun.write("\n")
        frun.write('pos 1.78')
        frun.write("\n")
        frun.write('jun t')
        frun.write("\n")
        frun.write('min 4')
        frun.write("\n")
        frun.write('fil')
        frun.write("\n")
        frun.write('sum out.sum')
        frun.write("\n")
        frun.write('loc')
        frun.write("\n")
        frun.write('stop')
        frun.close()
        try:
            os.system("cat %s/run.hyp | hyp2000" % (project_folder))
        except:
            pass

        try:
            lines = open(project_folder+'/out.sum').readlines()
            for line in lines:
                if line.startswith("   DATE"):
                    print(' ')
            else:
                #model = 'standard'
                model = vel_model[:-4]
                year = int('20'+line[2:4])
                month = int(line[5:7])
                day = int(line[8:10])
                hour = int(line[11:13])
                minute = int(line[14:16])
                seconds = float(line[17:22])
                time = UTCDateTime(year, month, day, hour, minute, seconds)
                lat = float(line[23:31])
                #lat_min = float(line[28:33])
                #lat = lat_deg + (lat_min / 60.)
                #if lat_negative:
                #    lat = -lat
                lon = float(line[32:41])
                #lon_min = float(line[39:44])
                #lon = lon_deg + (lon_min / 60.)
                #if lon_negative:
                #    lon = -lon

                depth = float(line[42:48]) # depth: negative down!
                rms = float(line[75:80])
                errXY = float(line[81:86])
                errZ = float(line[87:92])

                gap  = float(line[65:69])
                o = Origin()
                #self.catalog[0].set_creation_info_username(self.username)
                o.clear()
                o.method_id = "/".join(["smi:local", "location_method", "hyp2000"])
                o.origin_uncertainty = OriginUncertainty()
                o.quality = OriginQuality()
                ou = o.origin_uncertainty
                oq = o.quality
                o.longitude = lon
                o.latitude = lat
                o.depth = depth * (1e3)  # meters positive down!
                # all errors are given in km!
                ou.horizontal_uncertainty = errXY * 1e3
                ou.preferred_description = "horizontal uncertainty"
                o.depth_errors.uncertainty = errZ * 1e3
                oq.standard_error = rms
                oq.azimuthal_gap = gap
                o.depth_type = "from location"
                o.earth_model_id = "smi:local/earth_model/%s" % (model)
                o.time = time
                o.resource_id = ResourceIdentifier(id='smi:local/Origin/hyp2000location_1')
                o.arrivals = origin.arrivals
            event.origins.append(o)
            event.preferred_origin_id = o.resource_id
        except:
            pass
    return cat


def plot_map_catalog(cat=None):
#    catdfr = pd.read_csv(file,delimiter=r"\s+")
#    catdfr = catdfr.dropna()
    catdfr = simple_cat_df(cat)

    #catdfr = catdfr.reset_index(drop=True)
    #rutc = np.zeros((len(catdfr.index),1))




    from mpl_toolkits.basemap import Basemap
    # 1. Draw the map background
    #fig = plt.figure(figsize=(8, 8))
    plt.figure()
    lat0 = np.median(catdfr.iloc[:,0].values)
    lon0 = np.median(catdfr.iloc[:,1].values)
    m = Basemap(projection='lcc', resolution='h',
                lat_0=lat0, lon_0=lon0,
                width=1E6, height=.6E6)
    #m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    #m.drawcounties(color='gray')
    m.drawstates(color='gray')

    # 2. scatter city data, with color reflecting population
    # and size reflecting area
    m.scatter(catdfr.iloc[:,1].values,catdfr.iloc[:,0].values,s=catdfr.iloc[:,3].values**3*8,c=catdfr.index,marker='o',alpha=0.5,latlon=True)

    #m.scatter(catdfo.iloc[:,2].values,catdfo.iloc[:,1].values,s=catdfo.iloc[:,16].values**3*10,c=catdfo.index,marker='o',alpha=0.5,latlon=True)



    cbar = plt.colorbar()
    N_TICKS=8
    indexes = [catdfr.index[i].strftime('%Y-%m-%d') for i in np.linspace(0,catdfr.shape[0]-1,N_TICKS).astype(int)]

    #indexes = [catdfr.index[i].strftime('%Y-%m-%d') for i in np.linspace(0,catdfr.shape[0]-1,N_TICKS).astype(int)]
    cbar.ax.set_yticklabels(indexes)
    plt.show()
    plt.savefig('hypo_map.png')


def plot_gr_freq_catalog(cat=None,min_mag=2):
    catdf = simple_cat_df(cat)

    catdf['origintime'] = pd.to_datetime(catdf.index)

    catdf3 = catdf[catdf['magnitude']>=min_mag]

    m3eqcount = catdf3['origintime'].groupby(catdf3.origintime.dt.to_period("M")).agg('count')
    m3eqcountd = catdf3['origintime'].groupby(catdf3.origintime.dt.to_period("D")).agg('count')
    alleqcount = catdf['origintime'].groupby(catdf.origintime.dt.to_period("M")).agg('count')
    alleqcountd = catdf['origintime'].groupby(catdf.origintime.dt.to_period("D")).agg('count')

    df3 = m3eqcount.to_frame()
    df3d = m3eqcountd.to_frame()
    dfall = alleqcount.to_frame()
    dfalld = alleqcountd.to_frame()

    df3.index = df3.index.to_timestamp()
    df3d.index = df3d.index.to_timestamp()
    dfall.index = dfall.index.to_timestamp()
    dfalld.index = dfalld.index.to_timestamp()

    df3 = df3.resample('MS').sum()
    df3d = df3d.resample('D').sum()
    dfall = dfall.resample('MS').sum()
    dfalld = dfalld.resample('D').sum()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].step(dfall.index,dfall.origintime, color='k')
    #plt.xlim([datetime.date(2009, 1, 1), datetime.datetime.now()])
    axs[0, 0].set(ylabel = 'Earthquakes per month')
    axs[0, 1].step(dfalld.index,dfalld.origintime, color='k')
    #plt.xlim([datetime.date(2009, 1, 1), datetime.datetime.now()])
    axs[0, 1].set(ylabel = 'Earthquakes per day')
    axs[1, 0].step(df3.index,df3.origintime, color='k')
    #plt.xlim([datetime.date(2009, 1, 1), datetime.datetime.now()])
    axs[1, 0].set(ylabel = 'Earthquakes M>'+str(min_mag)+' per month')
    axs[1, 1].step(df3d.index,df3d.origintime, color='k')
    #plt.xlim([datetime.date(2009, 1, 1), datetime.datetime.now()])
    axs[1, 1].set(ylabel = 'Earthquakes M>'+str(min_mag)+' per day')
    plt.show()

    plt.figure()
    plt.savefig('freq_plot.png')

    rangemin = np.floor(np.min(catdf['magnitude'].values[~np.isnan(catdf['magnitude'].values)]))
    rangemax = np.ceil(np.max(catdf['magnitude'].values[~np.isnan(catdf['magnitude'].values)]))

    hist, edges = np.histogram(a=catdf['magnitude'].values[~np.isnan(catdf['magnitude'].values)], bins=101, range=(rangemin,rangemax))
    chist = np.cumsum(hist[::-1])[::-1]


    fig, ax = plt.subplots()
    ax.plot(edges[:-1], hist, marker='.', color='k', linestyle='')
    ax.plot(edges[:-1], chist, marker='o', color='k', linestyle='',label='')

    ax.set_yscale('log')
    ax.set_ylabel('N')
    ax.set_xlabel('Magnitude')
#    ax.set_xlim(1, 6)
#    ax.set_ylim(1e0, 4e4)
    ax.set_title('Gutenburg-Richter Distribution')
    plt.show()
    plt.savefig('gr_plot.png')



if __name__ == "__main__":
    easyQuake()





#start_date = date(2013, 12, 1)
#end_date = date(2013, 12, 2)
##end_date = date(2018, 1, 2)
#project_code = 'eq'
#project_folder = '/data/EasyQuake'
#for single_date in daterange(start_date, end_date):
#    print(single_date.strftime("%Y-%m-%d"))
#    dirname = single_date.strftime("%Y%m%d")
    #tempdate = glob.glob('/scratch/antarctica/TempWaveform/'+single_date.strftime("%Y%m%d")+'*/*SAC.bp')
    #os.path.basename(temp[0])
    #stanames = []
    #for f in tempdate:
        #print(f)
    #    stanames.append(os.path.basename(f).split(".")[1]+'.'+os.path.basename(f).split(".")[2])
    #stachan_uniq = set(stanames)
