#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
set of functions to drive easyQuake
"""

print(r"""

                         ____              __
  ___  ____ ________  __/ __ \__  ______ _/ /_____
 / _ \/ __ `/ ___/ / / / / / / / / / __ `/ //_/ _ \
/  __/ /_/ (__  ) /_/ / /_/ / /_/ / /_/ / ,< /  __/
\___/\__,_/____/\__, /\___\_\__,_/\__,_/_/|_|\___/
               /____/


Earthquake detection and location open-source software
Jake Walter - Oklahoma Geological Survey
http://github.com/jakewalter/easyQuake
https://easyquake.readthedocs.io
jwalter@ou.edu

""")

#import sys
#sys.path.append("/home/jwalter/syncpython")
from .phasepapy import fbpicker
pathgpd = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/gpd_predict'
pathEQT = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/EQTransformer'
pathhyp = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/hyp2000'
pathphasenet = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/phasenet'
pathseisbench = '/'.join(str(fbpicker.__file__).split("/")[:-2])+'/seisbench'

from .phasepapy import tables1D, assoc1D
from .phasepapy import tt_stations_1D
from .sta_lta.trigger_p_s import trigger_p_s
#from .sta_lta.trigger_p_s import trigger_p_s

import traceback
import os
#st = os.stat(pathgpd+'/gpd_predict.py')
#st1 = os.stat(pathEQT+'/mseed_predictor.py')
#import stat

from multiprocessing import Pool
from multiprocessing import cpu_count
#from multiprocessing import set_start_method
# set_start_method("spawn")
# from multiprocessing import get_context

#import os
from obspy import UTCDateTime
from obspy import Inventory, read_inventory
from obspy.clients.fdsn import Client
from obspy import read
import numpy as np
import glob

import obspy.taup as taup
from obspy.taup import TauPyModel
#from obspy.taup.velocity_model import VelocityModel
from obspy.taup.taup_create import build_taup_model
from obspy import geodetics
from obspy.clients.fdsn.mass_downloader import CircularDomain, RectangularDomain, Restrictions, MassDownloader
from obspy.core.event.base import WaveformStreamID
from sqlalchemy.orm import *
from sqlalchemy import create_engine
import pandas as pd
import sqlite3
from sqlite3 import Error
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

import re
from datetime import datetime


from obspy import Stream
from obspy.core.event import Catalog, Event, Magnitude, Origin, Pick, StationMagnitude, Amplitude, Arrival, OriginUncertainty, OriginQuality, ResourceIdentifier, Comment

import h5py



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


def download_mseed(dirname=None, project_folder=None, single_date=None, minlat=None, maxlat=None, minlon=None, maxlon=None, dense=False, raspberry_shake=False):
    """
    Downloads seismic data in miniSEED format from IRIS DMC within a specified geographic and temporal domain.
    
    Args:
    dirname (str): Name of directory where miniSEED files will be stored. If None, the directory will be named after the specified date.
    project_folder (str): Name of project folder where miniSEED files will be stored. If None, files will be stored in the current directory.
    single_date (datetime): Date in YYYY-MM-DD format specifying the day for which data will be downloaded.
    minlat (float): Minimum latitude for the geographic domain.
    maxlat (float): Maximum latitude for the geographic domain.
    minlon (float): Minimum longitude for the geographic domain.
    maxlon (float): Maximum longitude for the geographic domain.
    dense (bool): Whether to download data with high temporal and spatial resolution. If True, data with minimum inter-station distance of 1 meter will be downloaded. Otherwise, data with minimum inter-station distance of 5000 meters will be downloaded.
    raspberry_shake (bool): Whether to download data from Raspberry Shake stations in addition to the standard IRIS DMC stations.
    
    Returns:
    None: The function only downloads miniSEED files and does not return any values.
    
    Raises:
    None: The function does not raise any exceptions.
    
    Example:
    To download miniSEED files for January 1, 2023 in the geographic domain bounded by 40 to 42 degrees latitude and -120 to -118 degrees longitude with high temporal and spatial resolution and save them in a directory called "data" within a project folder called "project":
    
    python
    Copy code
    >>> date = datetime.datetime(2023, 1, 1)
    >>> download_mseed(dirname="data", project_folder="project", single_date=date, minlat=40, maxlat=42, minlon=-120, maxlon=-118, dense=True, raspberry_shake=False)
    
    """
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
    if raspberry_shake:
        mdl = MassDownloader(['RASPISHAKE'])
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



def build_tt_tables(lat1=None,long1=None,maxrad=None,starting=None, stopping=None, channel_codes=['EH','BH','HH','HN'],db=None,maxdist=500.,source_depth=5., delta_distance=1, model=None):
    """
    """
    # Create a connection to an sqlalchemy database
    tt_engine=create_engine(db,echo=False, connect_args={'check_same_thread': False})
    tt_stations_1D.BaseTT1D.metadata.create_all(tt_engine)
    TTSession=sessionmaker(bind=tt_engine)
    tt_session=TTSession()
    fdsnclient=Client()
    inv=fdsnclient.get_stations(starttime=starting,endtime=stopping,latitude=lat1,longitude=long1,maxradius=maxrad,channel='*H*',level='station')
    # Get inventory
    for net in inv:
        network=net.code
        for sta in net:
            # loccodes=[]
            # for ch in sta:
            #     for cc in channel_codes:
            #       if re.match(cc,ch.code):
            #         if not ch.location_code in loccodes:
            #           loccodes.append(ch.location_code)
            # for loc in loccodes:
            loc = ''
            print(sta.code,network,loc,sta.latitude,sta.longitude,sta.elevation)
            station=tt_stations_1D.Station1D(sta.code,network,loc,sta.latitude,sta.longitude,sta.elevation)
            tt_session.add(station)
    tt_session.commit()

    # Now we have to build our traveltime lookup tables
    # We will use IASP91 here but obspy.taup does let you build your own model
    if model is not None:
        filename = model
        #vmodel = VelocityModel.read_tvel_file(filename)    
        if os.path.exists(project_folder+'/'+f"{filename[:-5]}.npz"):
            velmod = TauPyModel(model=project_folder+'/'+f"{filename[:-5]}.npz")
        else:
            taup_model = build_taup_model(filename, output_folder=os.getcwd())
            velmod = TauPyModel(model=project_folder+'/'+f"{filename[:-5]}.npz")
    else:
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

def build_tt_tables_local_directory(dirname=None,project_folder=None,channel_codes=['EH','BH','HH','HN'],db=None,maxdist=800.,source_depth=5.,delta_distance=1, model=None):
    """
    This function builds travel-time lookup tables for seismic stations located in a specified directory using a specified model. The function takes several optional arguments, including the directory name, project folder, channel codes, database, maximum distance, source depth, delta distance, and model.

    The function begins by creating a connection to an SQLalchemy database and creating the necessary tables. It then reads in the station inventory from the specified directory and populates the station table in the database with the relevant information.
    
    The function then uses the specified travel-time model or the default IASP91 model to calculate travel times for each distance between 0 and the specified maximum distance, in increments of the specified delta distance. For each distance, the function calculates the minimum travel time for P- and S-waves and stores this information in the travel time table in the database.
    
    The function returns the station inventory.
    
    Args:
        dirname (str, optional): The directory containing the station inventory files. Defaults to None.
        project_folder (str, optional): The project folder containing the station inventory directory. Defaults to None.
        channel_codes (list of str, optional): The channel codes to be included. Defaults to ['EH', 'BH', 'HH', 'HN'].
        db (str, optional): The SQLAlchemy database connection string. Defaults to None.
        maxdist (float, optional): The maximum distance for which to calculate travel times, in km. Defaults to 800.0.
        source_depth (float, optional): The depth of the seismic source, in km. Defaults to 5.0.
        delta_distance (int, optional): The spacing between distances for which to calculate travel times, in km. Defaults to 1.
        model (str, optional): The name of the travel-time model to use. Defaults to None.
    Returns:
        inv (Inventory): The station inventory, populated with information from the specified directory.
    """
    # Create a connection to an sqlalchemy database
    tt_engine=create_engine(db,echo=False, connect_args={'check_same_thread': False})
    tt_stations_1D.BaseTT1D.metadata.create_all(tt_engine)
    TTSession=sessionmaker(bind=tt_engine)
    tt_session=TTSession()
    inv = Inventory()
    dir1a = glob.glob(project_folder+'/'+dirname+'/dailyinventory.xml') + glob.glob(project_folder+'/'+dirname+'/??.*.xml')
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
    if model is not None:
        filename = model
        #vmodel = VelocityModel.read_tvel_file(filename)
        if os.path.exists(project_folder+'/'+f"{filename[:-5]}.npz"):
            velmod = TauPyModel(model=project_folder+'/'+f"{filename[:-5]}.npz")
        else:
            taup_model = build_taup_model(filename, output_folder=os.getcwd())
            velmod = TauPyModel(model=project_folder+'/'+f"{filename[:-5]}.npz")

    else:
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
        #print(d_km,ptimes,stimes)
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
    dir1a = glob.glob(project_folder+'/'+dirname+'/dailyinventory.xml') + glob.glob(project_folder+'/'+dirname+'/??.*.xml')
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



                    
def queue_sta_lta(infile,outfile,dirname,filtmin=2, filtmax=15, t_sta=0.2, t_lta=2.5, trigger_on=4, trigger_off=2, trig_horz=6, trig_vert=10):
    #add sta/lta stuff
    fdir = []
    with open(infile) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)
    n_cpus1 = min(cpu_count(), nsta)
    if n_cpus1 == cpu_count():
        n_cpus = n_cpus1-1
    else:
        n_cpus = n_cpus1
    #with get_context("spawn").Pool() as pool:
    pool = Pool(n_cpus-1)
    #results = []
    for i in range(nsta):
        #try:
        print(str(i+1)+" of "+str(nsta)+" stations")
        print(fdir[i],outfile.split('.')[0]+str(i), filtmin, filtmax, t_sta, t_lta, trigger_on, trigger_off,trig_horz, trig_vert,)
        pool.apply(trigger_p_s, (fdir[i],outfile.split('.')[0]+str(i), filtmin, filtmax, t_sta, t_lta, trigger_on, trigger_off, trig_horz, trig_vert,))
        #print(r.get())
        #results.append((i,r))
    pool.close()
    pool.join()
    if os.path.exists(outfile):
        os.remove(outfile)
    filenames = glob.glob(outfile.split('.')[0]+'*')
    with open(outfile, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    for file1 in filenames:
        os.remove(file1)
                    

def pick_add(dbsession=None,fileinput=None,inventory=None):
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

def make_dayfile(dir1, make3):
    filelist = glob.glob(dir1+'/*mseed') or glob.glob(dir1+'/*SAC')
    stations = set()
    for file1 in filelist:
        station = file1.split('.')[1]
        net = file1.split('.')[0].split('/')[-1]
        netsta = net+'.'+station
        print(file1.split('.')[1])
        stations.add(netsta)
        
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
        if all(elem is None for elem in station3a): #check location code
            ind1 = np.empty((len(station3),1))
            ind1[:] = np.nan
            
            for station1 in station3:
                if station1.split('.')[2] == '00':
                    if get_chan1(station1)  == 'Z':
                        ind1 = 2
                    elif get_chan1(station1)  == 'N' or get_chan1(station1) == '1':
                        ind1 = 0
                    elif get_chan1(station1)  == 'E' or get_chan1(station1) == '2':
                        ind1 = 1
                    #print(ind1)
                    station3a[ind1] = station1       
        if any(elem is None for elem in station3a):
            if make3: #make single vertical comp, 3 channels
                if station3a[-1] is not None and station3a[0] is None and station3a[1] is None:
                    st = read(station3a[-1])
                    st.merge()
                    for tr in st:
                        if isinstance(tr.data, np.ma.masked_array):
                            tr.data = tr.data.filled()
                    
                    if len(station3a[-1].split('__')) == 1:
                        st[0].stats.channel = st[0].stats.channel[0:2]+'E'
                        st[0].write('.'.join(station3a[-1].split('__')[0].split('.')[0:3])+'.'+st[0].stats.channel[0:2]+'E.mseed')
                    else:
                        st[0].stats.channel = st[0].stats.channel[0:2]+'E'
                        st[0].write('.'.join(station3a[-1].split('__')[0].split('.')[0:3])+'.'+st[0].stats.channel[0:2]+'E'+'__'+'__'.join(station3a[-1].split('__')[1:3]))
                    if len(station3a[-1].split('__')) == 1:
                        st[0].stats.channel = st[0].stats.channel[0:2]+'N'
                        st[0].write('.'.join(station3a[-1].split('__')[0].split('.')[0:3])+'.'+st[0].stats.channel[0:2]+'N.mseed')
                    else:
                        st[0].stats.channel = st[0].stats.channel[0:2]+'N'
                        st[0].write('.'.join(station3a[-1].split('__')[0].split('.')[0:3])+'.'+st[0].stats.channel[0:2]+'N'+'__'+'__'.join(station3a[-1].split('__')[1:3]))
                    station3 = glob.glob(dir1+'/*'+stationin+'.*mseed') or glob.glob(dir1+'/*'+stationin+'.*SAC')
                    station3a = [None,None,None]
                    for station1 in station3:
                        if get_chan1(station1)  == 'Z':
                            ind1 = 2
                        elif get_chan1(station1)  == 'N' or get_chan1(station1) == '1':
                            ind1 = 0
                        elif get_chan1(station1)  == 'E' or get_chan1(station1) == '2':
                            ind1 = 1
                        #print(ind1)
                        station3a[ind1] = station1
                    print(station3a)

        if any(elem is None for elem in station3a):
            continue
            #continue
        day_strings.append((station3a[0]+' '+station3a[1]+' '+station3a[2]))

    day_string = "\n".join(day_strings)

    with open(dir1+'/dayfile.in', "w") as open_file:
        open_file.write(day_string)
    return dir1+'/dayfile.in'
        

def detection_continuous(dirname=None, project_folder=None, project_code=None, local=True, machine=True, machine_picker=None, single_date=None, make3=True, latitude=None, longitude=None, max_radius=None, fullpath_python=None, filtmin=2, filtmax=15, t_sta=0.2, t_lta=2.5, trigger_on=4, trigger_off=2, trig_horz=6.0, trig_vert=10.0, seisbenchmodel=None):
    """
    Continuous detection of seismic events using single-station waveform data.
    
    Args:
    
        dirname (str, optional): The directory name where the data will be stored.
        project_folder (str, optional): The project directory where the data will be stored.
        project_code (str, optional): The project code name.
        local (bool, optional): If True, read in data from a local directory. If False, read in data from an online database.
        machine (bool, optional): If True, use an automated detection algorithm. If False, use a manual detection algorithm.
        machine_picker (str, optional): The name of the automated detection algorithm to use. Only used if machine=True. Defaults to 'GPD'.
        single_date (datetime.datetime, optional): The date and time to start detection.
        make3 (bool, optional): Whether to split data into 3-hour chunks. Defaults to True.
        latitude (float, optional): The latitude of the location where data is to be collected. Only used if local=False.
        longitude (float, optional): The longitude of the location where data is to be collected. Only used if local=False.
        max_radius (float, optional): The maximum distance, in degrees, from the latitude and longitude to search for stations. Only used if local=False.
        fullpath_python (str, optional): The path to the Python executable. Only used if machine=True.
        filtmin (float, optional): The minimum frequency for the filter.
        filtmax (float, optional): The maximum frequency for the filter.
        t_sta (float, optional): The length of the short-term average window for the STA/LTA algorithm.
        t_lta (float, optional): The length of the long-term average window for the STA/LTA algorithm.
        trigger_on (float, optional): The threshold for triggering an event.
        trigger_off (float, optional): The threshold for ending an event.
        trig_horz (float, optional): The horizontal distance between events required for them to be considered separate.
        trig_vert (float, optional): The vertical distance between events required for them to be considered separate.
        seisbenchmodel (str, optional): The full path and model name of the seisbench trained model
    Returns:
        None
    
    This function performs continuous detection of seismic events using waveform data from a single station. It creates an SQLite database for storing the detection results, and uses automated detection algorithms (GPD, EQTransformer, or PhaseNet), or alternatively, STA/LTA.
    """
    

#    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
#    stopping = starting + 86430
    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
    stopping = starting + 86430
    dir1 = project_folder+'/'+dirname
    #print(single_date.strftime("%Y%m%d"))
    #print(dir1+'/1dassociator_'+project_code+'.db')
    if machine == True and machine_picker is None:
        machine_picker = 'GPD'
    if machine == False:
        machine_picker = 'STALTA'
    if os.path.exists(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'):
        os.remove(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db')
    db_assoc='sqlite:///'+dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'
    engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
    tables1D.Base.metadata.create_all(engine_assoc)
    Session=sessionmaker(bind=engine_assoc)
    session=Session()

    #### create infile
    
    infile = make_dayfile(dir1, make3)

    infile = dir1+'/dayfile.in'
    outfile = dir1+'/gpd_picks.out'

    #remove this later as it is called in association module?
    if local:
        inv = Inventory()
        dir1a = glob.glob(project_folder+'/'+dirname+'/dailyinventory.xml') + glob.glob(project_folder+'/'+dirname+'/??.*.xml')
        for file1 in dir1a:
            inv1a = read_inventory(file1)
            inv.networks.extend(inv1a)
    else:
        fdsnclient=Client()
        inv=fdsnclient.get_stations(starttime=starting,endtime=stopping,latitude=latitude,longitude=longitude,maxradius=max_radius,channel='*HZ',level='station')

    if machine == True and machine_picker is None:
        machine_picker = 'GPD'
    if machine == True and machine_picker == 'GPD':
        fullpath1 = pathgpd+'/gpd_predict.py'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if os.path.exists(outfile):
            os.remove(outfile)
        if fullpath_python:
            os.system(fullpath_python+" "+fullpath1+" -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        else:
            os.system("gpd_predict -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        try:
            pick_add(dbsession=session,fileinput=outfile,inventory=inv)
        except:
            pass
    elif machine == True and machine_picker == 'EQTransformer':
        fullpath2 = pathEQT+'/mseed_predictor.py'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if os.path.exists(outfile):
            os.remove(outfile)
        if fullpath_python:
            os.system(fullpath_python+" "+fullpath2+" -I %s -O %s -F %s" % (infile, outfile, pathEQT))
        else:
            os.system("mseed_predictor -I %s -O %s -F %s" % (infile, outfile, pathEQT))
        try:
            pick_add(dbsession=session,fileinput=outfile,inventory=inv)
        except:
            pass
    elif machine == True and machine_picker == 'PhaseNet':
        fullpath3 = pathphasenet+'/phasenet_predict.py'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if os.path.exists(outfile):
            os.remove(outfile)
        if fullpath_python:
            #print(pathphasenet)
            #python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/mseed.csv --data_dir=test_data/mseed --format=mseed --plot_figure
            os.system(fullpath_python+" "+fullpath3+" --model=%s/model/190703-214543 --data_list=%s --format=mseed --result_fname=%s --result_dir=%s" % (pathphasenet, infile, outfile, dir1))
        else:
            os.system("phasenet_predict --model=%s/model/190703-214543 --data_list=%s --format=mseed --result_fname=%s --result_dir=%s" % (pathphasenet, infile, outfile, dir1))
        try:
            pick_add(dbsession=session,fileinput=outfile,inventory=inv)
        except:
            pass
    elif machine == True and machine_picker == 'Seisbench':
        fullpath3 = pathseisbench+'/run_seisbench.py'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if os.path.exists(outfile):
            os.remove(outfile)
        if fullpath_python:
            #print(pathphasenet)
            #python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/mseed.csv --data_dir=test_data/mseed --format=mseed --plot_figure
            os.system(fullpath_python+" "+fullpath3+" -I %s -O %s -M %s" % (infile, outfile, seisbenchmodel))
        else:
            os.system("run_seisbench -I %s -O %s -M %s" % (infile, outfile, seisbenchmodel))
        try:
            pick_add(dbsession=session,fileinput=outfile,inventory=inv)
        except:
            pass
        
    else:
        machine_picker = 'STALTA'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if os.path.exists(outfile):
            os.remove(outfile)
        queue_sta_lta(infile, outfile, dirname, filtmin, filtmax, t_sta, t_lta, trigger_on, trigger_off, trig_horz, trig_vert)
        try:
            pick_add(dbsession=session,fileinput=outfile,inventory=inv)
        except:
            pass

        #picker = fbpicker.FBPicker(t_long = 5, freqmin = 1, mode = 'rms', t_ma = 20, nsigma = 7, t_up = 0.7, nr_len = 2, nr_coeff = 2, pol_len = 10, pol_coeff = 10, uncert_coeff = 3)
        #fb_pick(dbengine=engine_assoc,picker=picker,fileinput=infile)

def association_continuous(dirname=None, project_folder=None, project_code=None, maxdist = None, maxkm=None, single_date=None, local=True, nsta_declare=4, delta_distance=1, machine=True, machine_picker=None, latitude=None, longitude=None, max_radius=None, model=None, delete_assoc=False):
    """
    association_continuous: A function that performs association of seismic events with continuous data
    
    Args:
        dirname (str): Name of the directory where the project is located.
        project_folder (str): Name of the project folder.
        project_code (str): Project code name.
        maxdist (float): Maximum distance to search for stations (km).
        maxkm (float): Maximum epicentral distance for association (km).
        single_date (datetime): A datetime object that represents a single date.
        local (bool): Flag to indicate if the TT tables will be built from local data or will be downloaded from IRIS.
        nsta_declare (int): Minimum number of stations required to declare an event.
        delta_distance (float): Distance increment to create the TT tables (km).
        machine (bool): Flag to indicate which picking algorithm will be used.
        machine_picker (str): Name of the picking algorithm to be used.
        latitude (float): Latitude of the epicenter.
        longitude (float): Longitude of the epicenter.
        max_radius (float): Maximum radius to search for stations.
        model (str): Velocity model to use in the TT tables.
        delete_assoc (bool): Flag to indicate if the old association database will be deleted.
    
    Returns:
        None
    """
    starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0')
    stopping = starting + 86430
        
    dir1 = project_folder+'/'+dirname
    print(single_date.strftime("%Y%m%d"))
    #1dassociator_'+machine_picker+'_'+project_code+'.db'
    if machine == True and machine_picker is None:
        machine_picker = 'GPD'
    if machine == False:
        machine_picker = 'STALTA'
        
    if delete_assoc:
        if os.path.exists(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'):
            os.remove(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db')

    #print(dir1+'/1dassociator_'+project_code+'.db')
#    if os.path.exists(dir1+'/1dassociator_'+project_code+'.db'):
#        os.remove(dir1+'/1dassociator_'+project_code+'.db')
#    db_assoc='sqlite:///'+dir1+'/1dassociator_'+project_code+'.db'
    if os.path.exists(dir1+'/tt_ex_1D_'+machine_picker.lower()+'_'+project_code+'.db'):
        os.remove(dir1+'/tt_ex_1D_'+machine_picker.lower()+'_'+project_code+'.db')
    db_tt='sqlite:///'+dir1+'/tt_ex_1D_'+machine_picker.lower()+'_'+project_code+'.db' # Traveltime database44.448,longitude=-115.136
    #print(db_tt)
    if local:
        inventory = build_tt_tables_local_directory(dirname=dirname,project_folder=project_folder,channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5., delta_distance=delta_distance, model=model)
    else:
        inventory = build_tt_tables(lat1=latitude,long1=longitude,maxrad=max_radius,starting=starting, stopping=stopping, channel_codes=['EH','BH','HH'],db=db_tt,maxdist=maxdist,source_depth=5., delta_distance=delta_distance, model=model)
    inventory.write(dir1+'/dailyinventory.xml',format="STATIONXML")
    if not os.path.exists(dir1+'/station_list.csv'):
        stalat = []
        stalon = []
        staname = []
        netname = []
        elevs = []
        for net in inventory:
            for sta in net:
                stalat.append(sta.latitude)
                stalon.append(sta.longitude)
                staname.append(sta.code)
                netname.append(net.code)
                elevs.append(sta.elevation)
        stadf = pd.DataFrame({'net':netname,'sta':staname,'netsta':[a+'.'+b for a,b in zip(netname,staname)],'latitude':stalat,'longitude':stalon,'elevation (m)':elevs})
        stadf = stadf.drop_duplicates()
        stadf.to_csv(dir1+'/station_list.csv',index=False)
        # check if there is actually data there?
        # dayfile = pd.read_csv(dir1+'/dayfile.in', sep=" ", header=None)
        # for idx1 in dayfile.index:
        #     print(".".join(dayfile.iloc[idx1][0].split('/')[-1].split('.')[0:2]))
        #     match = stadf[stadf['netsta'].str.contains(".".join(dayfile.iloc[idx1][0].split('/')[-1].split('.')[0:2]))
    
    if not os.path.exists(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'):
        db_assoc='sqlite:///'+dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'
        engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
        tables1D.Base.metadata.create_all(engine_assoc)
        Session=sessionmaker(bind=engine_assoc)
        session=Session()
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        pick_add(dbsession=session,fileinput=outfile,inventory=inventory)

    db_assoc='sqlite:///'+dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'
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


def hypo_station(project_folder=None, project_code=None, catalog_year=None, year=None, daymode=None, single_date=None):
    hypo71_string_sta = ""
    station_strings = []
    if daymode:
        f1 = open(project_folder+'/'+'sta'+single_date.strftime("%Y%m%d"),'w')
    elif catalog_year:
        f1 = open(project_folder+'/'+'sta'+str(year),'w')
    else:
        f1 = open(project_folder+'/'+'sta','w')
    #f2 = open(project_folder+'/'+'station.dat', 'w')
    #for stas in temp:
    if catalog_year:
        files = sorted(glob.glob(project_folder+'/'+str(year)+'*/tt*'+project_code+'.db'))
    elif daymode:
        files = sorted(glob.glob(project_folder+'/'+single_date.strftime("%Y%m%d")+'/tt*'+project_code+'.db'))
    else:
        files = sorted(glob.glob(project_folder+'/*/tt*'+project_code+'.db')) or glob.glob(project_folder+'/tt*'+project_code+'.db')
    #print(files)
    stas1 = pd.DataFrame()
    for dfilesta in files:
        conn1 = create_connection(dfilesta)
        with conn1:
            cur1 = conn1.cursor()
            cur1.execute("SELECT * FROM stations")

            #rows = cur1.fetchall()

            for row in cur1:
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
    if catalog_year:
        station_filename = project_folder+'/'+str(year)+'station.dat'
    elif daymode:
        station_filename = project_folder+'/'+single_date.strftime("%Y%m%d")+'station.dat'
    else:
        station_filename = project_folder+'/'+'station.dat'
    with open(station_filename, "w") as open_file:
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

def combine_associated(project_folder=None, project_code=None, catalog_year=False, year=None, hypoflag=False, eventmode=False, daymode=False, single_date=None, machine_picker=None):
    # if machine_picker is None:
    #     machine_picker='*'
    # else:
    #     machine_picker = '_'+machine_picker.lower()
    if catalog_year:
        files = sorted(glob.glob(project_folder+'/'+str(year)+'*/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'))
        if machine_picker == 'GPD':
            files = sorted(glob.glob(project_folder+'/'+str(year)+'*/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db')) or sorted(glob.glob(project_folder+'/'+str(year)+'*/1dassociator_'+project_code+'.db'))
        hypo_station(project_folder, project_code, catalog_year=True, year=year)
        f0 = open(project_folder+'/pha_'+str(year)+'_'+project_code,'w')
    if eventmode:
        files = sorted(glob.glob(project_folder+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'))
        hypo_station(project_folder, project_code)
        f0 = open(project_folder+'/pha_'+project_code,'w')
    if daymode:
        files = sorted(glob.glob(project_folder+'/'+single_date.strftime("%Y%m%d")+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'))
        hypo_station(project_folder, project_code, single_date=single_date, daymode=True)
        f0 = open(project_folder+'/pha_'+single_date.strftime("%Y%m%d")+'_'+project_code,'w')
    if (catalog_year is False) and (daymode is False) and (eventmode is False):
        if machine_picker is None:
            machine_picker='*'
        else:
            machine_picker = machine_picker.lower()
        files = sorted(glob.glob(project_folder+'/*/1dassociator_'+machine_picker+'_'+project_code+'.db'))
        if machine_picker == 'GPD':
            files = sorted(glob.glob(project_folder+'/*/1dassociator'+machine_picker+'_'+project_code+'.db')) or sorted(glob.glob(project_folder+'/*/1dassociator_'+project_code+'.db'))
        hypo_station(project_folder, project_code)
        f0 = open(project_folder+'/pha_'+project_code,'w')

    
    #f0 = open(project_folder+'/pha_'+project_code,'w')
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
    # if catalog_year:
    #     cat.write(project_folder+'/'+project_code+'_'+str(year)+'_cat.xml',format="QUAKEML")
    # else:
    #     if not eventmode:
    #         cat.write(project_folder+'/'+project_code+'_cat.xml',format="QUAKEML")
    return cat, dfs2

def pytocto_file_quakeml(file):
    """
    Generate a Catalog object in QuakeML with a pyocto csv file
    """
    df = pd.read_csv(file, index_col=0)
    nevents = np.max(df['event_idx'])
    cat1 = Catalog()
    for evid1 in np.arange(0,nevents):
        #print(evid)
        event1 = df[df['event_idx']==evid1]
        origin = Origin()
        origin.latitude = event1['latitude'].iloc[0]
        origin.longitude = event1['longitude'].iloc[0]
        origin.depth = event1['depth'].iloc[0]*1000
        
        origin.time = UTCDateTime(event1['time'].iloc[0][:-6])
        origin.arrivals = []
        event = Event()
        evid = 'smi:local/Event/'+str(evid1)
        orid = 'smi:local/Origin/pyocto_association_'+str(evid1)
        event.resource_id = ResourceIdentifier(id=evid)
        origin.resource_id = ResourceIdentifier(id=orid)
        # event.resource_id = ResourceIdentifier(id='smi:local/Event/'+strday+str(rownum).zfill(3))
        # origin.resource_id = ResourceIdentifier(id='smi:local/Origin/'+strday+str(rownum).zfill(3)+'_1')
        for idx in event1.index:
            pick = df.iloc[idx]
            stream_id = WaveformStreamID(network_code=pick['station'].split('.')[0], station_code=pick['station'].split('.')[1], location_code="", channel_code=pick['channel'])
            p = Pick()
            p.time = UTCDateTime(datetime.datetime.utcfromtimestamp(pick['time_pick']))
            p.phase_hint = pick['phase']
            p.waveform_id = stream_id
            p.evaluation_mode = 'automatic'
            pres_id = 'smi:local/Pick/'+str(pick['pick_idx'])
            #res_id = ResourceIdentifier(prefix='Pick')
            #res_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
            p.resource_id = ResourceIdentifier(id=pres_id)
            #print(p)
    
            a = Arrival()
            #a.time = pick1[5]
            a.phase = pick['phase']
            a.pick_id = p.resource_id
            ares_id = 'smi:local/Arrival/'+str(pick['pick_idx'])
            #res_id = ResourceIdentifier(prefix='Pick')
            #res_id.convert_id_to_quakeml_uri(authority_id='obspy.org')
            a.resource_id = ResourceIdentifier(id=ares_id)
            a.time_weight = 1.0
            if not np.isnan(pick['residual']):
                a.time_residual = pick['residual']
            #print(a)
    
            # #origin.picks.append(p)
            # sta1 = pick1[1]
            # stas.append(sta1)
            # stalistall.add(sta1)
            origin.arrivals.append(a)
            event.picks.append(p)
        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id
        cat1.append(event)
    return cat1

def polarity(tr,pickP=None):
    """
    This function determines the polarity of a given seismogram trace, with respect to a given pickP value.
    It calculates the standard deviation of the trace and checks whether the difference between the value at pickP
    and the next maximum or minimum exceeds a threshold. If so, it determines the polarity as positive or negative,
    otherwise it sets the polarity as undecidable.
    
    :param tr: ObsPy Trace object containing the seismogram data.
    :type tr: obspy.core.trace.Trace
    :param pickP: The time in seconds after the trace start time at which the seismic arrival
    of interest (e.g. the P-wave) occurs. If not provided, defaults to None.
    :type pickP: float or None
    :return: The polarity of the seismogram trace as a string, which can be either "positive", "negative", or "undecidable".
    :rtype: str
    
    """
    #dt=tr.stats.delta
    #t = np.arange(0, tr.stats.npts/tr.stats.sampling_rate, dt)
    #index0=int(round((pickP-tr.stats.starttime)/dt,0))
    #index=index0
    #pol_coeff = 10
    #pol_len = 10
    #polarity = 'undecidable'
    # while True:
    #     if index>=tr.stats.npts-1-2:
    #         break
    #     elif (tr[index+1]-tr[index])*(tr[index+2]-tr[index+1])>0:
    #         index+=1
    #     else:
    #         break
    #     if tr[index+1] - tr[index0] > 0 and abs(tr[index+1] - tr[index0]) > pol_coeff * np.std(tr[index0 - pol_len: index0]):
    #         polarity='positive'
    #     elif tr[index+1] - tr[index0] < 0 and abs(tr[index+1] - tr[index0]) > pol_coeff * np.std(tr[index0 - pol_len: index0]):
    #         polarity='negative'
    #     else:
    #         polarity='undecidable'
    tr.filter('bandpass',freqmin=2,freqmax=15,corners=5,zerophase=True)
    polarity = 'undecidable'
    pwin = 2*tr.stats.sampling_rate # 2 sec window
    pickindP = (pickP - tr.stats.starttime)*tr.stats.sampling_rate-1
    pamp = np.max(tr.data[int(pickindP-pwin/2):int(pickindP + pwin/2)])
    pamp_min = np.min(tr.data[int(pickindP-pwin/2):int(pickindP + pwin/2)])
    pampt = np.where(tr.data==pamp)[0]
    pampt_min = np.where(tr.data==pamp_min)[0]
    if pampt<pampt_min:
        polarity='positive'
    elif pampt>pampt_min:
        polarity='negative'
    if pamp < 3*np.std(tr.data[int(pickindP-pwin*2):int(pickindP-pwin)]):
        polarity='undecidable'
    return polarity

                        #halfpamp = (pamp-pamp_min)/2
                            # tempa = np.array(argrelextrema((st3[0].data), np.less))[0]
                            # signa = -1
                            # indexa1 = (np.where(st3[0].data[tempa]==np.min(st3[0].data[tempa])))
                            # indexa = np.array(indexa1)[0][0]
                            # maxamp = st3[0].data[tempa][indexa]




def sp_ratio(st3,inv,pickP=None,all_picks=None,event=None):
    """
    This function determines the SP ratio of a given seismogram trace
    
    3c, rotate, Pamp - first peak for P, Samp - max S displacement amplitude on any component
    
    Following Hardebeck and Shearer 2003, but using Shelly et al, 2022 peak-to-peak amplitudes approach for calculating P amplitude (slightly smaller S-P ratios result)
    
    st3,inv,pick,event.picks,event
    """
    st3 = st3.merge()
    st3.detrend('demean')
    st3.taper(max_percentage=None,max_length=5)
    pre_filt = (0.05, 0.06, 30.0, 35.0)
    st3.remove_response(inventory=inv, output='DISP', pre_filt=pre_filt, zero_mean=True)
    st3.filter('bandpass',freqmin=2,freqmax=15,corners=5,zerophase=True)
    
    epid, az, baz = gps2dist_azimuth(event.preferred_origin().latitude, event.preferred_origin().longitude, inv[0][0].latitude, inv[0][0].longitude)
    #rotate to radial/transverse
    #check if it is Z12 rather than ZNE
    if len(st3.select(channel='*[ZNE]')) == 1:
        st3.rotate(method="->ZNE",inventory=inv)
        st3.rotate(method="NE->RT", back_azimuth=baz)
    #already ZNE
    elif len(st3) == 3:
        st3.rotate(method="NE->RT", back_azimuth=baz)
    
    for p in all_picks:
        if p.waveform_id.station_code == pickP.waveform_id.station_code:
            pickS = p
    
    # ML pickers tend to pick the peaks in the data so this is adjusted to straddle the pick time
    pickindP = (pickP.time - st3[0].stats.starttime)*st3[0].stats.sampling_rate-1
    pwin = 2*st3[0].stats.sampling_rate # 2 sec window
    # sum of radial and transverse components
    pampdata = st3[1].data + st3[2].data
    #pampdata = np.sqrt(st3[1].data**2 + st3[2].data**2)   
    pamp = np.max(pampdata[int(pickindP-pwin/2):int(pickindP + pwin/2)])
    pamp_min = np.min(pampdata[int(pickindP-pwin/2):int(pickindP + pwin/2)])
    halfpamp = (pamp-pamp_min)/2
    
    pickindS = (pickS.time - st3[0].stats.starttime)*st3[0].stats.sampling_rate-1
    swin = 2*st3[0].stats.sampling_rate # 2 sec window
    # sum of radial and transverse components
    samp = np.max([np.max(st3[0].data[int(pickindS):int(pickindS + swin)]), np.max(st3[1].data[int(pickindS):int(pickindS + swin)]), np.max(st3[2].data[int(pickindS):int(pickindS + swin)])])
    
    
    sp_ratio = samp/pamp # just for checking - this is original way of calculating it but we won't use it
    sp_ratio_half = samp/halfpamp

    return sp_ratio_half


def select_3comp_remove_response(project_folder=None,strday=None,pick=None,starttime=None,endtime=None):
    paz_wa = {'sensitivity': 2080, 'zeros': [0j], 'gain': 1,'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

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
    st3.merge(fill_value='interpolate')
    print(st3)
    for tr in st3:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled()
    st = st3.select(channel='[EHB]H[EN12]')
    for tr in st3:
        inventory_local = glob.glob(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.xml') or glob.glob(project_folder+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.xml')
        if len(inventory_local)>0:
            inv = read_inventory(inventory_local[0])
        else:
            try:
                #inv0 = read_inventory(project_folder+'/'+strday+'*/dailyinventory.xml')
                try:
                    inv0 = read_inventory(project_folder+'/'+strday+'*/dailyinventory.xml')
                    #print(project_folder+'/'+strday+'*/dailyinventory.xml')
                except:
                    inv0 = read_inventory(project_folder+'*/dailyinventory.xml') 
                    pass
                inv = inv0.select(network=pick.waveform_id.network_code, station=pick.waveform_id.station_code, time=starttime)
                if not inv:
                    inv = inv0.select(network='*', station=pick.waveform_id.station_code)
                    if not inv:
                        print('Getting response from DMC 1')
                        client = Client()
                        inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel="*",level="response")

            except:
                print('Station metadata error')
                print('Getting response from DMC 2')
                client = Client()
                inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel="*",level="response")
                #starttime = UTCDateTime(origin.time-10)
                #endtime = UTCDateTime(origin.time+10)
                #inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel=tr.stats.channel,level="response")
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
        
    return st3, inv

def select_3comp_include_response(project_folder=None,strday=None,pick=None,starttime=None,endtime=None):
    """
    Finds the appropriate traces and right response file, but makes no attempt to remove, etc.
    """
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
    st3.merge(fill_value='interpolate')
    #print(st3)
    for tr in st3:
        if isinstance(tr.data, np.ma.masked_array):
            tr.data = tr.data.filled()
    
    for tr in st3:
        inventory_local = glob.glob(project_folder+'/'+strday+'*/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.xml') or glob.glob(project_folder+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.xml')
        if len(inventory_local)>0:
            inv = read_inventory(inventory_local[0])
        else:
            try:
                try:
                    inv0 = read_inventory(project_folder+'/'+strday+'*/dailyinventory.xml')
                except:
                    inv0 = read_inventory(project_folder+'*/dailyinventory.xml') 
                    pass
                inv = inv0.select(network=pick.waveform_id.network_code, station=pick.waveform_id.station_code, time=starttime)
                if not inv:
                    inv = inv0.select(network='*', station=pick.waveform_id.station_code)
                    if not inv:
                        print('Getting response from DMC')
                        starttime = UTCDateTime(origin.time-10)
                        endtime = UTCDateTime(origin.time+10)
                        client = Client()
                        inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel="*",level="response")

            except:
                print('Station metadata error')
                print('Getting response from DMC')
                client = Client()
                inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel="*",level="response")
                #starttime = UTCDateTime(origin.time-10)
                #endtime = UTCDateTime(origin.time+10)
                #inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel=tr.stats.channel,level="response")
                pass
                #                    paz = [x for x in pazs if tr.stats.channel in x]
#                    attach_paz(tr, paz[0])
        #inv = client.get_stations(starttime=starttime, endtime=endtime, network="*", sta=tr.stats.station, loc="*", channel=tr.stats.channel,level="response")
        tr.stats.network = inv[0].code
        tr.stats.location = inv[0][0][0].location_code
        #pre_filt = (0.05, 0.06, 30.0, 35.0)
        tr.trim(pick.time-30, pick.time+120)



        
    return st3, inv


def magnitude_quakeml(cat=None, project_folder=None,plot_event=False,eventmode=False, cutoff_dist=200, estimate_sp=False):
    """
    Computes magnitudes for a set of earthquake events and saves them in QuakeML format.
    
    Args:
        cat (obspy.Catalog or None): The catalog of events for which to compute magnitudes. If None,
            a message is printed and the function returns immediately. Default is None.
        project_folder (str or None): The path to the project folder where the data files are stored.
            If None, a message is printed and the function returns immediately. Default is None.
        plot_event (bool): Whether to plot the event waveform data. Default is False.
        eventmode (bool): Whether the given project_folder is a daily or single event folder. If True,
            then project_folder should be the path to the daily folder. Default is False.
        cutoff_dist (float): The maximum distance from the event epicenter to consider for magnitude
            computation, in kilometers. Default is 200.
    
    Returns:
        None: The magnitudes are saved in QuakeML format by writing an additional event magnitude to the Event object.
    """

    print('Computing magnitudes')

    for event in cat:
        origin = event.origins[0]
        print(origin)
        event_lat = origin.latitude
        event_lon = origin.longitude
        # strday = str(origin.time.year).zfill(2)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)
        # if eventmode:
        #     strday = str(project_folder.split('/')[-1])
        # #    strday = str(origin.time.year).zfill(2)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)
        # print(strday)
        # strdaytime = strday+str(origin.time.hour).zfill(2)+str(origin.time.minute).zfill(2)[0]
        mags = []
        mags_iaspei = []

        st2 = Stream()
        for idx1, pick in enumerate(event.picks):
            strday = str(pick.time.year).zfill(2)+str(pick.time.month).zfill(2)+str(pick.time.day).zfill(2)
            #if eventmode:
                #str(project_folder.split('/')[:-2]).join('/')
                #strday = ('/').join(project_folder.split('/')[:-1])
                #project_folder = ('/').join(project_folder.split('/')[:-1])
                #print(strday)
                #strday = strday+str(pick.time.hour).zfill(2)+str(pick.time.minute).zfill(2)[0]
            strdaytime = strday+str(pick.time.hour).zfill(2)+str(pick.time.minute).zfill(2)[0]

            #    strday = str(origin.time.year).zfill(2)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)
            print(strday)
            
            if pick.phase_hint == 'S':
                ### make Amplitude
                st3 = []
                try:
                    starttime_inv=origin.time-10
                    endtime_inv=origin.time+10
                    st3, inv =  select_3comp_remove_response(project_folder,strday,pick,starttime_inv,endtime_inv)

                    tr1 = st3.select(channel='[EHB]HZ')[0]

                    sta_lat = inv[0][0].latitude
                    sta_lon = inv[0][0].longitude
                    epi_dist, az, baz = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
                    epi_dist = epi_dist / 1000
                    tr1.stats.distance = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)[0]
                    tr1.trim(pick.time-20,pick.time+60)
                    st2 += tr1
                    st = st3.select(channel='[EHB]H[EN12]')
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

                        if epi_dist < cutoff_dist:
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
                            stamag.mag = ml_iaspei
                            stamag.station_magnitude_type = 'ML'
                            stamag.amplitude_id = amp.resource_id
                            ## add them to the event
                            event.station_magnitudes.append(stamag)
                        else:
                            print('Station not within '+str(cutoff_dist)+' km of the epicenter - no station magnitude')
                        event.amplitudes.append(amp)
                except Exception:
                    print(traceback.format_exc())#input("push")
                    print('Something went wrong here')
                    pass

        for pick in event.picks:
            if pick.phase_hint == 'P':
                tr1 = st2.select(station=pick.waveform_id.station_code)
                #print(pick.waveform_id.station_code)
                #print(st2)
                try:
                    tr = tr1[0].copy()
                    pol = polarity(tr,pick.time)
                    pick.polarity = pol
                    print(pol)
                    if estimate_sp:
                        #if pol == 'positive' or pol == 'negative':
                        st3, inv = select_3comp_include_response(project_folder,strday,pick,starttime_inv,endtime_inv)
                        #st3 = st.select(station=pick.waveform_id.station_code)
                        sp_ratio1 = sp_ratio(st3,inv,pick,event.picks,event)
                        print(str(sp_ratio1)+' s/p ratio '+pick.waveform_id.station_code)
                        pick.comments.append(Comment(text='sp_ratio:{0}'.format(sp_ratio1)))
                        
                            
                        
                        
                        
                except Exception:
                    print(traceback.format_exc())#input("push")
                    print('Something went wrong during polarity pick')
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
                import matplotlib.pyplot as plt
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
            print(traceback.format_exc())#input("push")

            pass
    if not eventmode:
        cat.write(project_folder+'/cat.xml',format="QUAKEML")
    return cat



def single_event_xml(catalog=None,project_folder=None, format="QUAKEML"):
    """
    Writes earthquake event data to an XML file in the specified format.
    
    Args:
    - catalog: an obspy Catalog object containing earthquake event data
    - project_folder: a string specifying the path to the project folder where the XML file should be written
    - format: a string specifying the format of the XML file (default is 'QUAKEML')
    
    Returns:
    - None
    
    Example:
    >>> cat = obspy.read_events('data.xml')
    >>> single_event_xml(catalog=cat, project_folder='project_folder', format='QUAKEML')
    
    This function creates a folder named after the specified format (e.g. 'quakeml') within the specified project folder.
    For each earthquake event in the catalog, a corresponding XML file is created within the format folder using the 
    event's resource ID as the filename. If the format folder does not exist, it will be created.
    """
    xmlspath = project_folder+'/'+format.lower()
    if not os.path.exists(xmlspath):
        os.makedirs(xmlspath)
    for ev in catalog:
        filename = str(ev.resource_id).split('/')[-1] + ".xml"
        ev.write(xmlspath+'/'+filename, format=format)

def daily_catalog_xml(catalog=None,project_folder='.', format="QUAKEML"):
    xmlspath = project_folder+'/'+format.lower()
    if not os.path.exists(xmlspath):
        os.makedirs(xmlspath)
    catdf = simple_cat_df(catalog)
    date1 = date(int(catdf.origintime.min().strftime("%Y")),int(catdf.origintime.min().strftime("%m")),int(catdf.origintime.min().strftime("%d")))
    date2 = date(int(catdf.origintime.max().strftime("%Y")),int(catdf.origintime.max().strftime("%m")),int(catdf.origintime.max().strftime("%d")))
    for single_date in daterange(date1, date2+timedelta(1)):
        filename = single_date.strftime("%Y").zfill(4)+single_date.strftime("%m").zfill(2)+single_date.strftime("%d").zfill(2) + ".xml"
        cat2 = catalog.filter("time > "+single_date.strftime("%Y-%m-%d")+"T00:00", "time < "+(single_date+timedelta(1)).strftime("%Y-%m-%d")+"T00:00")
        if len(cat2) > 0:
            cat2.write(xmlspath+'/'+filename, format=format)


def join_all_xml(xml_folder=None, filename=None, format="QUAKEML"):
    from obspy import read_events
    xmlfiles = glob.glob(xml_folder+'/*xml')
    cat = Catalog()
    for file in xmlfiles:
        cat0 = read_events(file)
        cat.extend(cat0)
    cat.write(filename+'.xml', format=format)


def fix_picks_catalog(catalog=None, project_folder=None, filename=None):
    """
    Fixes the picks in a catalog by checking actual waveform data
    
    Parameters
    catalog : obspy.core.event.Catalog
    The catalog containing the events and picks.
    project_folder : str
    The path to the project folder.
    filename : str, optional
    The path to the output QUAKEML file.
    
    Returns
    obspy.core.event.Catalog
    A new catalog with the fixed picks.
    
    Notes
    This function assumes that waveform data is stored in MiniSEED files in the
    following directory structure:
    project_folder/YYYYMMDD/network.station.location.channel.mseed
    
    If the waveform data for a pick is not found, the function attempts to fix the
    pick by searching for waveform data for the same station and channel on the
    same day, and updating the channel code accordingly in the QuakeML file.
    
    The function modifies a copy of the input catalog, leaving the original
    catalog unchanged.
    """
    cat2 = catalog.copy()
    for event in cat2:
        print(event.preferred_origin().time)
        for pick in event.picks:
            filethere = glob.glob(project_folder+'/'+pick.time.strftime("%Y%m%d")+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.*.'+pick.waveform_id.channel_code+'*mseed')
            if pick.waveform_id.channel_code[-1] == 'E':
                filethere1 = glob.glob(project_folder+'/'+pick.time.strftime("%Y%m%d")+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.*.'+pick.waveform_id.channel_code[0:2]+'1*mseed')
            if filethere:
                print('it is there')
            else:
                print('change pick')
                st = read(filethere1[0])
                pick.waveform_id.channel_code = st[0].stats.channel
    if filename is not None:
        cat2.write(filename,'QUAKEML')
    return cat2


def cut_event_waveforms(catalog=None, project_folder=None, length=120, filteryes=True, plotevent=False, cutall=False):
    """
    The function cut_event_waveforms takes an earthquake catalog, and cuts and plots the waveform data for each event within a specified time range. The waveform data is read from mseed files located in a specified directory. The function saves the waveform data and plots in a new subdirectory under the project folder. If filteryes is set to True, the waveform data is also filtered.

    Parameters:
    
        catalog : obspy.core.event.Catalog
        An earthquake catalog object containing event and waveform data.
        project_folder : str
        The path of the project folder where the waveform files are located and output should be saved.
        length : int
        The length of the waveform data to be extracted in seconds.
        filteryes : bool
        Whether or not to filter the waveform data. Default is True.
        plotevent : bool
        Whether or not to plot the waveforms for each event. Default is False.
    Returns:
    
        cat2 : obspy.core.event.Catalog
    The same input earthquake catalog object with any changes made to waveform data.
    
    RUN fix_picks first
    TODO association plot - change event name down to second
    
    """
    dirname = project_folder+'/events'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    def nearest(items, pivot):
        return np.where(items == min(items, key=lambda x: abs(x - pivot)))[0]
    for ev in catalog:
        filename = str(ev.resource_id).split('/')[-1] + ".xml"
        ev.write(dirname+'/'+filename, format="QUAKEML")
        origin = ev.preferred_origin() or ev.origins[0]
        print(origin.time)
        strday = str(origin.time.year).zfill(4)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)

        st1 = Stream()
        #arrivals = []
        picks = []
        picktimes = []
        stacheck = set()
        for _i, arrv in enumerate(origin.arrivals):
            pick = arrv.pick_id.get_referred_object()
            try:
                st1 += read(project_folder+'/'+strday+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*'+pick.waveform_id.channel_code+'*mseed') or read(project_folder+'/'+strday+'/*'+pick.waveform_id.station_code+'*'+pick.waveform_id.channel_code+'*SAC')
            except:
                st1 += read(project_folder+'/'+strday+'/'+pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'*'+pick.waveform_id.channel_code[0:2]+'1*mseed') or read(project_folder+'/'+strday+'/*'+pick.waveform_id.station_code+'*'+pick.waveform_id.channel_code+'*SAC')
                pass
            #arrivals.append(arrv)
            stacheck.add(pick.waveform_id.network_code+'.'+pick.waveform_id.station_code+'.'+pick.waveform_id.channel_code)
            picks.append(pick.phase_hint)
            picktimes.append(pick.time)
            #stas.append(pick.waveform_id.station_code)
        if cutall:
            st3 = Stream()
            st2 = read(project_folder+'/'+strday+'/*mseed') or read(project_folder+'/'+strday+'/*SAC')
            for tr2 in st2:
                if (tr2.stats.network+'.'+tr2.stats.station+'.'+tr2.stats.channel) not in stacheck:
                    st3 += tr2
            st3 = st3.slice(origin.time-30, origin.time + length)
            st3.write(dirname+'/'+str(ev.resource_id).split('/')[-1] + "_nopicks.mseed")

        st = st1.slice(origin.time-30, origin.time + length)
        st.write(dirname+'/'+str(ev.resource_id).split('/')[-1] + ".mseed")
        
        os.system('cp '+project_folder+'/'+strday+'/*dailyinventory.xml '+dirname+'/'+str(ev.resource_id).split('/')[-1]+'_inv.xml')
        #os.system(fullpath_python+" "+fullpath1+" -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        
        
        if plotevent:
            try:
                
                if filteryes:
                    st.filter('highpass',freq=1)
                    st = st.slice(origin.time-10, origin.time + length)
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(10,10))
                axes = fig.subplots(len(st), 1, sharex=True)
                lines, labels = ([], [])
                min_x = []
                max_x = []
                for ax, tr, p , pt in zip(axes, st, picks, picktimes):
                        x = np.arange(0, tr.stats.endtime - tr.stats.starttime + tr.stats.delta, tr.stats.delta)
                        y = tr.data
                        x = np.array([(tr.stats.starttime + _x).datetime for _x in x])
                        min_x1, max_x1 = (x[0], x[-1])
                        ax.plot(x, y, 'k', linewidth=1.2)
                        if 'P' in p.upper():
                            pcolor = 'red'
                        #    label = 'P-pick'
                        if 'S' in p.upper():
                            pcolor = 'blue'
                        #    label = 'S-pick'
                        ax.axvline(x=pt.datetime, color=pcolor, linewidth=2,
                                  linestyle='--')
                        line = ax.axvline(x=origin.time.datetime, color='k', linewidth=1,
                                  linestyle='-')
                        #ax.set_ylabel(tr.id, rotation=0, horizontalalignment="right")
                        ax.yaxis.tick_right()
                        #ind1 = nearest(x,pt.datetime)
                        ax.set_ylim([np.max(np.abs(y))*-1.1,np.max(np.abs(y))*1.1])
                        min_x.append(min_x1)
                        max_x.append(max_x1)
                        #labels.append(label)
                        lines.append(line)
                axes[-1].set_xlim([np.min(min_x), np.max(max_x)])
                #axes[-1].set_xlabel("Time")
                plt.subplots_adjust(hspace=0)
                fig.legend(lines, labels)
                fig.savefig(dirname+'/'+str(ev.resource_id).split('/')[-1] + ".png")
                plt.title('M '+str(ev.preferred_magnitude().mag)+' '+str(origin.time))
                plt.close(fig)
            except:
                pass


def detection_association_event(project_folder=None, project_code=None, maxdist = None, maxkm=None, local=True, machine=True, machine_picker=None, fullpath_python=None, approxorigintime=None, downloadwaveforms=True, delta_distance=1, latitude=None, longitude=None, max_radius=None):
    approxotime = UTCDateTime(approxorigintime)
    dirname = str(approxotime.year)+str(approxotime.month).zfill(2)+str(approxotime.day).zfill(2)+str(approxotime.hour).zfill(2)+str(approxotime.minute).zfill(2)+str(approxotime.second).zfill(2)
    #starting = UTCDateTime(single_date.strftime("%Y")+'-'+single_date.strftime("%m")+'-'+single_date.strftime("%d")+'T00:00:00.0') -
    starting = approxotime - 60
    stopping = approxotime + 360
    dir1 = project_folder+'/'+dirname
    print(dir1)
    if machine == True and machine_picker is None:
        machine_picker = 'GPD'
    if machine == False:
        machine_picker = 'STALTA'
    if downloadwaveforms:
        download_mseed_event_radial(dirname=dirname, project_folder=project_folder, starting=starting, stopping = stopping, lat1=latitude, lon1=longitude, maxrad=max_radius)
    #print(single_date.strftime("%Y%m%d"))
    #print(dir1+'/1dassociator_'+project_code+'.db')
    if os.path.exists(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'):
        os.remove(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db')

    db_assoc='sqlite:///'+dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'
    engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
    tables1D.Base.metadata.create_all(engine_assoc)
    Session=sessionmaker(bind=engine_assoc)
    session=Session()

    make3 = True
    infile = make_dayfile(dir1, make3)

    #infile = dir1+'/dayfile.in'
    outfile = dir1+'/gpd_picks.out'
    fileinassociate = outfile

    if local:
        inv = Inventory()
        dir1a = glob.glob(project_folder+'/'+dirname+'/dailyinventory.xml') + glob.glob(project_folder+'/'+dirname+'/??.*.xml')
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
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if fullpath_python:
            os.system(fullpath_python+" "+fullpath1+" -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        else:
            os.system("gpd_predict -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))
        pick_add(dbsession=session,fileinput=outfile,inventory=inv)
    elif machine == True and machine_picker == 'EQTransformer':
        fullpath2 = pathEQT+'/mseed_predictor.py'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if fullpath_python:
            print(fullpath_python+" "+fullpath2+" -I %s -O %s -F %s" % (infile, outfile, pathEQT))
            os.system(fullpath_python+" "+fullpath2+" -I %s -O %s -F %s" % (infile, outfile, pathEQT))
        else:
            os.system("mseed_predictor -I %s -O %s -F %s" % (infile, outfile, pathEQT))
        pick_add(dbsession=session,fileinput=outfile,inventory=inv)
    elif machine == True and machine_picker == 'PhaseNet':
        fullpath3 = pathphasenet+'/phasenet_predict.py'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        if fullpath_python:
            print(pathphasenet)
            #python phasenet/predict.py --model=model/190703-214543 --data_list=test_data/mseed.csv --data_dir=test_data/mseed --format=mseed --plot_figure
            os.system(fullpath_python+" "+fullpath3+" --model=%s/model/190703-214543 --data_list=%s --format=mseed --result_fname=%s --result_dir=%s" % (pathphasenet, infile, outfile, dir1))
        else:
            os.system("phasenet_predict --model=%s/model/190703-214543 --data_list=%s --format=mseed --result_fname=%s --result_dir=%s" % (pathphasenet, infile, outfile, dir1))
        pick_add(dbsession=session,fileinput=outfile,inventory=inv)
    else:
        machine_picker = 'STALTA'
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        queue_sta_lta(infile, outfile, dirname, filtmin, filtmax, t_sta, t_lta, trigger_on, trigger_off, trig_horz, trig_vert)
        pick_add(dbsession=session,fileinput=outfile,inventory=inv)
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
    

    if not os.path.exists(dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'):
        db_assoc='sqlite:///'+dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'
        engine_assoc=create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
        tables1D.Base.metadata.create_all(engine_assoc)
        Session=sessionmaker(bind=engine_assoc)
        session=Session()
        outfile = dir1+'/'+machine_picker.lower()+'_picks.out'
        pick_add(dbsession=session,fileinput=outfile,inventory=inventory)
        session.close()

    db_assoc='sqlite:///'+dir1+'/1dassociator_'+machine_picker.lower()+'_'+project_code+'.db'
    
    assocXX=assoc1D.LocalAssociator(db_assoc, db_tt, max_km = maxkm, aggregation = 1, aggr_norm = 'L2', cutoff_outlier = 10, assoc_ot_uncert = 10, nsta_declare = 3, loc_uncert_thresh = 0.2)
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
    cat, dfs = combine_associated(project_folder=dir1, project_code=project_code, eventmode=True, machine_picker=machine_picker)
    if len(cat)>0:
        cat = magnitude_quakeml(cat=cat, project_folder=dir1, plot_event=False,estimate_sp=True)
    #cat.write('catalog_idaho.xml',format='QUAKEML')
    #single_event_xml(cat,dir1,"QUAKEML")
    for idx1, ev in enumerate(cat):
        filename = dirname+'_'+machine_picker.lower() + "_"+str(idx1)+".xml"
        ev.write(project_folder+'/'+filename, format='QUAKEML')


def simple_cat_df(cat=None, uncertainty=False):
    """
    This function takes in an ObsPy Catalog object containing earthquake events and returns a pandas DataFrame
    containing various attributes of the events.
    
    Parameters:
        cat (obspy.Catalog): Catalog object containing earthquake events.
        uncertainty (bool): Whether to include origin uncertainty parameters in the returned DataFrame. Default is False.
        
    Returns:
        pandas.DataFrame: DataFrame containing the following columns:
        - origintime (datetime): Origin time of the event.
        - latitude (float): Latitude of the event.
        - longitude (float): Longitude of the event.
        - depth (float): Depth of the event.
        - magnitude (float): Magnitude of the event.
        - type (str): Magnitude type of the event.
        - horizontal_error (float): Horizontal uncertainty of the event. Only included if uncertainty is True.
        - vertical_error (float): Vertical uncertainty of the event. Only included if uncertainty is True.
        - num_arrivals (int): Number of arrivals used to locate the event. Only included if uncertainty is True.
        - rms (float): Root-mean-square of the residuals of the event's origin time, latitude, longitude, and depth.
        Only included if uncertainty is True.
        - azimuthal_gap (float): Azimuthal gap of the event. Only included if uncertainty is True.
        - id (str): Resource identifier of the event.
    """
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
            try:
                rms.append(origin1.quality.standard_error)
                az_gap.append(origin1.quality.azimuthal_gap)
                hor_err.append(origin1.origin_uncertainty.horizontal_uncertainty)
                vert_err.append(origin1.depth_errors.uncertainty)
                n_arr.append(len(origin1.arrivals))
            except:
                rms.append(float("NAN"))
                az_gap.append(float("NAN"))
                hor_err.append(float("NAN"))
                vert_err.append(float("NAN"))
                n_arr.append(len(origin1.arrivals))
                pass

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
    """
    This function converts a catalog in QuakeML format into input files for HypoDD. It creates a file containing event and phase information and a file containing station location metadata. If specified, it downloads station metadata from the FDSN web service.
    
    Args:
        cat: obspy catalog object. A catalog of seismic events in QuakeML format.
        download_station_metadata: bool. If True, download station location metadata from the FDSN web service.
        project_folder: str. The path to the project folder where the input files will be saved. If not provided, the current working directory is used.
        project_code: str. The name of the project. The files will be named using this code.
        
    Returns:
        None. The function writes the input files to disk.
        
    Raises:
        No exceptions are raised by this function. Any errors encountered are printed to the console.

    """
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
                if len(pick.waveform_id.station_code) == 5:
                    pick_string = string.format(
                        station_id=pick.waveform_id.station_code,
                        travel_time=travel_time,
                        weight=weight,
                        phase=pick.phase_hint.upper())
                elif len(pick.waveform_id.station_code) == 4:
                    pick_string = string.format(
                        station_id=pick.waveform_id.network_code+'.'+pick.waveform_id.station_code,
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




def plot_hypodd_catalog(file=None,fancy_plot=False):
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

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    # 1. Draw the map background
    #fig = plt.figure(figsize=(8, 8))
    catdfr.iloc[:,1] = pd.to_numeric(catdfr.iloc[:,1].replace("**********",np.nan))
    catdfr.iloc[:,2] = pd.to_numeric(catdfr.iloc[:,2].replace("***********",np.nan))
    lat0 = np.nanmedian(catdfr.iloc[:,1].values)
    lon0 = np.nanmedian(catdfr.iloc[:,2].values)
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
    if fancy_plot:
        m.scatter(catdfr.iloc[:,2].values,catdfr.iloc[:,1].values,s=catdfr.iloc[:,16].values**3*8,c=catdfr.index,marker='o',alpha=0.5,latlon=True)
        cbar = plt.colorbar()
        N_TICKS=8
        indexes = [catdfr['rutc'].iloc[i].strftime('%Y-%m-%d') for i in np.linspace(0,catdfr.shape[0]-1,N_TICKS).astype(int)]
        cbar.ax.set_yticklabels(indexes)
    else:
        m.scatter(catdfr.iloc[:,2].values,catdfr.iloc[:,1].values,s=10,c='k',marker='o',alpha=1,latlon=True)

    plt.savefig('hypoDDmap.png')
    plt.show()


def locate_hyp2000(cat=None, project_folder=None, vel_model=None, fullpath_hyp=None, daymode=False, catalog_year=False, year=None, single_date=None):
    """
    Generate a hypoinverse input file (pha file) for earthquake locations from an obspy Catalog object.
    The function creates a pha file containing P and S picks, and a run.hyp file containing
    necessary parameters to perform location analysis. It requires a velocity model to be specified.
    
    Parameters:
        cat (obspy.Catalog object): Obspy Catalog object containing earthquake events.
        project_folder (str): Path to the project folder.
        vel_model (str): Name of the velocity model to be used. Default is "standard.crh".
        fullpath_hyp (str): Optional path for the hypoinverse compiled program.
    
    Returns:
        None
    """
    if vel_model is None:
        velmodel = pathhyp+'/standard.crh'
        os.system("cp %s %s" % (velmodel,project_folder))
        vel_model = project_folder+'/'+'standard.crh'
    else:
        vel_model = project_folder+'/'+vel_model

    for idx1, event in enumerate(cat):
        origin = event.origins[0] #event.preferred_origin() or event.origins[0]
        stas = []
        picks1a = []
        for _i, arrv in enumerate(origin.arrivals):
            pick = arrv.pick_id.get_referred_object()
            stas.append(pick.waveform_id.station_code)
            if arrv.phase == 'P' or arrv.phase == 'S':
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
        
        if daymode:
            outfile = project_folder+'/'+single_date.strftime("%Y%m%d")+'out.sum'
            phafile = project_folder+'/'+single_date.strftime("%Y%m%d")+'pha'
            runfile = project_folder+'/'+single_date.strftime("%Y%m%d")+'run.hyp'
            prtfile = project_folder+'/'+single_date.strftime("%Y%m%d")+'out.prt'
            stafile = project_folder+'/sta'+single_date.strftime("%Y%m%d")
        elif catalog_year:
            outfile = project_folder+'/'+str(year)+'out.sum'
            phafile = project_folder+'/'+str(year)+'pha'
            runfile = project_folder+'/'+str(year)+'run.hyp'
            stafile = project_folder+'/sta'+str(year)
            prtfile = project_folder+'/'+single_date.strftime("%Y%m%d")+'run.hyp'
        else:    
            outfile = project_folder+'/out.sum'
            phafile = project_folder+'/pha'
            runfile = project_folder+'/run.hyp'
            stafile = project_folder+'/sta'
            prtfile = project_folder+'/out.prt'

        
        if os.path.exists(outfile):
            os.system('rm '+outfile)
        fcur = open(phafile,'w')
        fcur.write(str(hypo71_string))
        fcur.close()

        frun = open(runfile,'w')
        frun.write("crh 1 '"+vel_model+"'")
        frun.write("\n")
        frun.write('h71 3 2 2')
        frun.write("\n")
        #frun.write("sta '"+project_folder+"/sta'")
        frun.write("sta '"+stafile+"'")
        frun.write("\n")
        frun.write("phs '"+phafile+"'")
        frun.write("\n")
        frun.write('pos 1.78')
        frun.write("\n")
        frun.write('dis 2 1000 0.04 0.08')        #DIS 4 50 1 3
        frun.write("\n")
        frun.write('rms 4 .16 1.5 3')        #rms 4 .16 1.5 3
        frun.write("\n")
        frun.write('jun t')
        frun.write("\n")
        frun.write('min 4')
        frun.write("\n")
        frun.write("prt '"+prtfile+"'")
        frun.write("\n")
        frun.write('fil')
        frun.write("\n")
        frun.write("sum '"+outfile+"'")
        frun.write("\n")
        frun.write('loc')
        frun.write("\n")
        frun.write('stop')
        frun.close()
        try:
            if fullpath_hyp:
                os.system("cat %s | %s/hyp2000" % (runfile, fullpath_hyp))
            else:
                os.system("cat %s | hyp2000" % (runfile))
        except:
            pass

        try:
            lines = open(outfile).readlines()
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
                #o.arrivals = origin.arrivals
                lines = open(project_folder+'/out.prt').readlines()
                while True:
                    try:
                        line = lines.pop(0)
                    except:
                        break
                    if line.startswith(" STA NET COM L CR DIST AZM"):
                        break
                for i in range(len(lines)):
                # check which type of phase
                    if lines[i][32] == "P":
                        type1 = "P"
                    elif lines[i][32] == "S":
                        type1 = "S"
                    else:
                        continue
                    # get values from line
                    station = lines[i][0:6].strip()
                    if station == "":
                        station = lines[i-1][0:6].strip()
                        distance = float(lines[i-1][18:23])
                        azimuth = int(lines[i-1][23:26])
                        incident = int(lines[i-1][27:30])
                    else:
                        station = station
                        distance = float(lines[i][18:23])
                        azimuth = int(lines[i][23:26])
                        incident = int(lines[i][27:30])
                    if lines[i][31] == "I":
                        onset = "impulsive"
                    elif lines[i][31] == "E":
                        onset = "emergent"
                    else:
                        onset = None
                    if lines[i][33] == "U":
                        polarity = "positive"
                    elif lines[i][33] == "D":
                        polarity = "negative"
                    else:
                        polarity = None
                    res = float(lines[i][61:66])
                    weight = float(lines[i][68:72])
                    phase_hint = type1
                    for p in event.picks:
                        if station is not None and station != p.waveform_id.station_code[-4:]:
                            continue
                        if phase_hint is not None and phase_hint != p.phase_hint[-4:]:
                            continue
                        pickid = p
                    arrival = Arrival(origin=o, pick=pickid)
                    arrival.time_residual = res
                    arrival.azimuth = azimuth
                    arrival.distance = kilometer2degrees(distance)
                    arrival.takeoff_angle = incident
                    arrival.phase = phase_hint
                    if onset and not pickid.onset:
                        pickid.onset = onset
                    if polarity and not pick.polarity:
                        pickid.polarity = polarity
                    # we use weights 0,1,2,3 but hypo2000 outputs floats...
                    arrival.time_weight = weight
                    arrival.pick_id = pickid.resource_id
                    o.arrivals.append(arrival)
                    #o.quality.used_phase_count += 1

                    #print(type, station, distance, azimuth, incident, res, weight)


            event.origins.append(o)
            event.preferred_origin_id = o.resource_id
        except:
            pass
    return cat


def quakeml_to_growclust(project_folder=None, phase_file='phase.dat', station_file='station.dat', dt_file='dt.cc',cc_threshold = 0.6, download_station_metadata=False):
    #run quakeml_to_hypodd first
    #event list is the 
    with open(project_folder+'/'+phase_file, 'r') as in_f: 
        with open(project_folder+'/evlist.txt', 'w') as out_f: 
            for ln in in_f:
                if ln.startswith('#'): 
                    out_f.write('{}\n'.format(' '.join(ln.split()[1:])))
                    
    stacheck = set()
    with open(project_folder+'/'+dt_file, 'r') as in_f:
        with open(project_folder+'/dtcc.txt' , 'w') as out_f: 
            temp = []
            #if float(ln.split()[-2])>0.6:
            for ln in in_f:
                if ln.startswith('#'):
                    out_f.write('{}'.format(ln))
                if ln[2] == '.':
                    if not ln.startswith('#'):
                        if np.abs(float(ln.split()[1]))<1:
                            if float(ln.split()[-2])>cc_threshold:
                                out_f.write('{}'.format(ln[3:]))
                                stacheck.add(ln[3:].split()[0])
                else:
                    if not ln.startswith('#'):
                        if np.abs(float(ln.split()[1]))<1:
                            if float(ln.split()[-2])>cc_threshold:
                                out_f.write('{}'.format(ln))
                                stacheck.add(ln.split()[0])
    
    #temp = pd.read_csv('stlist.txt',delimiter=r"\s+",header=None)                 
    #temp = temp.drop_duplicates(subset=[0], keep="first")    
    #temp.to_csv('test.txt', index=False, sep=' ')            
    with open(project_folder+'/'+station_file, 'r') as in_f: 
        with open(project_folder+'/stlist.txt' , 'w') as out_f: 
            for ln in in_f:
                #print(ln[2])
                if ln[2] == '.':
                    if ln[3:].split()[0] in stacheck:
                        out_f.write('{}'.format(ln[3:]))
                else:
                    if ln.split()[0] in stacheck:
                        out_f.write('{}'.format(ln))
                    

    

    if download_station_metadata:
        temp = pd.read_csv(project_folder+'/evlist.txt',delimiter=r"\s+",header=None)
        times = []
        for i in temp.index:
            times.append(UTCDateTime(temp.iloc[i,0],temp.iloc[i,1],temp.iloc[i,2],temp.iloc[i,3],temp.iloc[i,4],temp.iloc[i,5]))
        tmin = np.array(times).min() 
        tmax = np.array(times).max() 
        latmed = temp[6].median()
        lonmed = temp[7].median()
        station_dat_file = project_folder+'/stlist.txt'
        stalist=set()
        with open(project_folder+'/'+dt_file, 'r') as in_f: 
            for ln in in_f:
                #print(ln[2])
                if not ln.startswith('#'): 
                    stalist.add(ln.split(' ')[0])
        print('Downloading station location metadata')
        client = Client()
        station_strings = []
        for sta in stalist:
            print(sta)
            try:
                net, sta1 = sta.split('.')
            except:
                sta1 = sta
                net = '*'
                pass
            #sta1 = sta
            try:
                inva = client.get_stations(starttime=tmin, endtime=tmax, network=net, station=sta1, level='station')
            except:
                inva = None
                pass
            if inva is not None:
                dists = []
                netnames = []
                for net in inva:
                    for sta in net:
                        dists.append(gps2dist_azimuth(sta.latitude, sta.longitude, latmed, lonmed)[0])
                        netnames.append(net.code)
                rightnet = inva.select(network=netnames[np.where(dists==np.min(dists))[0][0]])
                station_lat = rightnet[0][np.where(dists==np.min(dists))[0][0]].latitude
                station_lon = rightnet[0][np.where(dists==np.min(dists))[0][0]].longitude
                station_elev = rightnet[0][np.where(dists==np.min(dists))[0][0]].elevation
                station_strings.append("%s %.6f %.6f %i" % (sta1,  station_lat,  station_lon,  station_elev))
            #inva1[0][0].latitude
        station_string = "\n".join(set(station_strings))
        with open(station_dat_file, "w") as open_file:
            open_file.write(station_string)
    # with open(project_folder+'/'+dt_file, 'r') as in_f: 
    #     with open(project_folder+'/xcordata.txt' , 'w') as out_f: 
    #         for ln in in_f: 
    #             if ln[2] == '.':
    #                 if len(ln[3:].split(' ')[0]) == 4:
    #                     out_f.write('{}{}'.format('   ',ln[3:]))
    #                 elif len(ln[3:].split(' ')[0]) == 5:
    #                     out_f.write('{}{}'.format('  ',ln[3:]))
    #             else:
    #                 if ln[0] != '#':
    #                     out_f.write('{}{}'.format('   ',ln))
    #                 else:
    #                     out_f.write(ln) 


    
    

def reduce_catalog(cat=None, num_arr=None, vert_unc=None):
    events = list(cat.events)
    temp_events = []
    for event in events:
        #print(event)
        if num_arr:
            if len(event.preferred_origin().arrivals) >= num_arr:
                temp_events.append(event)
        if vert_unc:
            if event.preferred_origin().depth_errors.uncertainty is not None:
                if event.preferred_origin().depth_errors.uncertainty < vert_unc:
                    temp_events.append(event)

    events = temp_events
    from obspy import Catalog
    cat2 = Catalog(events=events)
    return cat2

def duplicate_remove(cat=None,seconds=5):
    """
    Look for possible duplicate events by identifying earthquakes within X seconds of one another
    
    Parameters:
        cat (obspy.Catalog object): Obspy Catalog object containing earthquake events.
        seconds (float): Inter-event minimum time
    
    Returns:
        Catalog that is likely smaller
    """
    time_seconds = seconds
    times = [e.preferred_origin().time.datetime for e in list(cat.events)]
    li = []
    for i in range(len(times)):
          li.append([times[i],i])
    li.sort()
    sort_index = []
    #sorted((d for d in times))
    for x in li:
          sort_index.append(x[1])

    events = list(cat.events)
    temp_events = []
    for idx1 in range(len(sort_index)):
        #print(event)
        event = events[sort_index[idx1]]
        #print(sort_index[idx1])
        temp_events.append(event)
    events2 = temp_events
    cat2 = Catalog(events=events2)
    catdf2 = simple_cat_df(cat2, True)

    catdf2['delta'] = (catdf2['origintime']-catdf2['origintime'].shift()).fillna(pd.Timedelta(seconds=100))
    notdups = catdf2[catdf2.delta > pd.Timedelta(seconds=time_seconds)].index
    events3 = list(cat2.events)
    cat3 = Catalog(events=[events3[i] for i in notdups])
    return cat3



def plot_map_catalog(cat=None, filename=None, points=False):
#    catdfr = pd.read_csv(file,delimiter=r"\s+")
#    catdfr = catdfr.dropna()
    import matplotlib.pyplot as plt
    catdfr = simple_cat_df(cat)

    #catdfr = catdfr.reset_index(drop=True)
    #rutc = np.zeros((len(catdfr.index),1))

    from mpl_toolkits.basemap import Basemap
    # 1. Draw the map background
    #fig = plt.figure(figsize=(8, 8))
    #plt.figure()
    lat0 = np.median(catdfr.iloc[:,1].values)
    lon0 = np.median(catdfr.iloc[:,2].values)
    m = Basemap(projection='lcc', resolution='h',
                lat_0=lat0, lon_0=lon0,
                width=1E6, height=.6E6)
    #m.shadedrelief()
    try:
        m.drawcoastlines(color='gray')
    except:
        pass
    m.drawcountries(color='gray')
    #m.drawcounties(color='gray')
    m.drawstates(color='gray')

    # 2. scatter city data, with color reflecting population
    # and size reflecting area
    catdfr['magnitude'] = catdfr['magnitude'].fillna(1)
    if points:
        m.scatter(catdfr.iloc[:,2].values,catdfr.iloc[:,1].values,s=20,c='k',marker='.',alpha=0.5,latlon=True)
    else:
        m.scatter(catdfr.iloc[:,2].values,catdfr.iloc[:,1].values,s=catdfr.iloc[:,4].values**3*8,c=catdfr.index,marker='o',alpha=0.5,latlon=True)
        cbar = plt.colorbar()
        N_TICKS=8
        indexes = [catdfr['origintime'].loc[i].strftime('%Y-%m-%d') for i in np.linspace(0,catdfr.shape[0]-1,N_TICKS).astype(int)]

        cbar.ax.set_yticklabels(indexes)
    plt.show()
    if filename:
        plt.savefig(filename+'.png')
    else:
        plt.savefig('hypo_map.png')


def plot_gr_freq_catalog(cat=None,min_mag=2):
    import matplotlib.pyplot as plt
    catdf = simple_cat_df(cat)

    #catdf['origintime'] = pd.to_datetime(catdf.index)

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

def make_station_list_csv(project_folder=None):
    files = sorted(glob.glob(project_folder+'/*/dailyinventory.xml'))
    for file in files:
        dir1 = project_folder+'/'+file.split('/')[-2]
        if not os.path.exists(dir1+'/station_list.csv'):
            inventory = read_inventory(dir1+'/dailyinventory.xml')
            stalat = []
            stalon = []
            staname = []
            netname = []
            elevs = []
            for net in inventory:
                for sta in net:
                    stalat.append(sta.latitude)
                    stalon.append(sta.longitude)
                    staname.append(sta.code)
                    netname.append(net.code)
                    elevs.append(sta.elevation)
            stadf = pd.DataFrame({'net':netname,'sta':staname,'netsta':[a+'.'+b for a,b in zip(netname,staname)],'latitude':stalat,'longitude':stalon,'elevation (m)':elevs})
            stadf = stadf.drop_duplicates()
            stadf.to_csv(dir1+'/station_list.csv',index=False)


def daymode_catalog(project_folder=None,project_code=None,single_date=None, machine_picker=None,fullpath_hyp=None):
    cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code,daymode=True,single_date=single_date,machine_picker=machine_picker)
    cat = magnitude_quakeml(cat=cat, project_folder=project_folder,plot_event=False)
    cat = locate_hyp2000(cat=cat, project_folder=project_folder,fullpath_hyp=fullpath_hyp,daymode=True,single_date=single_date)
    cat.write(project_folder+'/catalog_'+project_code+'_hyp_'+machine_picker.lower()+'_'+single_date.strftime("%Y%m%d")+'.xml',format='QUAKEML')
    cat2 = simple_cat_df(cat,True)
    cat2.to_csv(project_folder+'/catalog_'+project_code+'_hyp_'+machine_picker.lower()+'_'+single_date.strftime("%Y%m%d")+'.csv')

def yearmode_catalog(project_folder=None,project_code=None, year=None, machine_picker=None,fullpath_hyp=None):
    cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code, catalog_year=True, year=year, machine_picker=machine_picker)
    cat = magnitude_quakeml(cat=cat, project_folder=project_folder,plot_event=False)
    cat = locate_hyp2000(cat=cat, project_folder=project_folder,fullpath_hyp=fullpath_hyp, catalog_year=True, year=year)
    cat.write(project_folder+'/catalog_'+project_code+'_hyp_'+machine_picker.lower()+'_'+str(year)+'.xml',format='QUAKEML')
    cat2 = simple_cat_df(cat,True)
    cat2.to_csv(project_folder+'/catalog_'+project_code+'_hyp_'+machine_picker.lower()+'_'+str(year)+'.csv')


def quakeml_to_hdf5(cat=None, project_folder=None, makecsv=True):
    """
    Convert seismic waveform data associated with each earthquake event in the `cat` object into HDF5 format. If `makecsv` is set to `True`, a CSV file of the event information will be created in the `project_folder` directory.

    Parameters:
        cat (obspy.core.event.Catalog): An ObsPy Catalog object containing earthquake event information.
        project_folder (str): The path of the directory where the output HDF5 and CSV files will be saved.
        makecsv (bool): If True, a CSV file of the event information will be created in the `project_folder` directory.

    Returns:
        None
    """

    #make a training dataset in STEAD format for retraining data
    output_merge = project_folder+'/merge.hdf5'
    if os.path.exists(project_folder+'/merge.hdf5'):
        os.remove(project_folder+'/merge.hdf5')
    HDF0 = h5py.File(output_merge, 'a')
    HDF0.create_group("data")
    def samplename(stream):
        st = str(stream[0].stats.starttime)
        st_name = st[0:13]+st[14:16]+st[17:19]
        stend = str(stream[0].stats.endtime)
        st_nameend = st[0:13]+st[14:16]+st[17:19]
        filename = stream[0].stats.network + '.' + stream[0].stats.station
        for tr in stream:
            filename = filename + '.' + tr.stats.channel
        filename = filename + '.' + st_name + st_nameend
        return filename

    for event in cat:
        #print(t0)
        origin = event.preferred_origin()
        evo = origin.time
        #evid = origin.id
        otime = UTCDateTime(evo)
        print(otime)

        origin = event.preferred_origin() or cat.origins[0]
        try:
            magnitude = event.preferred_magnitude().mag
            magtype = event.preferred_magnitude().magnitude_type
        except:
            magnitude = -2
            magtype = 'ML'
        contdir =  project_folder+'/'+str(origin.time.year)+str(origin.time.month).zfill(2)+str(origin.time.day).zfill(2)

        stas = []
        picks1a = []
        for _i, arrv in enumerate(origin.arrivals):
            pick = arrv.pick_id.get_referred_object()
            stas.append(pick.waveform_id.station_code)
            picks1a.append(pick.waveform_id.network_code+' '+pick.waveform_id.station_code + ' '+pick.phase_hint+' '+str(pick.time)+' '+pick.waveform_id.channel_code)
        stalist = list(set(stas))

        for states in stalist:
            numP = -9
            numS = -9
            #print(states)
            for num, line in enumerate(picks1a):
                if states in line and 'P' in line:
                    numP = num
                if states in line and 'S' in line:
                    numS = num
            #print(stalistall)
            if numP > -1 and numS > -1:
                pickS = picks1a[numS]
                pickP = picks1a[numP]
                sacyes = None
                try:
                    st3 = read(contdir+'/'+pickP.split(' ')[0]+'.'+pickP.split(' ')[1]+'*'+pickP.split(' ')[-1][0:2]+'*mseed',debug_headers=True)
                except:
                    try:
                        st3 = read(contdir+'/*'+pickP.split(' ')[1]+'.*SAC',debug_headers=True)
                        sacyes = 1
                    except:
                        pass
                if sacyes is None:
                    for tr in st3:
                        inventory_local = glob.glob(contdir+'/'+pickP.split(' ')[0]+'.'+pickP.split(' ')[1]+'.xml')
                        if len(inventory_local)>0:
                            inv = read_inventory(inventory_local[0])
                        else:
                            try:
                                inv0 = read_inventory(contdir+'/'+'dailyinventory.xml')
                                inv = inv0.select(network=pickP.split(' ')[0], station=pickP.split(' ')[1], time=origin.time)
                                if len(inv) ==0:
                                    try:
                                        inv = read_inventory(contdir+'/'+'dailyinventory.xml')
                                    except:
                                        pass
                            except:
                                pass

                if sacyes is None:
                    stationlat = inv.networks[0].stations[0].latitude
                    stationlon = inv.networks[0].stations[0].longitude
                    stationelev = inv.networks[0].stations[0].elevation
                    epi_dist1, az1, baz1 = gps2dist_azimuth(inv.networks[0].stations[0].latitude, inv.networks[0].stations[0].longitude, origin.latitude, origin.longitude)
                else:
                    try:
                        inv0 = read_inventory(contdir+'/'+'dailyinventory.xml')
                        inv = inv0.select(network=pickP.split(' ')[0], station=pickP.split(' ')[1], time=origin.time)
                    except:
                        stationlat = st3[0].stats.sac.stla
                        stationlon = st3[0].stats.sac.stlo
                        stationelev = st3[0].stats.sac.stel
                        epi_dist1, az1, baz1 = gps2dist_azimuth(st3[0].stats.sac.stla, st3[0].stats.sac.stlo, origin.latitude, origin.longitude)
                        pass

                dist1 = epi_dist1/1000
                p_arriv_time = UTCDateTime(pickP.split(' ')[-2])-origin.time
                s_arriv_time = UTCDateTime(pickS.split(' ')[-2])-origin.time

                st_1a = st3
                if len(st_1a)>1:
                    st_1 = st_1a.select(location='')
                    if len(st_1)<1:
                        st_1 = st_1a
                else:
                    st_1 = st_1a
                st_1.trim(UTCDateTime(pickP.split(' ')[-2])-60,UTCDateTime(pickP.split(' ')[-2])+60)
                if int(st_1[0].stats.sampling_rate) != 100:
                    st_1.resample(100.0)
                filename = samplename(st_1)
                data = np.array(st_1)
                data = data.T
                HDFr = h5py.File(output_merge, 'a')
                dsF = HDFr.create_dataset("data/"+filename, data.shape, data=data, dtype=np.float64)
                dsF.attrs['network_code'] = st_1[0].stats.network
                dsF.attrs['receiver_code'] = st_1[0].stats.station
                dsF.attrs['receiver_type'] = st_1[0].stats.channel
                dsF.attrs['receiver_latitude'] = stationlat
                dsF.attrs['receiver_longitude'] = stationlon
                dsF.attrs['receiver_elevation_m'] = stationelev
                dsF.attrs['p_arrival_sample'] = int(100*(UTCDateTime(pickP.split(' ')[-2])-st_1[0].stats.starttime))
                dsF.attrs['p_status'] = 'manual'
                dsF.attrs['p_weight'] = 1
                dsF.attrs['p_travel_sec'] = p_arriv_time
                dsF.attrs['s_arrival_sample'] = int(100*(UTCDateTime(pickS.split(' ')[-2])-st_1[0].stats.starttime))
                dsF.attrs['s_status'] = 'manual'
                dsF.attrs['s_weight'] = 1
                dsF.attrs['source_id'] = 'evid'
                dsF.attrs['source_origin_time'] = str(origin.time)
                dsF.attrs['source_origin_uncertainty_sec'] = []
                dsF.attrs['source_latitude'] = origin.latitude
                dsF.attrs['source_longitude'] = origin.longitude
                dsF.attrs['source_error_sec'] = []
                dsF.attrs['source_gap_deg'] = []
                try:
                    dsF.attrs['source_horizontal_uncertainty_km'] = np.sqrt((origin.origin_uncertainty["max_horizontal_uncertainty"])**2 + (origin.origin_uncertainty["min_horizontal_uncertainty"])**2)/1000
                except:
                    dsF.attrs['source_horizontal_uncertainty_km'] = []
                    pass
                dsF.attrs['source_depth_km'] = origin.depth/1000
                try:
                    dsF.attrs['source_depth_uncertainty_km'] = origin.depth_errors.uncertainty/1000
                except:
                    dsF.attrs['source_depth_uncertainty_km'] = []
                dsF.attrs['source_magnitude'] = magnitude
                dsF.attrs['source_magnitude_type'] = magtype
                dsF.attrs['source_magnitude_author'] = 'ogs'
                dsF.attrs['source_mechanism_strike_dip_rake'] = []
                dsF.attrs['source_distance_deg'] = kilometers2degrees(epi_dist1/1000)
                dsF.attrs['source_distance_km'] = epi_dist1/1000
                dsF.attrs['back_azimuth_deg'] = az1
                dsF.attrs['snr_db'] = []
                dsF.attrs['coda_end_sample'] = []
                dsF.attrs['trace_start_time'] = str(st_1[0].stats.starttime)
                dsF.attrs['trace_category'] = 'earthquake_local'
                dsF.attrs['trace_name'] =   filename
                HDFr.flush()
                HDFr.close()


    if makecsv:
        import csv
        output_merge = 'merge.hdf5'

        csvfile = open(output_merge.split('.')[0]+'.csv', 'w')
        output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['network_code','receiver_code','receiver_type','receiver_latitude','receiver_longitude',
                                      'receiver_elevation_m','p_arrival_sample','p_status','p_weight','p_travel_sec',
                                      's_arrival_sample','s_status','s_weight',
                                      'source_id','source_origin_time','source_origin_uncertainty_sec',
                                      'source_latitude','source_longitude','source_error_sec',
                                      'source_gap_deg','source_horizontal_uncertainty_km', 'source_depth_km', 'source_depth_uncertainty_km',
                                      'source_magnitude', 'source_magnitude_type', 'source_magnitude_author','source_mechanism_strike_dip_rake',
                                      'source_distance_deg', 'source_distance_km', 'back_azimuth_deg', 'snr_db', 'coda_end_sample',
                                      'trace_start_time', 'trace_category', 'trace_name'])

        dtfl = h5py.File(output_merge, 'r')

        for evi in dtfl['data']:
            x = dtfl.get('data/'+str(evi))
            try:
                output_writer.writerow([x.attrs['network_code'],
                                        x.attrs['receiver_code'],
                                        x.attrs['receiver_type'],
                                        x.attrs['receiver_latitude'],
                                        x.attrs['receiver_longitude'],
                                        x.attrs['receiver_elevation_m'],
                                        x.attrs['p_arrival_sample'],
                                        x.attrs['p_status'],
                                        x.attrs['p_weight'],
                                        x.attrs['p_travel_sec'],
                                        x.attrs['s_arrival_sample'],
                                        x.attrs['s_status'],
                                        x.attrs['s_weight'],
                                        x.attrs['source_id'],
                                        x.attrs['source_origin_time'],
                                        x.attrs['source_origin_uncertainty_sec'],
                                        x.attrs['source_latitude'],
                                        x.attrs['source_longitude'],
                                        x.attrs['source_error_sec'],
                                        x.attrs['source_gap_deg'],
                                        x.attrs['source_horizontal_uncertainty_km'],
                                        x.attrs['source_depth_km'],
                                        x.attrs['source_depth_uncertainty_km'],
                                        x.attrs['source_magnitude'],
                                        x.attrs['source_magnitude_type'],
                                        x.attrs['source_magnitude_author'],
                                        x.attrs['source_mechanism_strike_dip_rake'],
                                        x.attrs['source_distance_deg'],
                                        x.attrs['source_distance_km'],
                                        x.attrs['back_azimuth_deg'],
                                        x.attrs['snr_db'],
                                        x.attrs['coda_end_sample'],
                                        x.attrs['trace_start_time'],
                                        x.attrs['trace_category'],
                                        x.attrs['trace_name']]);
                csvfile.flush()
            except:
                pass
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
