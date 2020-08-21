#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:52:45 2020

@author: jwalter
"""
from easyQuake import detection_continuous
from easyQuake import association_continuous
from datetime import date
from easyQuake import combine_associated
from easyQuake import magnitude_quakeml
from easyQuake import simple_cat_df
import os

from datetime import timedelta, date, datetime

os.system('wget http://wichita.ogs.ou.edu/eq/stream/temp_3comp.mseed -P /scratch/scratch')
os.system('/home/jwalter/bin/mssplit -D /scratch/scratch -T %N.%S.%C.%L.D.%Y.%j.mseed /scratch/scratch/temp_3comp.mseed')
os.system('rm /scratch/scratch/*PW17*')
project_folder = '/scratch'
project_code = 'okla'

dirname = 'scratch'
single_date = date.today()
#Use the line just below if system time is on UTC time
#single_date = date.today()-timedelta(1)
lat1 = 35.5
lon1 = -97.5
#need to set lat/lon because we dont download metadata with the mseed waveforms

detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, machine=True, local=False, single_date=single_date, latitude=lat1, longitude=lon1, max_radius=4)
    
association_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, maxdist=300, maxkm=300, single_date=single_date, local=False, latitude=lat1, longitude=lon1, max_radius=4)

cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code)
cat = magnitude_quakeml(cat=cat, project_folder=project_folder,plot_event=False)
cat.write('/scratch/scratch/cat_okla_scratch'+single_date.strftime("%Y%m%d")+'.xml',format='QUAKEML')

catdf = simple_cat_df(cat)
catdf.to_csv('/scratch/scratch/cat_okla_scratch'+single_date.strftime("%Y%m%d")+'.csv')

os.system('rm /scratch/scratch/*.mseed')

#export to SC3ML file rather than QuakeML
from easyQuake import single_event_xml
format = "SC3ML"
os.system('rm -r /scratch/scratch/'+format.lower())

single_event_xml(cat,'/scratch/scratch',format)


import shutil
shutil.make_archive('sc3ml', 'zip', '/scratch/scratch/sc3ml')



#put it somewhere else (somewhere SC3 processing server can wget it)
import paramiko
from scp import SCPClient
def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client
ssh = createSSHClient('domain', 'port', 'user', 'password')
scp = SCPClient(ssh.get_transport())
scp.put('sc3ml.zip', recursive=True, remote_path='fullpath')
scp.close()


#in SC3 system, run the following to drop in origins and it will form an event
#!/usr/bin/bash
#FILES=/home/sysop/ml/*xml
#for f in $FILES
#do
#scdispatch -i $f -O add --routingtable Pick:PICK,Amplitude:AMPLITUDE,Origin:LOCATION,StationMagnitude:MAGNITUDE,Magnitude:MAGNITUDE
#done


