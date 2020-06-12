#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:52:45 2020

@author: jwalter
"""

from easyQuake import association_continuous
from easyQuake import daterange
from datetime import date


start_date = date(2019, 11, 1)
end_date = date(2020, 1, 1)
#end_date = date(2018, 1, 2)
project_code = 'okla'
project_folder = '/data/okla/ContWaveform'

maxdist = 200
maxkm = 200

from multiprocessing import Pool
pool = Pool(20)
for single_date in daterange(start_date, end_date):
    print(single_date.strftime("%Y-%m-%d"))
    dirname = single_date.strftime("%Y%m%d")
    #download_mseed_event_radial(dirname=dirname, project_folder='/data/antarctica/aeluma', starting=start, stopping=end, lat1=event.lat, lon1=event.lon, maxrad=4)
    lat1 = 35.5
    lon1 = -97.5
    #detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, machine=True,local=True,single_date=single_date)
    pool.apply_async(association_continuous, (dirname, project_folder, project_code, maxdist, maxkm, single_date, False))
pool.close()
pool.join()
    
    #association_continuous(dirname, project_folder, project_code, maxdist, maxkm, single_date, False)
    #association_continuous(dirname, project_folder, project_code, maxdist, maxkm, single_date, False)
    #download_mseed(dirname=dirname, project_folder=project_folder, single_date=single_date, minlat=lat_a, maxlat=lat_b, minlon=lon_a, maxlon=lon_b)
    #detection_assocation_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, maxdist=250, maxkm=250, single_date=single_date, local=False, latitude=lat1, longitude=lon1, max_radius=4)

##magnitude_quakeml(project_folder=project_folder,start=start_date,end=start_date)
#cat, dfs = combine_associated_hyp71files(project_folder=project_folder, project_code=project_code)
##magnitude_quakeml(cat=cat, project_folder=project_folder)
#cat2 = magnitude_quakeml(cat=cat, project_folder=project_folder,plot_event=True)
#cat2.write('catalog_okla.xml', format='QUAKEML')
#
#
#catdf = simple_cat_df(cat)
#plt.plot(catdf.index,catdf.mag,'.')
#plt.savefig('temp.png')
#
#
#catogs = pd.read_csv('https://ogsweb.ou.edu/api/earthquake?start=202001010000&end=202001022359&mag=0&format=csv')
#catogs.index = pd.DatetimeIndex(catogs['origintime'])
#plt.plot(catogs.index,catogs.magnitude.values,'.')