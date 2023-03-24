#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:32:57 2023

@author: jwalter
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapefile as shp  # Requires the pyshp package
import pandas as pd
import matplotlib.path as mpltPath
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature

catdf = pd.read_csv('https://jakewalter.mynetgear.com/data/catalog_okla_hyp_all.csv')
catdf = catdf2
catdf = catdf[catdf['vertical_error']<20000]
catdf = catdf.reset_index(drop=True)

map = Basemap(llcrnrlon=-104,llcrnrlat=33,urcrnrlon=-93,urcrnrlat=38, resolution='l', projection='tmerc',lat_0=35,lon_0=-98)
x,y = map(catdf2.iloc[:,4],catdf2.iloc[:,3])
X=np.transpose(np.array((x,y)))
depth = -np.array(catdf2.iloc[:,5])*1000




                    
                    
                    
#### earthquake
catdf = pd.read_csv('http://wichita.ogs.ou.edu/eq/catalog/complete/complete.csv')
eqtempin = catdf[['latitude','longitude']].to_numpy()
eqpath = mpltPath.Path(polygon)
insidestate = eqpath.contains_points(eqtempin)
catdf = catdf[insidestate]
catdf = catdf.reset_index(drop=True)
catdf['origintime'] = pd.to_datetime(catdf['origintime'])
catdf['magnitude'] = catdf['magnitude'].str.replace("None", "0").astype(float)
catdf['magnitude'] = catdf['magnitude'].astype(float)
catdf = catdf[(catdf['origintime']>time1) & (catdf['origintime']<time2)]
catdf.sort_values(by=['origintime'], inplace=True)
catdf = catdf.reset_index(drop=True)

lat_a = 36.36
lat_b = 36.49
lon_a = -97.01
lon_b = -96.80

catdf = catdf[(catdf['latitude']>lat_a) & (catdf['latitude']<lat_b) & (catdf['longitude']>lon_a) & (catdf['longitude']<lon_b)]

from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt



fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})                 # make a new plotting range

ax.set_extent(extent)
ax.gridlines()
#ax.coastlines()
ax.set_xticks(np.linspace(extent[0],extent[1],14),crs=ccrs.PlateCarree()) # set longitude indicators
ax.set_yticks(np.linspace(extent[2],extent[3],14)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
lon_formatter = LongitudeFormatter(number_format='0.1f',degree_symbol='',dateline_direction_label=True) # format lons
lat_formatter = LatitudeFormatter(number_format='0.1f',degree_symbol='') # format lats
ax.xaxis.set_major_formatter(lon_formatter) # set lons
ax.yaxis.set_major_formatter(lat_formatter) # set lats
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
#ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.BORDERS.with_scale('50m'))
ax.add_feature(cfeature.STATES.with_scale('50m'))
ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')


#ax.plot(allwells['Longitude'], allwells['Latitude'], markersize=1,marker='s',linestyle='',color='g',transform=ccrs.PlateCarree())
#ax.plot(catdfml['longitude'], catdfml['latitude'], markersize=1.5,marker='o',linestyle='',color='r',transform=ccrs.PlateCarree())
ax.plot(catdf['longitude'], catdf['latitude'], markersize=1,marker='o',linestyle='',color='k',transform=ccrs.PlateCarree())
ax.plot(catdfmonth['longitude'], catdfmonth['latitude'], markersize=5,marker='o',linestyle='',color='yellow',transform=ccrs.PlateCarree())
ax.plot(catdfweek['longitude'], catdfweek['latitude'], markersize=5,marker='o',linestyle='',color='orange',transform=ccrs.PlateCarree())
ax.plot(catdfday['longitude'], catdfday['latitude'], markersize=5,marker='o',linestyle='',color='red',transform=ccrs.PlateCarree())
