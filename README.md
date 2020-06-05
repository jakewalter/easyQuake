# easyQuake

Simplified machine-learning driven earthquake detection, location, and analysis in one easy-to-implement python package.

If you find this useful, please cite:

```
Walter et al. (in prep) easyQuake: Putting machine learning to work for your regional seismic network or local earthquake study
```

## Requirements
This code leverages machine-learning for earthquake detection. You should have suitable hardward to run CUDA/Tensorflow, which usually means some sort of GPU. This has been tested on servers with nVidia compute cards, but also runs well on a modest multi-core with gaming PC nVidia card. The event-mode can be run efficiently enough on a laptop.

## Running easyQuake
The example runs easyQuake for a recent M6.5 earthquake in Idaho

```
from easyQuake import download_mseed
from easyQuake import daterange
from datetime import date
from easyQuake import combine_associated
from easyQuake import detection_continuous
from easyQuake import association_continuous

from easyQuake import magnitude_quakeml
from easyQuake import simple_cat_df

import matplotlib.pyplot as plt
maxkm = 300
maxdist=300
lat_a = 42
lat_b = 47.5
lon_a = -118
lon_b = -111


start_date = date(2020, 3, 31)
end_date = date(2020, 4, 1)

project_code = 'idaho'
project_folder = '/data/id'
for single_date in daterange(start_date, end_date):
    print(single_date.strftime("%Y-%m-%d"))
    dirname = single_date.strftime("%Y%m%d")
    download_mseed(dirname=dirname, project_folder=project_folder, single_date=single_date, minlat=lat_a, maxlat=lat_b, minlon=lon_a, maxlon=lon_b)
    detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, single_date=single_date, machine=True,local=True)
    association_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, maxdist=maxdist, maxkm=maxkm, single_date=single_date, local=True)

cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code)
cat = magnitude_quakeml(cat=cat, project_folder=project_folder,plot_event=True)
cat.write('catalog_idaho.xml',format='QUAKEML')


catdf = simple_cat_df(cat)
plt.figure()
plt.plot(catdf.index,catdf.magnitude,'.')
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* code was used from obspyck, gpd, hashpy, and others
* would not be possible without obspy
* this work was developed at the Oklahoma Geological Survey

