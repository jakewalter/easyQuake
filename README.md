# easyQuake

Simplified machine-learning driven earthquake detection, location, and analysis in one easy-to-implement python package.

On most systems you should be able to simply:
```
pip install easyQuake
```
To stay on the bleeding edge of updates:
```
pip install easyQuake --upgrade
```

Or if you need to tweak something, like the number of GPUs in gpd_predict, you could:
```
git clone https://github.com/jakewalter/easyQuake.git
cd easyQuake
pip install .
```

If you find this useful, please cite:

```
Walter, J. I., P. Ogwari, A. Thiel, F. Ferrer, and I. Woelfel (2021), easyQuake: Putting machine 
learning to work for your regional seismic network or local earthquake study, Seismological Research 
Letters, 92(1): 555–563, https://doi.org/10.1785/0220200226
```

## Requirements
This code leverages machine-learning for earthquake detection with the choice of either the GPD (https://github.com/interseismic/generalized-phase-detection) or EQTransformer (https://github.com/smousavi05/EQTransformer) pickers. You should have suitable hardware to run CUDA/Tensorflow, which usually means some sort of GPU. This has been tested on servers with nvidia compute cards and modest multi-core desktop with consumer gaming nvidia card (e.g. Geforce 1050 Ti). The event-mode can be run efficiently enough on a laptop.

* Requires nvidia-cuda-toolkit, obspy, keras==2.3.1, tensorflow-gpu==2.1 (if using multiple GPUs only tensorflow 1.15 is tested), basemap
* I've found that the the easiest way to install cuda, tensorflow, and keras is through installing Anaconda python and running ```conda install tensorflow-gpu==2.1```
* Because tensorflow-gpu 2.1 requires python 3.7 (not the latest version), you might find an easier road creating a new environment:
```
conda create -n easyquake python=3.7 anaconda
conda activate easyquake
conda install tensorflow-gpu==2.1
conda install keras
conda install obspy -c conda-forge
pip install easyQuake
```

## Running easyQuake

The first example is a simple one in "event mode" - try it:

```
from easyQuake import detection_association_event

detection_association_event(project_folder='/scratch', project_code='ok', maxdist = 300, maxkm=300, local=True, machine=True, latitude=36.7, longitude=-98.4, max_radius=3, approxorigintime='2021-01-27T14:03:46', downloadwaveforms=True)
```

This next example runs easyQuake for a recent M6.5 earthquake in Idaho for the 2 days around the earthquake (foreshocks and aftershocks). The catalog from running the example is in the examples folder: https://github.com/jakewalter/easyQuake/blob/master/examples/catalog_idaho_2days.xml

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
end_date = date(2020, 4, 2)

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

## Tips for successful outputs

Within your systems, consider running driver scripts as nohup background processes ```nohup python ~/work_dir/okla_daily.py &```. In this way, one could ```cat nohup.out | grep Traceback``` to understand python errors or ```grep nohup.out | Killed``` to understand when the system runs out of memory.

## Video intros to easyQuake

Most recent updates, recorded for the 2021 SSA Annual meeting: https://www.youtube.com/watch?v=bjBqPL9pD5w

Recorded for the fall 2020 Virtual SSA Eastern Section meeting: https://www.youtube.com/watch?v=coS2OwTWO3Y

## Running easyQuake with SLURM

If you have access to shared computing resources that utilize SLURM, you can drive easyQuake by making a bash script to run the example code or any code (thanks to Xiaowei Chen at OU). Save the following to a drive_easyQuake.sh and then run it
```
#!/bin/bash
#
#SBATCH --partition=gpu_cluster
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --output=easyquake_%J_stdout.txt
#SBATCH --error=easyquake_%J_stderr.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=easyquake
#SBATCH --mail-user=user@school.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/drive/group/user/folder
conda init bash
bash
conda activate easyquake
python idaho_example.py
```
## Version brief notes

Verson 0.6 (2/24/2021) - Implemented choice of GPD or EQTransformer pickers for the picking stage

Version 0.5 (2/10/2021) - includes embedded hypoinverse location functionality, rather than the simple location with the associator.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* code was used or otherwise changed to suit our purposes from obspy (https://github.com/obspy/obspy/wiki), gpd (https://github.com/interseismic/generalized-phase-detection), PhasePApy (https://github.com/austinholland/PhasePApy), EQTransformer (https://github.com/smousavi05/EQTransformer) and others
* would not be possible without the robust documentation in the obspy project
* this work was developed at the Oklahoma Geological Survey

