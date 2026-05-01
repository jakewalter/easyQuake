# easyQuake

[![CI](https://github.com/jakewalter/easyQuake/actions/workflows/ci.yml/badge.svg)](https://github.com/jakewalter/easyQuake/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/easyquake/badge/?version=latest)](https://easyquake.readthedocs.io/en/latest/?badge=latest)

Simplified machine-learning driven earthquake detection, location, and analysis in one easy-to-implement python package.

For full documentation, including a complete function reference, see: https://easyquake.readthedocs.io/

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

**Version 2.0+ (current, Python 3.10+)**

This code leverages machine-learning for earthquake detection with the choice of the GPD (https://github.com/interseismic/generalized-phase-detection), EQTransformer (https://github.com/smousavi05/EQTransformer), PhaseNet (https://github.com/AI4EPS/PhaseNet), or SeisBench (https://github.com/seisbench/seisbench) pickers. Suitable GPU hardware is recommended for continuous processing, but the event-mode can be run efficiently on a laptop CPU.

* **Python 3.10 or 3.11** (tested in CI on both Ubuntu and macOS)
* **TensorFlow >= 2.12** (GPU support: install a CUDA-enabled build separately if needed)
* **PyTorch >= 1.13**
* **obspy >= 1.3**, **pandas >= 1.5**, **tqdm**

### Recommended conda environment (current version)
```
conda create -n easyquake python=3.11
conda activate easyquake
conda install -c conda-forge obspy
pip install tensorflow torch torchvision torchmetrics
pip install easyQuake
```

If you have an NVIDIA GPU, install CUDA-enabled TensorFlow and PyTorch instead:
```
conda create -n easyquake python=3.11
conda activate easyquake
conda install -c conda-forge obspy
pip install tensorflow[and-cuda] torch torchvision torchmetrics --index-url https://download.pytorch.org/whl/cu118
pip install easyQuake
```

### Extras / partial installs

If you only need certain ML backends, use the package extras:
```
pip install easyQuake[lite]    # obspy + pandas + tqdm only (no ML)
pip install easyQuake[tf]      # adds TensorFlow only
pip install easyQuake[torch]   # adds PyTorch only
pip install easyQuake[ml]      # adds both TensorFlow and PyTorch
```

### SeisBench note
The SeisBench picker runs in its **own separate environment** because its dependencies conflict with the TensorFlow environment used by GPD/EQTransformer/PhaseNet. easyQuake will call it as a subprocess automatically:
```
conda create -n seisbench python=3.10
conda activate seisbench
pip install seisbench torch torchvision torchmetrics obspy
```

---

### Legacy version (1.x, Python 3.7–3.8)

If you need to run easyQuake with the legacy TensorFlow 2.2 stack (e.g., on an older CUDA system), pin to the 1.4 release:
```
pip install "easyQuake==1.4.0"
```

Or build from the tagged commit:
```
git clone https://github.com/jakewalter/easyQuake.git
cd easyQuake
git checkout v1.4.0
pip install .
```

The legacy conda environment:
```
conda create -n easyquake python=3.7 anaconda
conda activate easyquake
conda install tensorflow-gpu==2.2
conda install keras
conda install obspy -c conda-forge
pip install "easyQuake==1.4.0"
```

> **Note:** Legacy installs required `keras==2.3.1`, `h5py==2.10.0`, `tensorflow-gpu==2.2`, and `protobuf==3.20.*`. These are incompatible with Python 3.9+ and are no longer installed by default.

---

## Running easyQuake

The first example is a simple one in "event mode" - try it:

```
from easyQuake import detection_association_event

detection_association_event(project_folder='/scratch', project_code='ok', maxdist = 300, maxkm=300, local=True, machine=True, latitude=36.7, longitude=-98.4, max_radius=3, approxorigintime='2021-01-27T14:03:46', downloadwaveforms=True)
```

This next example runs easyQuake for a recent M6.5 earthquake in Idaho for the 2 days around the earthquake (foreshocks and aftershocks). The catalog from running the example is in the examples folder: https://github.com/jakewalter/easyQuake/blob/master/examples/catalog_idaho_2days.xml

If you don't have a suitable computer, try it in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakewalter/easyQuake/blob/master/examples/easyquake_demo.ipynb)

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
    #run it with EQTransformer instead of GPD picker
    #detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, machine=True, machine_picker='EQTransformer', local=True, single_date=single_date)
    #PhaseNet
    #detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, machine=True, machine_picker='PhaseNet', local=True, single_date=single_date)
    association_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, maxdist=maxdist, maxkm=maxkm, single_date=single_date, local=True)
    ### IMPORTANT - must call the specific picker to create association and catalogs specific to that picker within each dayfolder!!
    #association_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, maxdist=maxdist, maxkm=maxkm, single_date=single_date, local=True, machine_picker='EQTransformer')
    #association_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, maxdist=maxdist, maxkm=maxkm, single_date=single_date, local=True, machine_picker='PhaseNet')

cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code)
#cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code, machine_picker='EQTransformer')
#cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code, machine_picker='PhaseNet')
cat = magnitude_quakeml(cat=cat, project_folder=project_folder,plot_event=True)
cat.write('catalog_idaho.xml',format='QUAKEML')


catdf = simple_cat_df(cat)
plt.figure()
plt.plot(catdf.index,catdf.magnitude,'.')
```

## Quasi-realtime mode

The `realtime/` folder contains scripts for quasi-realtime earthquake detection driven by a SeedLink stream. Rather than processing daylong files, these scripts watch for short data packets from a SeedLink server, write them to a rolling directory structure, and trigger `detection_association_event` on each new snippet as it arrives.

**Scripts:**
- `rt_easyquake.py` — Main realtime loop. Watches for `rt.xml` trigger files written by the SeedLink client and launches detection/association in a background thread for each new time window. Performs a periodic full re-scan (hourly) to catch any missed windows.
- `seedlink_connection_v5.py` — SeedLink client that connects to a server, receives waveforms, and writes them into the project folder structure.
- `seedlink_sds_connection.py` — Alternate client that writes data into an SDS (SeisComP Data Structure) archive layout.

**Quick start:**
```bash
# Start the SeedLink data ingest (edit host/port/stations inside the script)
nohup python realtime/seedlink_connection_v5.py &

# Start realtime detection (edit project folder and picker name)
nohup python realtime/rt_easyquake.py /scratch/realtime PhaseNet &
```

The realtime scripts write SeisComP-compatible XML event files (e.g. `*_seiscomp.xml`) to the project folder. These can be ingested into a running SeisComP system automatically — see the `realtime/README.md` for the `inotify`-based dispatcher pattern that calls `scdispatch` whenever a new XML file is detected.

For fully native SeisComP integration (picks published directly to the SC messaging bus), see the **SeisComP Integration** section at the bottom of this page.

## Tips for successful outputs

Within your systems, consider running driver scripts as nohup background processes ```nohup python ~/work_dir/okla_daily.py &```. In this way, one could ```cat nohup.out | grep Traceback``` to understand python errors or ```grep nohup.out | Killed``` to understand when the system runs out of memory.

**GPU memory errors:** If you see `ResourceExhaustedError` or similar GPU OOM messages, try reducing `batch_size` in the picker call, or set `CUDA_VISIBLE_DEVICES=-1` to force CPU inference.

**TensorFlow warnings at startup:** TF 2.12+ is verbose about GPU discovery. These are informational only — easyQuake will automatically fall back to CPU if no compatible GPU is found.

**PyTorch CUDA mismatch:** If SeisBench fails with "no kernel image is available for execution on the device", your installed PyTorch was not built for your CUDA version. Reinstall PyTorch with the correct CUDA tag from https://pytorch.org/get-started/locally/ or run with CPU (`CUDA_VISIBLE_DEVICES=-1`).

**protobuf conflicts:** TensorFlow >= 2.12 requires `protobuf >= 3.20, < 4`. If you see protobuf errors, run `pip install "protobuf>=3.20,<4"`.

**Import errors after upgrading from 1.x:** The 2.0 rewrite removed `keras` as a standalone dependency and dropped the `tensorflow-gpu` split package. Uninstall both and reinstall: `pip uninstall keras tensorflow-gpu tensorflow && pip install tensorflow`.

## Video intros to easyQuake

Most recent updates, recorded for the 2021 SSA Annual meeting: https://www.youtube.com/watch?v=bjBqPL9pD5w

Recorded for the fall 2020 Virtual SSA Eastern Section meeting: https://www.youtube.com/watch?v=coS2OwTWO3Y

## About EasyQuake

Stay up to date on the latest description of EasyQuake contents: https://easyquake.readthedocs.io/en/latest/About.html

A complete **Function Reference** covering all 44+ public functions (data download, detection, association, magnitude, location, catalog utilities, format conversion, and plotting) is available at: https://easyquake.readthedocs.io/en/latest/Additional.html

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

Version 2.0 (4/30/2026) = Major modernization. Requires Python 3.10+. All ML detection models (GPD, EQTransformer, PhaseNet, SeisBench) rewritten to be compatible with TensorFlow >= 2.12 and PyTorch >= 1.13. Dropped legacy `tensorflow-gpu`, `keras==2.3.1`, and `h5py==2.10.0` pinned dependencies. Added extras_require install options (`lite`, `tf`, `torch`, `ml`). CI tested on Python 3.10 and 3.11. PhaseNet entrypoint fixed to work without arguments. assoc1D import fixes. Documentation overhauled: Sphinx upgraded to 7.x, ReadTheDocs builds fixed (added `.readthedocs.yaml`), complete function reference added covering all 44+ public functions.

Version 1.4 (9/30/2024) = Long overdue version update, including modules for PyOcto association conversion to QuakeML file and seisbench picker integration.

Version 1.3 (11/22/2022) = PhaseNet now included, in addition to GPD and EQTransformer pickers. Numerous other bugs squashed.

Version 1.2 (8/1/2022) - Rewrote the non-ML picker to be easier to work with (recursive_sta_lta from obpsy) and include input of those parameters within detection_continuous function.

Version 0.9 (2/23/2022) - Modules to cut easyQuake event waveforms from continuous data (cut_event_waveforms) and module for converting easyQuake catalog (or any QuakeML-formatted catalog) to HDF5 (quakeML_to_hdf5) for training new ML models

Version 0.8 (7/30/2021) - Several major bug fixes and improved controls for Hypoinverse location

Verson 0.6 (2/24/2021) - Implemented choice of GPD or EQTransformer pickers for the picking stage

Version 0.5 (2/10/2021) - includes embedded hypoinverse location functionality, rather than the simple location with the associator.

## SeisComP Integration

For users running a SeisComP seismic network management system, easyQuake ML picks can be published directly to the SeisComP messaging bus as native `Pick` objects — making them available to `scautoloc`, `scevent`, and other SC modules as if they came from `scautopick`.

This is provided by the companion module **sceasyquake**: https://github.com/jakewalter/easyQuake_seiscomp

`sceasyquake` is a SeisComP 5 module that:
- Connects to a SeedLink server and buffers incoming waveforms
- Runs continuous ML phase picking (P and S) using any of the supported backends: PhaseNet, GPD, EQTransformer, or any SeisBench model
- Publishes picks and SNR amplitudes to the SC messaging `PICK` group in real time
- Works as a drop-in companion to `scautopick` — both can run simultaneously

**Supported pickers:**

| Backend | Model | Notes |
|---|---|---|
| `phasenet` | PhaseNet | easyQuake native, TF >= 2.12 |
| `gpd` | GPD | easyQuake native, threshold default 0.994 |
| `eqtransformer` | EQTransformer | P + S + detection confidence |
| `seisbench` | any SeisBench model | configurable via `picker.model` |
| `auto` | PhaseNet (SeisBench) | auto-selects best available backend |

**Quick install:**
```bash
git clone https://github.com/jakewalter/easyQuake_seiscomp.git
cd easyQuake_seiscomp/sceasyquake
export SC_PYTHON=$(seiscomp exec which python3)
$SC_PYTHON -m pip install obspy seisbench
$SC_PYTHON -m pip install -e ~/easyQuake   # optional: bundled weights
bash install.sh
seiscomp enable sceasyquake
seiscomp start sceasyquake
```

See the [easyQuake_seiscomp README](https://github.com/jakewalter/easyQuake_seiscomp) for full installation, configuration, benchmarking, and troubleshooting details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* code was used or otherwise changed to suit our purposes from obspy (https://github.com/obspy/obspy/wiki), GPD (https://github.com/interseismic/generalized-phase-detection), PhasePApy (https://github.com/austinholland/PhasePApy), EQTransformer (https://github.com/smousavi05/EQTransformer), PhaseNet (https://github.com/AI4EPS/PhaseNet), and others
* would not be possible without the robust documentation in the obspy project
* this work was developed at the Oklahoma Geological Survey

