.. _Installation:

***************
Installation
***************

**Version 2.0+ requires Python 3.10 or 3.11 and TensorFlow >= 2.12.**

In most cases you would want to utilize this software on an NVIDIA CUDA-capable server, desktop, or laptop. The machine-learning pickers are configured to run with TensorFlow and/or PyTorch. The event-mode can be run on a laptop CPU.

On most systems you should be able to simply::

	pip install easyQuake

As development rapidly continues, upgrade::

	pip install easyQuake --upgrade

If you want to tweak something, like the number of GPUs in gpd_predict, you could::

	git clone https://github.com/jakewalter/easyQuake.git
	cd easyQuake
	pip install .

Recommended conda environment (CPU or GPU)
--------------------------------------------

The easiest way to install CUDA, TensorFlow, and PyTorch while avoiding version conflicts is through Anaconda. Create a new environment with Python 3.11::

	conda create -n easyquake python=3.11
	conda activate easyquake
	conda install -c conda-forge obspy
	pip install tensorflow torch torchvision torchmetrics
	pip install easyQuake

For GPU support, install CUDA-enabled builds instead::

	conda create -n easyquake python=3.11
	conda activate easyquake
	conda install -c conda-forge obspy
	pip install tensorflow[and-cuda] torch torchvision torchmetrics --index-url https://download.pytorch.org/whl/cu118
	pip install easyQuake

Extras / partial installs
---------------------------

If you only need certain ML backends::

	pip install easyQuake[lite]    # obspy + pandas + tqdm only (no ML)
	pip install easyQuake[tf]      # adds TensorFlow only
	pip install easyQuake[torch]   # adds PyTorch only
	pip install easyQuake[ml]      # adds both TensorFlow and PyTorch

SeisBench picker (separate environment)
-----------------------------------------

The SeisBench picker must run in its own environment because its dependencies conflict with the TensorFlow environment used by GPD/EQTransformer/PhaseNet. easyQuake will call it as a subprocess automatically::

	conda create -n seisbench python=3.10
	conda activate seisbench
	pip install seisbench torch torchvision torchmetrics obspy

Legacy version (1.x — Python 3.7 / TF 2.2)
--------------------------------------------

If you need to run easyQuake with the older TensorFlow 2.2 stack (e.g., on an older CUDA system), pin to the 1.4 release::

	pip install "easyQuake==1.4.0"

Or build from the tagged commit::

	git clone https://github.com/jakewalter/easyQuake.git
	cd easyQuake
	git checkout v1.4.0
	pip install .

The legacy conda environment::

	conda create -n easyquake python=3.7 anaconda
	conda activate easyquake
	conda install tensorflow-gpu==2.2
	conda install keras
	conda install obspy -c conda-forge
	pip install "easyQuake==1.4.0"

Smoke test
-----------

One simple test that everything (especially TensorFlow) is working is to run "event-mode" on a short snippet of data for an earthquake detected by OGS in Oklahoma::

	from easyQuake import detection_association_event
	detection_association_event(project_folder='/scratch', project_code='ok', maxdist = 300, maxkm=300, local=True, machine=True, latitude=36.7, longitude=-98.4, max_radius=3, approxorigintime='2021-01-27T14:03:46', downloadwaveforms=True)


