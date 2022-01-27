.. _Installation:

***************
Installation
***************

In most cases you would want to utilize this software on an NVIDIA CUDA-capable server, desktop, or laptop. The machine-learning pickers are configured to run with TensorFlow and that is the trickiest part of the installation process.

On most systems you should be able to simply::

	pip install easyQuake

As development rapidly continues, upgrade::
	
	pip install easyQuake --upgrade


If you want to tweak something, like the number of GPUs in gpd_predict, you could::

	git clone https://github.com/jakewalter/easyQuake.git
	cd easyQuake
	pip install .

The easiest way to install CUDA, TensorFlow, and Keras (while avoiding conflicting versions, etc.) is through installing Anaconda python and allowing conda-install to manage the dependencies. Since this has it's own headaches in running less than current versions of Python, we suggest a new conda environment. This can be achieved with (last tested January 2022)::
	
	conda create -n easyquake python=3.7 anaconda
	conda activate easyquake
	conda install tensorflow-gpu==2.1
	conda install keras
	conda install obspy -c conda-forge
	pip install easyQuake

One simple test that everything (especially TensorFlow) is working is to run "event-mode" on a short snippet of data for an earthquake detected by OGS in Oklahoma::

	from easyQuake import detection_association_event
	detection_association_event(project_folder='/scratch', project_code='ok', maxdist = 300, maxkm=300, local=True, machine=True, latitude=36.7, longitude=-98.4, max_radius=3, approxorigintime='2021-01-27T14:03:46', downloadwaveforms=True)


