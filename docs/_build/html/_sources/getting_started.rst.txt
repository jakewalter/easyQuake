.. _Getting_Started:


*****************
Running easyQuake
*****************

.. _event-mode:

Event mode
=============================

Check that everything is installed and working as it should by detecting a single Oklahoma event on a snippet of downloaded miniseed data from the OGS network::
	
	from easyQuake import detection_association_event
	detection_association_event(project_folder='/scratch', project_code='ok', maxdist=300, maxkm=300, local=True, machine=True, latitude=36.7, longitude=-98.4, max_radius=3, approxorigintime='2021-01-27T14:03:46', downloadwaveforms=True)

A few things are going on here. You are downloading waveforms, detecting picks with the GPD picker, associating events to get an approximate location, and then approximating an event magnitude from the amplitudes of the horizontal seismograph components (i.e. Richter magnitude). These are steps that you would normally separate into 4, but they are all in one here. 

We can set various paramaters here. In this case, we choose an approximate location for the data downloader (*lat* and *long*) and a maximum radius (*max_radius*) over which to download data from stations. Next we set *maxdist* and *maxkm*, the variables that control the maximum distance from a station to a possible hypocenter (based on detected S-P times) and the maximum distance over which to precompute the travel-times for the 1D traveltime lookup tables. The variables *local* and *machine* set whether the station metadata (or inventory file) should be within the day folder (it will be since we are downloading the waveforms) and whether the picker is the machine-learning picker. If everything runs at it should, it should create an event file ending with *.xml*, which is a QuakeML-formatted event file. You can inspect this with a text editor or open it with obspy with the *read_events* module.

Core modules
=============

Data Download
--------------
In reality, we don't really want to do earthquake detection based on a general idea of the time and location of an event. We want to determine events from continuos seismograms.


YYYYDDMM

Earthquake Detection
---------------------


Event Association
------------------


Earthquake Magnitude
--------------------


Additional Modules
===================

Tips for Success
================

Most of the time it is beneficial to run jobs overnight and in the background (or several days for longer datasets)::

	> nohup python yourscript.py &

If something goes wrong, you can inspect the nohup.out file (or just the end of it)::

	> tail -n 100 nohup.out
