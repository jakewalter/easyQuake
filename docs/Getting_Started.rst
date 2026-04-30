.. _Getting_Started:


******************
Running easyQuake
******************

.. _event-mode:

Event mode
=============================

Check that everything is installed and working as it should by detecting a single Oklahoma event on a snippet of downloaded miniseed data from the OGS network::
	
	from easyQuake import detection_association_event
	detection_association_event(project_folder='/scratch', project_code='ok', maxdist=300, maxkm=300, local=True, machine=True, latitude=36.7, longitude=-98.4, max_radius=3, approxorigintime='2021-01-27T14:03:46', downloadwaveforms=True)

A few things are going on here. You are downloading waveforms, detecting picks with the GPD picker, associating events to get an approximate location, and then approximating an event magnitude from the amplitudes of the horizontal seismograph components (i.e. Richter magnitude). These are steps that you would normally separate into 4, but they are all in one here. 

We can set various parameters here. In this case, we choose an approximate location for the data downloader (*lat* and *long*) and a maximum radius (*max_radius*) over which to download data from stations. Next we set *maxdist* and *maxkm*, the variables that control the maximum distance from a station to a possible hypocenter (based on detected S-P times) and the maximum distance over which to precompute the travel-times for the 1D traveltime lookup tables. The variables *local* and *machine* set whether the station metadata (or inventory file) should be within the day folder (it will be since we are downloading the waveforms) and whether the picker is the machine-learning picker. If everything runs at it should, it should create an event file ending with *.xml*, which is a QuakeML-formatted event file. You can inspect this with a text editor or open it with obspy with the *read_events* module.

Core modules
=============

Data Download
--------------
In reality, we don't really want to do earthquake detection based on a general idea of the time and location of an event. We want to determine events from continuous seismograms.

The easyQuake package can download waveforms for you and will organize the downloaded waveforms in a uniform manner so that it can run the subsequent submodules on the dataset. It leverages the obspy mass_downloader, with a few tweaks, so that by selecting a latitude and longitude bounding box, you will end up with folders in your working directory organized by year, day, and then month (YYYYDDMM).

Within each folder, the daylong miniseed files will be there, as well as the stationXML inventory metadata files. If you choose to work with a local dataset, it is easiest if you work with the same folder structure for your data.

As a tutorial example, let's gather data adjacent to the March 31, 2020 M6.5 Central Idaho earthquake::

        from easyQuake import download_mseed
        from easyQuake import daterange
        from datetime import date
        lat_a = 42
        lat_b = 47.5
        lon_a = -118
        lon_b = -111
        start_date = date(2020, 3, 31)
        end_date = date(2020, 4, 2)

        project_code = 'idaho'
        project_folder = '/data/id'
        for single_date in daterange(start_date, end_date):
            dirname = single_date.strftime("%Y%m%d")
            download_mseed(dirname=dirname, project_folder=project_folder, single_date=single_date, minlat=lat_a, maxlat=lat_b, minlon=lon_a, maxlon=lon_b)


Earthquake Detection
---------------------
Earthquake detection leverages either one of two machine-learning pickers: GPD (Ross et al., 2018) or EQTransformer (Mousavi et al., 2020). The earthquake detection module is simply a one-liner after data has been downloaded or otherwise organized, as described above. To continue with the tutorial exercise::
        
        from easyQuake import detection_continuous
        from easyQuake import daterange
        from datetime import date
        start_date = date(2020, 3, 31)
        end_date = date(2020, 4, 2)

        project_code = 'idaho'
        project_folder = '/data/id'
        for single_date in daterange(start_date, end_date):
            dirname = single_date.strftime("%Y%m%d")
            detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, single_date=single_date, machine=True,local=True)
            # run it with EQTransformer instead of GPD picker
            #detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, machine=True, machine_picker='EQTransformer', local=True, single_date=single_date)

It should be becoming apparent that the easyQuake submodules only work on a given day of data. Thus, the user needs to consider strategies by which several days of data can be analyzed in the loop with the daterange function. As you can see from the above code snippet, it runs the detection on each day of data and the detection is done in serial. Each day can only run at one time for a single workstation. This is due to the limited GPU resources available to us (one or two GPUs on a single workstation) - the machine learning pickers will automatically allocate all the GPU memory it can find so you cannot run detection in parallel - it would lead to errors that are difficult to track.

In the above code snippet, you can see uncomment the EQTransformer line and compare your detection results between GPD and EQTransformer pickers.

During this stage, you should see your video card working on the dataset. Run it in ipython and you can check that it is running correctly in a different terminal window by checking that video memory is being used by python with the "nvidia-smi" terminal command.

Event Association
------------------
For event association, we modified the PhasePApy package (Chen and Holland, 2016) to work within the easyQuake data structure and outputs from the picker. Either ML picker generates a file called gpd_picks.out that is simply a list of plausible detected events, time of detection, and station name. Another way to check that the pickers are working is whether there is data in the gpd_picks.out file. In the previous step, those detections are added to an sqlite file called 1dassociator.db, which should be within each day folder where detection has been completed. As above, the association can be completed on the example tutorial dataset::
        
        from easyQuake import association_continuous
        from easyQuake import daterange
        from easyQuake import combine_associated
        from datetime import date
        start_date = date(2020, 3, 31)
        end_date = date(2020, 4, 2)
        maxdist = 300
        maxkm = 300
        
        project_code = 'idaho'
        project_folder = '/data/id'
        for single_date in daterange(start_date, end_date):
            dirname = single_date.strftime("%Y%m%d")
            association_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, maxdist=maxdist, maxkm=maxkm, single_date=single_date, local=True)
        cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code)


This script runs the association step, in serial, within each day folder. Obviously, for a large amount of data, this would take quite a long time but is fine for the example tutorial. For longer datasets it would be better to utilize the Python multiprocessing package. As an example::
        
        from easyQuake import association_continuous
        from easyQuake import daterange
        from easyQuake import combine_associated
        from datetime import date
        start_date = date(2020, 3, 31)
        end_date = date(2020, 4, 2)
        maxdist = 300
        maxkm = 300
        
        from multiprocessing import Pool
        pool = Pool(20)
        project_code = 'idaho'
        project_folder = '/data/id'
        for single_date in daterange(start_date, end_date):
            dirname = single_date.strftime("%Y%m%d")
            pool.apply_async(association_continuous, (dirname, project_folder, project_code, maxdist, maxkm, single_date, True, 4, 1))
        pool.close()
        pool.join()  
        cat, dfs = combine_associated(project_folder=project_folder, project_code=project_code)

The final line in the association example combine all the day folder possible events and saves a *idaho_cat.xml* Obsby-compatible Catalog file in the *project_folder*. This file is also QuakeML compatible and includes pick and origin info. The catalog will be built out in subsequent post-processing steps (below).

Earthquake Magnitude
--------------------
Finally, we estimate earthquake magnitudes and the submodule *magnitude_quakeml* includes the amplitude and station magnitudes in the Catalog file::
        
        from easyQuake import magnitude_quakeml
        from easyQuake import simple_cat_df
        from obspy import read_events
        import matplotlib.pyplot as plt
        cat = read_events('idaho_cat.xml')
        cat = magnitude_quakeml(cat=cat, project_folder=project_folder,plot_event=True)
        cat.write('catalog_idaho.xml', format='QUAKEML')

        #test to see whether it was a success 
        catdf = simple_cat_df(cat)
        plt.figure()
        plt.plot(catdf.index,catdf.magnitude,'.')


Additional Modules
===================

In addition to the core modules, we have written various submodules to extend easyQuake/

Hypoinverse 
-------------
Assuming that the user's computer has hypoinverse installed (*hyp2000* as it is compiled on most systems) and it resides in the user's path, you can drive hypoinverse with easyQuake. The steps above give a location at 5 km depth during association, which can be quite different than the absolute location::
        
        from easyQuake import locate_hyp2000
        cat = locate_hyp2000(cat=cat, project_folder=project_folder)
        cat.write('catalog_idaho_hyp.xml', format='QUAKEML')

If you examine the Catalog object, you can see that there are more than one origin for most events::
        
        print(cat[0].origins)

Interface with Seiscomp 
-----------------------
At OGS, our analysts review ML-derived events and re-adjust picks, etc. The easyQuake events can be easily passed to a Seiscomp system. For this example, Machine #1 can be the computer that runs easyQuake, while Machine #2 will be the Seiscomp production system. First, on Machine #1::
        
        #export to SC3ML file rather than QuakeML
	from easyQuake import single_event_xml
	format = "SC3ML"
	#remove the previous zip file
	os.system('rm -r /scratch/scratch/'+format.lower()+'*')
	single_event_xml(cat,'/scratch/scratch',format)
	import shutil
	shutil.make_archive('/scratch/scratch/sc3ml', 'zip', '/scratch/scratch/', '/scratch/scratch/sc3ml')
	os.system('scp /scratch/scratch/sc3ml.zip /machine2/directory/ml/')

Then on Machine #2, we have a listener written in bash that waits for file ingestion and strips away the event info so that it can be added to the event queue for review::
        
        #!/bin/bash
	/usr/local/bin/inotifywait -m /home/sysop/incoming_ML -e create -e moved_to |
    	while read directory action file; do
        	if [[ "$file" =~ .*xml$ ]]; then # Does the file end with .xml?
            		scdispatch -i /home/sysop/incoming_ML/$file -O add --routingtable Pick:PICK,Amplitude:AMPLITUDE,Origin:LOCATION,StationMagnitude:MAGNITUDE,Magnitude:MAGNITUDE
            		rm /home/sysop/incoming_ML/$file
        	fi
    	done

HypoDD or Growclust relative relocation
---------------------------------------
Oftentimes, ML pickers also are not 100% accurate and we want to determine relative locations through the use of HypoDD or Growclust::
       
        git clone https://github.com/jakewalter/hypoDDpy.git
        cd hypoDDpy
        pip install .

Once this is installed, you can run it and generate the cross-correlations in the way in which it is described in hypoDDpy below. Note that you should run the easyQuake function fix_picks_catalog before to make sure that the components are correct (easyQuake does not usually correctly have the accurate ::
       
        from easyQuake import fix_picks_catalog
        cat2 = fix_picks_catalog(cat, project_folder)
        cat2.write('catalog_fixed.xml','QUAKEML')
        relocator = HypoDDRelocator(working_dir="relocate1",
            cc_time_before=0.05,
            cc_time_after=0.2,
            cc_maxlag=0.2,
            cc_filter_min_freq=2.0,
            cc_filter_max_freq=14.0,
            cc_p_phase_weighting={"Z": 1.0},
            cc_s_phase_weighting={"E": 1.0, "N": 1.0, "1": 1.0, "2": 1.0},
            cc_min_allowed_cross_corr_coeff=0.4)

        # Add the necessary files. Call a function multiple times if necessary.
        relocator.add_event_files(glob.glob("/data/proj_dir/catalog_fixed.xml"))
        relocator.add_waveform_files(glob.glob("/data/proj_dir/20*/*mseed"))
        relocator.add_station_files(glob.glob("/data/proj_dir/20*/*.xml"))

        # Setup the velocity model. This is just a constant velocity model.
        relocator.setup_velocity_model(
            model_type="layered_p_velocity_with_constant_vp_vs_ratio",
            layer_tops=[(0, 2.7),(0.3,2.95),(1.0,4.15),(1.5,5.8),(21,6.3)],
            vp_vs_ratio=1.73)

        # Start the relocation with the desired output file.
        relocator.start_relocation(output_event_file="relocated_events.xml")            

You can use the easyQuake utility::
       
        from easyQuake import quakeml_to_growclust
        quakeml_to_growclust(project_folder='.')


HypoDD or Growclust relative relocation
---------------------------------------
Focal mechanisms::
       
        git clone https://github.com/jakewalter/hashpy.git
        cd hashpy
        pip install .
        

Tips for Success
================

Most of the time it is beneficial to run jobs overnight and in the background (or several days for longer datasets)::

	> nohup python yourscript.py &

If something goes wrong, you can inspect the nohup.out file (or just the end of it)::

	> tail -n 100 nohup.out

Utility Modules
================

easyQuake includes several utility modules for post-processing, format conversion, and catalog preparation.

Cut Event Waveforms
--------------------

After building a catalog, you can cut event waveforms from your continuous data archive into individual event miniSEED files. This is useful for building training datasets for new ML models::

        from easyQuake import cut_event_waveforms
        cut_event_waveforms(cat=cat, project_folder=project_folder, pre_sec=5, post_sec=30)

The output is organized by event in the project folder. Each event gets a subfolder with the cut waveforms.

Convert Catalog to HDF5 (for ML training)
-------------------------------------------

The ``quakeML_to_hdf5`` utility converts a QuakeML-format catalog (with associated picks) into the HDF5 waveform format expected by EQTransformer and similar models for transfer learning or re-training::

        from easyQuake import quakeML_to_hdf5
        quakeML_to_hdf5(cat=cat, project_folder=project_folder, pre_sec=5, post_sec=30)

This is particularly useful if you want to fine-tune a picker on your regional dataset.

Convert to GrowClust format
-----------------------------

If you want to run the GrowClust relative relocation algorithm, easyQuake can output your catalog in the correct format::

        from easyQuake import quakeml_to_growclust
        quakeml_to_growclust(project_folder='.')

PyOcto Association
-------------------

As an alternative to the PhasePApy 1D associator, easyQuake supports the PyOcto associator (v1.4+). PyOcto can be faster for dense networks::

        from easyQuake import detection_continuous, pyocto_continuous
        # ... (run detection_continuous as above) ...
        pyocto_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code, single_date=single_date)

SeisBench Picker
=================

Version 1.4+ includes integration with `SeisBench <https://github.com/seisbench/seisbench>`_, which provides a unified interface to many pretrained models (PhaseNet, GPD, EQTransformer, and more) along with fine-tuned regional weight sets.

Because SeisBench has conflicting dependencies with the TensorFlow environment, it runs in a **separate conda environment** and easyQuake calls it as a subprocess.

Setting up the SeisBench environment::

        conda create -n seisbench python=3.10
        conda activate seisbench
        pip install seisbench torch torchvision torchmetrics obspy

Running continuous detection with SeisBench::

        from easyQuake import detection_continuous
        detection_continuous(dirname=dirname, project_folder=project_folder, project_code=project_code,
                             single_date=single_date, machine=True, local=True,
                             machine_picker='seisbench',
                             seisbenchmodel='/path/to/your/model.pth')

The ``seisbenchmodel`` argument accepts the full path to a SeisBench-format model directory or a ``.pth`` state-dict file. If you have trained a regional model with SeisBench, point it here.

Quasi-Realtime Detection
=========================

The ``realtime/`` folder contains scripts for quasi-realtime earthquake detection driven by a SeedLink stream. Instead of processing full daylong files, the realtime scripts receive short data packets from a SeedLink server and trigger the detection/association pipeline on each new snippet as it arrives.

**How it works:**

1. ``seedlink_connection_v5.py`` connects to your SeedLink server, receives waveform packets, and writes them into a time-windowed directory structure under your project folder. It writes an ``rt.xml`` sentinel file when a window is complete.
2. ``rt_easyquake.py`` runs a polling loop (every 5 seconds) that detects new ``rt.xml`` files and spawns a background thread calling ``detection_association_event`` on that time window. It also performs a periodic re-scan every hour to catch any missed windows.
3. Detected events are written as ``*_seiscomp.xml`` files in the project folder root, ready for dispatch to a SeisComP system.

**Running the realtime stack:**

Edit the host, port, and station list in ``seedlink_connection_v5.py`` for your network, then::

        # In terminal 1 — ingest waveforms from SeedLink
        nohup python realtime/seedlink_connection_v5.py &

        # In terminal 2 — run detection on each new window
        nohup python realtime/rt_easyquake.py /scratch/realtime PhaseNet &

        # Check what's happening
        tail -f nohup.out

**Sending events to SeisComP:**

On the SeisComP server, use an ``inotify``-based dispatcher to call ``scdispatch`` whenever a new XML file appears in the incoming directory (see ``realtime/README.md`` for the full script).

For fully native SeisComP integration — where picks are published directly to the SC messaging bus without any file-based handoff — see the ``sceasyquake`` module in the companion repository: https://github.com/jakewalter/easyQuake_seiscomp

SeisComP Native Integration
============================

The companion **sceasyquake** module provides deep integration with SeisComP 5, publishing ML picks directly to the SeisComP messaging bus as native ``Pick`` and ``Amplitude`` objects. This means picks flow into ``scautoloc`` automatically, events are formed by ``scevent``, and everything appears in ``scolv`` for analyst review — without any file-based intermediate step.

See the full documentation at: https://github.com/jakewalter/easyQuake_seiscomp

**Quick install:**

.. code-block:: bash

    git clone https://github.com/jakewalter/easyQuake_seiscomp.git
    cd easyQuake_seiscomp/sceasyquake
    export SC_PYTHON=$(seiscomp exec which python3)
    $SC_PYTHON -m pip install obspy seisbench
    bash install.sh
    seiscomp enable sceasyquake
    seiscomp start sceasyquake

**Minimal configuration** (``$SEISCOMP_ROOT/etc/sceasyquake.cfg``)::

    seedlink.host      = localhost
    seedlink.port      = 18000
    streams.codes      = TX.*.00.HH?
    picker.backend     = phasenet
    picker.device      = cuda
    picker.threshold   = 0.5
    picker.step_seconds = 5

