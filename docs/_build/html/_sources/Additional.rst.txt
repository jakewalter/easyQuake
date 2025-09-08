.. _Additional:

***************
Function Reference
***************

This section provides searchable docstrings for key functions in easyQuake modules. Use your browser's search to quickly find function usage and arguments.

-------------------
easyQuake.easyQuake
-------------------

.. rubric:: download_mseed

Downloads seismic data in miniSEED format from IRIS DMC within a specified geographic and temporal domain.

**Args:**
    dirname (str): Name of directory where miniSEED files will be stored. If None, the directory will be named after the specified date.
    project_folder (str): Name of project folder where miniSEED files will be stored. If None, files will be stored in the current directory.
    single_date (datetime): Date in YYYY-MM-DD format specifying the day for which data will be downloaded.
    minlat (float): Minimum latitude for the geographic domain.
    maxlat (float): Maximum latitude for the geographic domain.
    minlon (float): Minimum longitude for the geographic domain.
    maxlon (float): Maximum longitude for the geographic domain.
    dense (bool): Whether to download data with high temporal and spatial resolution. If True, data with minimum inter-station distance of 1 meter will be downloaded. Otherwise, data with minimum inter-station distance of 5000 meters will be downloaded.
    raspberry_shake (bool): Whether to download data from Raspberry Shake stations in addition to the standard IRIS DMC stations.

**Returns:**
    None

.. rubric:: build_tt_tables_local_directory

This function builds travel-time lookup tables for seismic stations located in a specified directory using a specified model. The function takes several optional arguments, including the directory name, project folder, channel codes, database, maximum distance, source depth, delta distance, and model.

**Args:**
    dirname (str, optional): The directory containing the station inventory files. Defaults to None.
    project_folder (str, optional): The project folder containing the station inventory directory. Defaults to None.
    channel_codes (list of str, optional): The channel codes to be included. Defaults to ['EH', 'BH', 'HH', 'HN'].
    db (str, optional): The SQLAlchemy database connection string. Defaults to None.
    maxdist (float, optional): The maximum distance for which to calculate travel times, in km. Defaults to 800.0.
    source_depth (float, optional): The depth of the seismic source, in km. Defaults to 5.0.
    delta_distance (int, optional): The spacing between distances for which to calculate travel times, in km. Defaults to 1.
    model (str, optional): The name of the travel-time model to use. Defaults to None.

**Returns:**
    inv (Inventory): The station inventory, populated with information from the specified directory.

.. rubric:: detection_continuous

Continuous detection of seismic events using single-station waveform data.

**Args:**
    dirname (str, optional): The directory name where the data will be stored.
    project_folder (str, optional): The project directory where the data will be stored.
    project_code (str, optional): The project code name.
    local (bool, optional): If True, read in data from a local directory. If False, read in data from an online database.
    machine (bool, optional): If True, use an automated detection algorithm. If False, use a manual detection algorithm.
    machine_picker (str, optional): The name of the automated detection algorithm to use. Only used if machine=True. Defaults to 'GPD'.
    single_date (datetime.datetime, optional): The date and time to start detection.
    make3 (bool, optional): Whether to split data into 3-hour chunks. Defaults to True.
    latitude (float, optional): The latitude of the location where data is to be collected. Only used if local=False.
    longitude (float, optional): The longitude of the location where data is to be collected. Only used if local=False.
    max_radius (float, optional): The maximum distance, in degrees, from the latitude and longitude to search for stations. Only used if local=False.
    fullpath_python (str, optional): The path to the Python executable. Only used if machine=True.
    filtmin (float, optional): The minimum frequency for the filter.
    filtmax (float, optional): The maximum frequency for the filter.
    t_sta (float, optional): The length of the short-term average window for the STA/LTA algorithm.
    t_lta (float, optional): The length of the long-term average window for the STA/LTA algorithm.
    trigger_on (float, optional): The threshold for triggering an event.
    trigger_off (float, optional): The threshold for ending an event.
    trig_horz (float, optional): The horizontal distance between events required for them to be considered separate.
    trig_vert (float, optional): The vertical distance between events required for them to be considered separate.
    seisbenchmodel (str, optional): The full path and model name of the seisbench trained model

**Returns:**
    None

-------------------
easyQuake.gpd_predict.gpd_predict
-------------------

.. rubric:: sliding_window

Calculate a sliding window over a signal.

**Args:**
    data : numpy array. The array to be slided over.
    size : int. The sliding window size
    stepsize : int. The sliding window stepsize. Defaults to 1.
    axis : int. The axis to slide over. Defaults to the last axis.
    copy : bool. Return strided array as copy to avoid sideffects when manipulating the output array.

**Returns:**
    data : numpy array. A matrix where row in last dimension consists of one instance of the sliding window.

-------------------
easyQuake.phasenet.phasenet_predict
-------------------

.. rubric:: pred_fn

Runs prediction for seismic phase picking using a trained PhaseNet model.

**Args:**
    args: Parsed command-line arguments.
    data_reader: DataReader object for input data.
    figure_dir: Directory for output figures (optional).
    prob_dir: Directory for output probabilities (optional).
    log_dir: Directory for logs (optional).

**Returns:**
    0 on success.

-------------------
easyQuake.seisbench.run_seisbench
-------------------

.. rubric:: main

Runs seisbench model for seismic phase picking and outputs picks to file.

**Args:**
    -I: Input file (required)
    -O: Output file (required)
    -M: Model full path and filename (required)
    -F: Path where GPD lives (optional)

**Returns:**
    None

