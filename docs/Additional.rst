.. _Additional:

******************
Function Reference
******************

This section provides a complete reference for all public functions in easyQuake. Functions are grouped by category. Use your browser's Ctrl+F / Cmd+F to search for a specific function name or keyword.

.. contents:: Categories
   :local:
   :depth: 1

-------------------
easyQuake.easyQuake
-------------------

Data Download
~~~~~~~~~~~~~

.. rubric:: download_mseed

Downloads one day of seismic waveform data in miniSEED format from IRIS/FDSN via ObsPy MassDownloader. Stores files and StationXML in ``project_folder/dirname``.

**Args:**

    dirname (str): Sub-directory name inside *project_folder* where files are saved.
    project_folder (str): Root project directory.
    single_date (datetime.date): Calendar day to download.
    minlat, maxlat (float): Latitude bounds of the download domain.
    minlon, maxlon (float): Longitude bounds of the download domain.
    dense (bool): If ``True``, minimum inter-station spacing is 1 m (dense network). If ``False`` (default), 5000 m spacing is enforced.
    raspberry_shake (bool): Also download from the Raspberry Shake network (``https://data.raspberryshake.org``) when ``True``.

**Returns:** None

**Example:**

.. code-block:: python

    from datetime import date
    download_mseed(dirname='20230101', project_folder='/data/myproject',
                   single_date=date(2023, 1, 1),
                   minlat=44.0, maxlat=45.0, minlon=-116.0, maxlon=-115.0)

----

.. rubric:: download_mseed_event

Downloads waveform data for a time window bounded by a rectangular geographic domain. Useful for downloading waveforms around a known event time.

**Args:**

    dirname (str): Sub-directory where files are saved.
    project_folder (str): Root project directory.
    starting (UTCDateTime): Start of download window.
    stopping (UTCDateTime): End of download window.
    minlat, maxlat (float): Latitude bounds.
    minlon, maxlon (float): Longitude bounds.
    maxrad (float): Unused in this variant (kept for API consistency).

**Returns:** None

----

.. rubric:: download_mseed_event_radial

Downloads waveform data for a time window within a circular geographic domain centred on a point. Preferred for single-event targeted downloads.

**Args:**

    dirname (str): Sub-directory where files are saved.
    project_folder (str): Root project directory.
    starting (UTCDateTime): Start of download window.
    stopping (UTCDateTime): End of download window.
    lat1 (float): Centre latitude.
    lon1 (float): Centre longitude.
    maxrad (float): Search radius in degrees.

**Returns:** None

----

Travel-Time Tables
~~~~~~~~~~~~~~~~~~

.. rubric:: build_tt_tables

Builds 1-D travel-time lookup tables by downloading station metadata from IRIS for a geographic region and time window, then computing P and S travel times with ObsPy TauPy.

**Args:**

    lat1 (float): Centre latitude of the search region.
    long1 (float): Centre longitude of the search region.
    maxrad (float): Search radius in degrees.
    starting (UTCDateTime): Start of metadata query window.
    stopping (UTCDateTime): End of metadata query window.
    channel_codes (list[str]): Channel prefixes to include, e.g. ``['EH','BH','HH']``.
    db (str): SQLAlchemy connection string for the travel-time SQLite database.
    maxdist (float): Maximum source-receiver distance in km. Default 500.
    source_depth (float): Fixed source depth in km. Default 5.
    delta_distance (int): Distance increment in km. Default 1.
    model (str or None): TauPy velocity model name (e.g. ``'iasp91'``). Defaults to ``'iasp91'``.

**Returns:** ``obspy.Inventory`` – the downloaded station inventory.

----

.. rubric:: build_tt_tables_local_directory

Builds travel-time lookup tables from StationXML files already present in a local data directory. Use this when data was downloaded with ``download_mseed`` and station metadata is stored locally.

**Args:**

    dirname (str): Sub-directory containing StationXML / dailyinventory.xml files.
    project_folder (str): Root project directory.
    channel_codes (list[str]): Channel prefixes to include. Default ``['EH','BH','HH','HN']``.
    db (str): SQLAlchemy connection string for the travel-time SQLite database.
    maxdist (float): Maximum source-receiver distance in km. Default 800.
    source_depth (float): Fixed source depth in km. Default 5.
    delta_distance (int): Distance increment in km. Default 1.
    model (str or None): TauPy velocity model name. Defaults to ``'iasp91'``.

**Returns:** ``obspy.Inventory`` – the combined station inventory from the directory.

----

.. rubric:: build_tt_tables_local_directory_ant

Antenna-array variant of ``build_tt_tables_local_directory``. Intended for very dense networks where individual station spacing is small. Channel codes default to ``['EH','BH','HH']`` (no strong-motion).

**Args:** Same as ``build_tt_tables_local_directory`` except ``channel_codes`` default and internal distance handling differ.

**Returns:** ``obspy.Inventory``

----

Detection
~~~~~~~~~

.. rubric:: detection_continuous

Runs one day of single-station ML picking (or STA/LTA) on a directory of waveform files. Creates a per-day SQLite picks database. Called in sequence before ``association_continuous``.

Supported ``machine_picker`` values:

* ``'GPD'`` – Generalized Phase Detection (default)
* ``'EQTransformer'`` – EQTransformer deep-learning picker
* ``'PhaseNet'`` – PhaseNet CNN picker
* ``'Seisbench'`` – any SeisBench-compatible model (requires separate conda environment)
* ``None`` / ``machine=False`` – classic STA/LTA trigger

**Args:**

    dirname (str): Sub-directory containing the daily waveform files.
    project_folder (str): Root project directory.
    project_code (str): Short project identifier string.
    local (bool): ``True`` to read station inventory from local files; ``False`` to query FDSN.
    machine (bool): ``True`` to use an ML picker; ``False`` for STA/LTA.
    machine_picker (str or None): Picker name (see above). Defaults to ``'GPD'`` when ``machine=True``.
    single_date (datetime.datetime): Day to process.
    make3 (bool): Synthesise missing horizontal components from the vertical when only one component is available. Default ``True``.
    latitude, longitude (float): Used when ``local=False`` to query FDSN for station inventory.
    max_radius (float): FDSN query radius in degrees (only used when ``local=False``).
    fullpath_python (str or None): Path to the Python interpreter; only needed for the CLI fallback path.
    filtmin, filtmax (float): Bandpass corner frequencies for STA/LTA (Hz). Defaults 2 / 15 Hz.
    t_sta, t_lta (float): STA and LTA window lengths in seconds. Defaults 0.2 / 2.5 s.
    trigger_on, trigger_off (float): STA/LTA ratio thresholds. Defaults 4 / 2.
    trig_horz, trig_vert (float): STA/LTA coincidence thresholds for horizontal / vertical channels. Defaults 6 / 10.
    seisbenchmodel (str or None): Full path to a SeisBench model checkpoint (only used when ``machine_picker='Seisbench'``).
    use_multiprocessing (bool): Use multiprocessing for STA/LTA computation across stations. Default ``False``.

**Returns:** None

**Example:**

.. code-block:: python

    from datetime import datetime
    detection_continuous(dirname='20230101', project_folder='/data/myproject',
                         project_code='IDAHO', machine=True, machine_picker='GPD',
                         single_date=datetime(2023, 1, 1))

----

.. rubric:: queue_sta_lta

Runs the STA/LTA trigger on all stations listed in an input file and writes picks to an output file. Supports optional multiprocessing.

**Args:**

    infile (str): Path to the ``dayfile.in`` station file list.
    outfile (str): Path where pick output is written.
    dirname (str): Data directory (used for context only).
    filtmin (float): High-pass filter corner in Hz. Default 2.
    filtmax (float): Low-pass filter corner in Hz. Default 15.
    t_sta (float): Short-term average window in seconds. Default 0.2.
    t_lta (float): Long-term average window in seconds. Default 2.5.
    trigger_on (float): STA/LTA on-threshold. Default 4.
    trigger_off (float): STA/LTA off-threshold. Default 2.
    trig_horz (float): Horizontal coincidence threshold. Default 6.
    trig_vert (float): Vertical coincidence threshold. Default 10.
    use_multiprocessing (bool): Use a multiprocessing Pool. Default ``False``.

**Returns:** None

----

Association & Catalog Building
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: association_continuous

Associates ML/STA-LTA picks from one day into candidate earthquake events using the PhaseAssoc 1-D associator. Builds travel-time tables, loads picks into an SQLite database, and runs the association algorithm. Results are stored in per-day SQLite databases consumed by ``combine_associated``.

**Args:**

    dirname (str): Sub-directory for the day being processed.
    project_folder (str): Root project directory.
    project_code (str): Short project identifier.
    maxdist (float): Maximum station search distance in km for travel-time table.
    maxkm (float): Maximum epicentral distance for event association in km.
    single_date (datetime.datetime): Day to associate.
    local (bool): Read station inventory locally (``True``) or from FDSN (``False``).
    nsta_declare (int): Minimum number of stations needed to declare an event. Default 3.
    delta_distance (int): Travel-time table distance increment in km. Default 1.
    assoc_ot_uncert (float): Origin-time uncertainty window in seconds. Default 10.
    cutoff_outlier (float): Distance cutoff for outlier removal in km. Default 40.
    loc_uncert_thresh (float): Location uncertainty threshold in degrees. Default 0.5.
    machine (bool): ``True`` if ML picks were used; ``False`` for STA/LTA.
    machine_picker (str or None): Picker name used for detection (must match the ``detection_continuous`` run).
    latitude, longitude (float): Used when ``local=False``.
    max_radius (float): FDSN query radius in degrees (only used when ``local=False``).
    model (str or None): TauPy velocity model name.
    delete_assoc (bool): Delete an existing association database before running. Default ``False``.

**Returns:** None

----

.. rubric:: combine_associated

Reads all per-day association SQLite databases in a project folder, merges them into a single ObsPy ``Catalog``, optionally re-locates events with Hyp2000, and returns the catalog.

**Args:**

    project_folder (str): Root project directory.
    project_code (str): Short project identifier.
    catalog_year (bool): Limit to a specific year when ``True``.
    year (int or None): Four-digit year (used when ``catalog_year=True``).
    hypoflag (bool): Run Hyp2000 location on events in the combined catalog.
    eventmode (bool): Each sub-directory is a single event (vs. one day per directory).
    daymode (bool): Restrict to a single day when ``True``.
    single_date (datetime.datetime or None): Day to restrict to (used when ``daymode=True``).
    machine_picker (str or None): Picker name; used to look up database filenames.

**Returns:** ``(obspy.Catalog, pandas.DataFrame)`` – the combined event catalog and a summary DataFrame.

----

.. rubric:: detection_association_event

One-shot pipeline for a single candidate event: optionally downloads radial waveforms, runs ML picking, associates picks, and returns the event catalog. Used by the quasi-realtime module.

**Args:**

    project_folder (str): Root project directory.
    project_code (str): Short project identifier.
    maxdist (float): Maximum station search distance in km.
    maxkm (float): Maximum epicentral distance for association in km.
    local (bool): Read station inventory locally.
    machine (bool): Use ML picker.
    machine_picker (str or None): Picker name.
    fullpath_python (str or None): Path to Python interpreter.
    approxorigintime (str or UTCDateTime): Approximate origin time of the candidate event.
    downloadwaveforms (bool): Download waveforms from FDSN before picking. Default ``True``.
    delta_distance (int): Travel-time table distance increment in km. Default 1.
    latitude, longitude (float): Epicentre hint for waveform download.
    max_radius (float): Download radius in degrees.

**Returns:** None (catalog written to disk)

----

.. rubric:: pytocto_file_quakeml

Converts a PyOcto output file to an ObsPy QuakeML catalog.

**Args:**

    file (str): Path to the PyOcto output file.

**Returns:** ``obspy.Catalog``

----

Magnitude Estimation
~~~~~~~~~~~~~~~~~~~~

.. rubric:: magnitude_quakeml

Computes local magnitude (ML) for each event in a catalog by measuring peak amplitudes on horizontal-component waveforms with instrument response removed. Optionally estimates P-wave polarity and S/P amplitude ratio. Writes the updated catalog to ``project_folder/cat.xml``.

**Args:**

    cat (obspy.Catalog): Input catalog.
    project_folder (str): Root project directory containing per-day waveform sub-directories.
    plot_event (bool): Save a section plot for each event. Default ``False``.
    cutoff_dist (float): Maximum epicentral distance in km to include a station magnitude. Default 200.
    estimate_sp (bool): Estimate S/P amplitude ratio (proxy for focal depth). Default ``False``.
    eventmode (bool): ``True`` if ``project_folder`` is a single-event directory. Default ``False``.
    dirname (str or None): Day string (used in event-mode).

**Returns:** ``obspy.Catalog`` with magnitudes appended.

----

.. rubric:: polarity

Determines the first-motion polarity of a P-wave pick on a single trace.

**Args:**

    tr (obspy.Trace): Seismic trace containing the P arrival.
    pickP (UTCDateTime or None): Time of the P pick.

**Returns:** ``str`` – ``'positive'``, ``'negative'``, or ``'undecided'``.

----

.. rubric:: sp_ratio

Calculates the S-to-P amplitude ratio for a three-component station, which can be used as a rough focal depth indicator.

**Args:**

    st3 (obspy.Stream): Three-component stream with response removed.
    inv (obspy.Inventory): Instrument response inventory.
    pickP (obspy.Pick or None): P-wave pick object.
    all_picks (list): All picks for the event.
    event (obspy.Event): Parent event object.

**Returns:** ``float`` – S/P amplitude ratio.

----

Location
~~~~~~~~

.. rubric:: locate_hyp2000

Re-locates events in a catalog using the Hyp2000 (HYPOINVERSE) program. Generates ``.pha`` phase files and ``run.hyp`` control files, calls the Hyp2000 binary, and reads back the origin solution into the catalog.

**Args:**

    cat (obspy.Catalog): Input catalog with picks.
    project_folder (str): Root project directory.
    vel_model (str or None): Path or name of a ``.crh`` velocity model file. Defaults to the bundled ``standard.crh``.
    fullpath_hyp (str or None): Path to the Hyp2000 binary directory. If ``None``, ``hyp2000`` must be on ``$PATH``.
    daymode (bool): Use day-scoped output files. Default ``False``.
    catalog_year (bool): Use year-scoped output files. Default ``False``.
    year (int or None): Year (used when ``catalog_year=True``).
    single_date (datetime.datetime or None): Day (used when ``daymode=True``).

**Returns:** ``obspy.Catalog`` with new Hyp2000 origin appended to each event.

----

Catalog Utilities
~~~~~~~~~~~~~~~~~

.. rubric:: simple_cat_df

Converts an ObsPy Catalog to a pandas DataFrame with one row per event.

**Args:**

    cat (obspy.Catalog): Input catalog.
    uncertainty (bool): Include location uncertainty columns (``horizontal_error``, ``vertical_error``, ``rms``, ``azimuthal_gap``, ``num_arrivals``). Default ``False``.

**Returns:** ``pandas.DataFrame`` with columns ``origintime``, ``latitude``, ``longitude``, ``depth``, ``magnitude``, ``type``, ``id`` (plus uncertainty columns if requested).

----

.. rubric:: catdf_narrowbounds

Filters a catalog DataFrame to a rectangular geographic box.

**Args:**

    catdf (pandas.DataFrame): DataFrame produced by ``simple_cat_df``.
    lat_a, lat_b (float): Minimum and maximum latitude.
    lon_a, lon_b (float): Minimum and maximum longitude.

**Returns:** Filtered ``pandas.DataFrame``.

----

.. rubric:: reduce_catalog

Filters an ObsPy Catalog to retain only events meeting minimum quality criteria.

**Args:**

    cat (obspy.Catalog): Input catalog.
    num_arr (int or None): Minimum number of arrivals required.
    vert_unc (float or None): Maximum vertical (depth) uncertainty in metres.

**Returns:** ``obspy.Catalog``

----

.. rubric:: duplicate_remove

Removes likely duplicate events from a catalog. Events within *seconds* of another event (sorted chronologically) are treated as duplicates and the later one is removed.

**Args:**

    cat (obspy.Catalog): Input catalog.
    seconds (float): Minimum inter-event time in seconds. Default 5.

**Returns:** ``obspy.Catalog`` with duplicates removed.

----

.. rubric:: fix_picks_catalog

Verifies that the waveform file referenced by each pick actually exists in the project directory. If a pick channel code does not match any file (e.g. ``HHE`` vs ``HH1``), the code is updated to the channel found on disk.

**Args:**

    catalog (obspy.Catalog): Input catalog.
    project_folder (str): Root project directory.
    filename (str or None): If provided, write the corrected catalog to this QuakeML file path.

**Returns:** Corrected ``obspy.Catalog``.

----

.. rubric:: cut_event_waveforms

Cuts waveform windows around each event in a catalog, saves them as miniSEED files in ``project_folder/events/``, and optionally plots each event record section. Run ``fix_picks_catalog`` first to ensure channel codes are correct.

**Args:**

    catalog (obspy.Catalog): Input catalog.
    project_folder (str): Root project directory.
    length (int): Waveform cut length in seconds after the origin time. Default 120.
    filteryes (bool): Apply a 1 Hz high-pass filter before plotting. Default ``True``.
    plotevent (bool): Save a PNG record-section plot for each event. Default ``False``.
    cutall (bool): Also cut all stations without picks and save as a ``_nopicks.mseed`` file. Default ``False``.

**Returns:** ``obspy.Catalog`` (same as input).

----

.. rubric:: make_station_list_csv

Reads all ``dailyinventory.xml`` files found in sub-directories of ``project_folder`` and writes a ``station_list.csv`` file to each sub-directory that does not yet have one.

**Args:**

    project_folder (str): Root project directory.

**Returns:** None

----

.. rubric:: daymode_catalog

Convenience pipeline for a single day: calls ``combine_associated``, ``magnitude_quakeml``, and ``locate_hyp2000`` in sequence, then saves QuakeML and CSV output.

**Args:**

    project_folder (str): Root project directory.
    project_code (str): Short project identifier.
    single_date (datetime.datetime): Day to process.
    machine_picker (str or None): Picker name used during detection.
    fullpath_hyp (str or None): Path to Hyp2000 binary directory.

**Returns:** None

----

.. rubric:: yearmode_catalog

Same as ``daymode_catalog`` but spans a full calendar year.

**Args:**

    project_folder (str): Root project directory.
    project_code (str): Short project identifier.
    year (int): Four-digit year.
    machine_picker (str or None): Picker name.
    fullpath_hyp (str or None): Path to Hyp2000 binary.

**Returns:** None

----

Format Conversion
~~~~~~~~~~~~~~~~~

.. rubric:: single_event_xml

Writes each event in a catalog to an individual QuakeML file in ``project_folder/<format>/``.

**Args:**

    catalog (obspy.Catalog): Input catalog.
    project_folder (str): Root project directory.
    format (str): Output format string passed to ObsPy. Default ``'QUAKEML'``.

**Returns:** None

----

.. rubric:: daily_catalog_xml

Writes one QuakeML file per calendar day in ``project_folder/<format>/``, named ``YYYYMMDD.xml``.

**Args:**

    catalog (obspy.Catalog): Input catalog.
    project_folder (str): Root project directory. Default ``'.'``.
    format (str): Output format. Default ``'QUAKEML'``.

**Returns:** None

----

.. rubric:: join_all_xml

Reads all ``*.xml`` files in a folder and merges them into a single catalog file.

**Args:**

    xml_folder (str): Path to the folder containing individual QuakeML files.
    filename (str): Output filename (without extension; ``.xml`` is appended).
    format (str): Output format. Default ``'QUAKEML'``.

**Returns:** None

----

.. rubric:: quakeml_to_hypodd

Converts an ObsPy Catalog to HypoDD input files (``phase.dat`` and ``station.dat``). Optionally downloads station metadata from FDSN.

**Args:**

    cat (obspy.Catalog): Input catalog.
    download_station_metadata (bool): Download station coordinates from FDSN. Default ``True``.
    project_folder (str): Root project directory.
    project_code (str): Short project identifier; used as a filename prefix.

**Returns:** None. Writes ``<project_code>.pha`` and ``<project_code>station.dat``.

----

.. rubric:: quakeml_to_growclust

Converts HypoDD-format phase and cross-correlation files to GrowClust input format. Filters cross-correlation pairs by ``cc_threshold`` and writes ``evlist.txt``, ``dtcc.txt``, and ``stlist.txt``.

.. note::
   Run ``quakeml_to_hypodd`` first to generate the phase and station files.

**Args:**

    project_folder (str): Root project directory.
    phase_file (str): HypoDD phase file name. Default ``'phase.dat'``.
    station_file (str): HypoDD station file name. Default ``'station.dat'``.
    dt_file (str): Cross-correlation differential time file name. Default ``'dt.cc'``.
    cc_threshold (float): Minimum cross-correlation coefficient to retain a pair. Default 0.6.
    download_station_metadata (bool): Download station coordinates from FDSN for the GrowClust station list. Default ``False``.

**Returns:** None

----

.. rubric:: quakeml_to_hdf5

Converts waveform data associated with each event in a catalog to STEAD-compatible HDF5 format for ML model re-training. Optionally writes a CSV summary.

**Args:**

    cat (obspy.Catalog): Input catalog.
    project_folder (str): Root project directory containing per-day waveform sub-directories.
    makecsv (bool): Write a CSV event summary to ``project_folder/``. Default ``True``.

**Returns:** None. Writes ``project_folder/merge.hdf5`` (and optionally a CSV).

----

Plotting
~~~~~~~~

.. rubric:: plot_map_catalog

Plots a map of earthquake epicentres from a catalog using Basemap, coloured by event index (time sequence). Saves to ``hypo_map.png``.

**Args:**

    cat (obspy.Catalog): Input catalog.
    filename (str or None): Output PNG filename (without extension). Defaults to ``'hypo_map'``.
    points (bool): Plot as small dots instead of magnitude-scaled circles. Default ``False``.

**Returns:** None

----

.. rubric:: plot_gr_freq_catalog

Plots earthquake frequency-of-occurrence (monthly and daily) and Gutenberg-Richter magnitude-frequency distribution. Saves ``freq_plot.png`` and ``gr_plot.png``.

**Args:**

    cat (obspy.Catalog): Input catalog.
    min_mag (float): Minimum magnitude for the completeness-threshold subplot. Default 2.

**Returns:** None

----

.. rubric:: plot_hypodd_catalog

Reads a HypoDD ``reloc`` output file and plots relocated epicentres on a Basemap.

**Args:**

    file (str): Path to the HypoDD ``reloc`` file.
    fancy_plot (bool): If ``True``, colour-code events by time and scale symbols by magnitude. Default ``False``.

**Returns:** None. Saves ``hypoDDmap.png``.

----

-----------------------------------
easyQuake.gpd_predict.gpd_predict
-----------------------------------

.. rubric:: sliding_window

Calculate a sliding window over a signal array.

**Args:**

    data (numpy.ndarray): Array to slide over.
    size (int): Window length in samples.
    stepsize (int): Step between successive windows. Default 1.
    axis (int): Axis along which to slide. Defaults to the last axis.
    copy (bool): Return a copy of the strided array to avoid side-effects. Default ``True``.

**Returns:** ``numpy.ndarray`` – matrix where each row (in the last dimension) is one window instance.

----

------------------------------------------
easyQuake.phasenet.phasenet_predict
------------------------------------------

.. rubric:: read_args

Parses command-line arguments for the PhaseNet predictor. Accepts an optional ``argv`` list for programmatic use.

**Args:**

    argv (list or None): Argument list. If ``None``, reads from ``sys.argv``. Default ``None``.

**Returns:** ``argparse.Namespace``

----

.. rubric:: pred_fn

Runs PhaseNet inference on the provided data reader and writes phase-pick results to disk.

**Args:**

    args: Parsed arguments (from ``read_args``).
    data_reader: ``DataReader`` object wrapping the input miniSEED files.
    figure_dir (str or None): Directory to write probability plots. Default ``None``.
    prob_dir (str or None): Directory to write probability arrays. Default ``None``.
    log_dir (str or None): Directory for TensorFlow logs. Default ``None``.

**Returns:** 0 on success.

----

-------------------------------
easyQuake.seisbench.run_seisbench
-------------------------------

.. rubric:: main

Command-line entry point for SeisBench-based phase picking. Reads a ``dayfile.in`` station list, applies a trained SeisBench model to each station's waveforms, and writes picks to an output file.

**Command-line flags:**

    ``-I``: Input ``dayfile.in`` file (required).
    ``-O``: Output picks file (required).
    ``-M``: Full path to the SeisBench model checkpoint (required).

**Returns:** None

----

-----------------------------------------
easyQuake.EQTransformer.mseed_predictor
-----------------------------------------

.. rubric:: main

Command-line entry point for EQTransformer phase picking. Reads a ``dayfile.in`` station list, loads the EQTransformer model (preferring a ``.keras`` format if available), and writes picks to an output file.

**Command-line flags:**

    ``-I``: Input ``dayfile.in`` file (required).
    ``-O``: Output picks file (required).
    ``-F``: Path to the EQTransformer model directory (required).

**Returns:** None

