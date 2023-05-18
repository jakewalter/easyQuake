#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:41:19 2023

@author: jwalter
"""

#To run loaded models
from pathlib import Path
import easyQuake.seisbench as sbm

from obspy import Inventory, read_inventory
import glob

import traceback
from multiprocessing import cpu_count
import torch

import argparse as ap
import os
from obspy import Stream, read
import numpy as np
def main():
    #os.system("gpd_predict -V -P -I %s -O %s -F %s" % (infile, outfile, pathgpd))

    parser = ap.ArgumentParser(
    prog='run_seisbench.py',
    description='Script to run seisbench within easyQuake')
    parser.add_argument(
        '-I',
        type=str,
        default=None,
        help='Input file')
    parser.add_argument(
        '-O',
        type=str,
        default=None,
        help='Output file')
    parser.add_argument(
        '-M',
        type=str,
        default=None,
        help='Model full path and filename')
    parser.add_argument(
        '-F',
        type=str,
        default=None,
        help='path where GPD lives')
    
    args = parser.parse_args()
    filter_data = True
    decimate_data = True
    freq_min = 3.0
    freq_max = 20.0


    # Replace with the actual path to the directory containing the models
    #model_directory = Path('/home/lmho/train_test_models/seisbench_models/gpd')
    
    # Specify the model to load
    #base_name = 'tam_mf'
    
    # Path to the model
    #model_path = model_directory / base_name
    model_path = args.M

    # Load the model
    version_str = '1'  # Specify the version string of the model you want to load
    loaded_model = sbm.GPD.load(model_path, version_str=version_str)
    
    loaded_model.cuda()



    # # Set the path to the parent directory containing the subdirectories with the streams
    # #parent_directory = Path('/data/tamnet/ml')
    # parent_directory = Path('/home/lmho/dummy_data')
    # #project_folder
    
    # # Define the range of dates
    # start_date = 20150101
    # end_date = 20150102
    
    # # Create a list of directories to iterate through
    # dirs_to_process = [parent_directory / str(date) for date in range(start_date, end_date + 1) if (parent_directory / str(date)).is_dir()]


    # # Iterate through the directories in the specified date range using tqdm to show progress
    # for directory in tqdm(dirs_to_process, desc="Processing directories"):
    #     # Load and group all mseed files in the directory by station
    #     streams = defaultdict(lambda: defaultdict(list))
    #     #trace_id_channel_mapping = {}
    #     for mseed_file in directory.glob('*.mseed'):
    #         stream = read(str(mseed_file))
    #         station_code = stream[0].stats.station
    #         channel = stream[0].stats.channel
    #         location_code = stream[0].stats.location if stream[0].stats.location else ""
    #         trace_id = f"{stream[0].stats.network}.{station_code}.{location_code}.{channel}"
    #         if channel.startswith("BH"):
    #             streams[station_code][channel].append(stream)
    #             #trace_id = stream[0].stats.network + '.' + station_code + '.'
    #             #trace_id_channel_mapping[trace_id] = channel

    # # Create a database session
    # db_assoc = 'sqlite:///' + str(directory) + '/1dassociator_seisbench_tamnnet.db'
    # engine_assoc = create_engine(db_assoc, echo=False, connect_args={'check_same_thread': False})
    # tables1D.Base.metadata.create_all(engine_assoc)
    # Session = sessionmaker(bind=engine_assoc)
    # session = Session()

    # # Load the inventory
    # inv = Inventory()
    # dir1a = glob.glob(str(directory)+'/dailyinventory.xml') + glob.glob(str(directory)+'/??.*.xml')
    # for file1 in dir1a:
    #     inv1a = read_inventory(file1)
    #     inv.networks.extend(inv1a)
            
    # Process the stream components (BHE, BHN, BHZ) for each station
    fdir = []

    with open(args.I) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)
    
    ofile = open(args.O, 'w')
    
    for i in range(nsta):
        try:
            if not os.path.isfile(fdir[i][0]):
                print("%s doesn't exist, skipping" % fdir[i][0])
                continue
            if not os.path.isfile(fdir[i][1]):
                print("%s doesn't exist, skipping" % fdir[i][1])
                continue
            if not os.path.isfile(fdir[i][2]):
                print("%s doesn't exist, skipping" % fdir[i][2])
                continue
            st = Stream()
            st += read(fdir[i][0])
            st += read(fdir[i][1])
            st += read(fdir[i][2])
            #st.resample(100)
            st.detrend(type='linear')
            if filter_data:
                st.filter(type='bandpass', freqmin=freq_min, freqmax=freq_max)
            if decimate_data:
                st.resample(100.0)
            st.merge(fill_value='interpolate')
            print(st)
            for tr in st:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled()
            chan = st[0].stats.channel
            sr = st[0].stats.sampling_rate
            dt = st[0].stats.delta
            net = st[0].stats.network
            sta = st[0].stats.station
            chan = st[0].stats.channel
            latest_start = np.max([x.stats.starttime for x in st])
            earliest_stop = np.min([x.stats.endtime for x in st])
            if (earliest_stop>latest_start):
                st.trim(latest_start, earliest_stop)
            #################### detect
            detections = loaded_model.classify(st)
            # Print the number of picks made by GPD
            print(f"Number of picks made by GPD: {len(detections)}")
            
            # Create the output file with the unique name based on the date
            # date_str = directory.name
            # output_file_path = directory / f"seisbench_gpd_{date_str}.out"
            
            # Create the output file
            #with open(output_file_path, "a") as output_file:
                # Write header line
                #output_file.write("trace_id, start_time, end_time, peak_time, peak_value, phase\n")
            
                # Iterate through the detections and write each one to the output file
            for detection in detections:
                # Extract the detection attributes
                trace_id1 = detection.trace_id.split('.')[:-1]
                #start_time = detection.start_time[0]
                #end_time = detection.end_time
                peak_time = detection.peak_time
                #peak_value = detection.peak_value
                phase = detection.phase
                #channel = trace_id_channel_mapping[trace_id]
                channel = None
                # for tr in st:
                #     if tr.id.startswith(detection.trace_id):
                #         channel = tr.stats.channel
                #         break

                # if channel is None:
                #     print(f"Channel not found for trace_id: {trace_id1}")
                #     continue


                # Assign the channel based on the phase
                if phase == "P":
                    channel = st[0].stats.channel[0:2]+'Z'
                if phase == "S":
                    st2 = st.select(channel='HHE')
                    if len(st2) == 1:
                        channel = st[0].stats.channel[0:2]+'E'
                    else:
                        channel = st[0].stats.channel[0:2]+'2'
                #     if "BHN" in [tr.stats.channel for tr in st if tr.id.startswith(trace_id)]:
                #         channel = "BHN"
                #     else:
                #         channel = "BHE"

                # Combine trace_id with the channel
                #trace_id_with_channel = f"{trace_id}{channel}"

                # Format the output line
                #ofile.write("%s %s %s S %s\n" % (net, sta, chan_pick_s, stamp_pick.isoformat()))
                output_line = f"{' '.join(trace_id1)} {channel} {phase} {peak_time}\n"
        
                # Write the detection attributes to the output file
                ofile.write(output_line)
                #print(output_line.strip())
            
                
                
                
                
                # tt = (np.arange(0, st[0].data.size, n_shift) + n_win) * dt
                # tt_i = np.arange(0, st[0].data.size, n_shift) + n_feat
                # #tr_win = np.zeros((tt.size, n_feat, 3))
                # sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)
                # sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)
                # sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)
                # tr_win = np.zeros((sliding_N.shape[0], n_feat, 3))
                # tr_win[:,:,0] = sliding_N
                # tr_win[:,:,1] = sliding_E
                # tr_win[:,:,2] = sliding_Z
                # tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
                # tt = tt[:tr_win.shape[0]]
                # tt_i = tt_i[:tr_win.shape[0]]
        
                # if args.V:
                #     ts = model.predict(tr_win, verbose=True, batch_size=batch_size)
                # else:
                #     ts = model.predict(tr_win, verbose=False, batch_size=batch_size)
        
                # prob_S = ts[:,1]
                # prob_P = ts[:,0]
                # prob_N = ts[:,2]
        
                # from obspy.signal.trigger import trigger_onset
                # trigs = trigger_onset(prob_P, min_proba, 0.1)
                # p_picks = []
                # s_picks = []
                # for trig in trigs:
                #     if trig[1] == trig[0]:
                #         continue
                #     pick = np.argmax(ts[trig[0]:trig[1], 0])+trig[0]
                #     stamp_pick = st[0].stats.starttime + tt[pick]
                #     chan_pick = st[0].stats.channel[0:2]+'Z'
                #     p_picks.append(stamp_pick)
                #     ofile.write("%s %s %s P %s\n" % (net, sta, chan_pick, stamp_pick.isoformat()))
        
                # trigs = trigger_onset(prob_S, min_proba, 0.1)
                # for trig in trigs:
                #     if trig[1] == trig[0]:
                #         continue
                #     pick = np.argmax(ts[trig[0]:trig[1], 1])+trig[0]
                #     stamp_pick = st[0].stats.starttime + tt[pick]
                #     chan_pick_s = st[0].stats.channel[0:2]+'E'
                #     s_picks.append(stamp_pick)
                #     ofile.write("%s %s %s S %s\n" % (net, sta, chan_pick_s, stamp_pick.isoformat()))
        
                # if plot:
                #     fig = plt.figure(figsize=(8, 12))
                #     ax = []
                #     ax.append(fig.add_subplot(4,1,1))
                #     ax.append(fig.add_subplot(4,1,2,sharex=ax[0],sharey=ax[0]))
                #     ax.append(fig.add_subplot(4,1,3,sharex=ax[0],sharey=ax[0]))
                #     ax.append(fig.add_subplot(4,1,4,sharex=ax[0]))
                #     for i in range(3):
                #         ax[i].plot(np.arange(st[i].data.size)*dt, st[i].data, c='k', \
                #                    lw=0.5)
                #     ax[3].plot(tt, ts[:,0], c='r', lw=0.5)
                #     ax[3].plot(tt, ts[:,1], c='b', lw=0.5)
                #     for p_pick in p_picks:
                #         for i in range(3):
                #             ax[i].axvline(p_pick-st[0].stats.starttime, c='r', lw=0.5)
                #     for s_pick in s_picks:
                #         for i in range(3):
                #             ax[i].axvline(s_pick-st[0].stats.starttime, c='b', lw=0.5)
                #     plt.tight_layout()
                #     plt.show()
        except Exception:
            print('Station issue')
            traceback.print_exc()
            pass
    ofile.close()
    
    

    # for station, channels in streams.items():
    #     components = ['BHE', 'BHN', 'BHZ']
    #     stream_data = []
    #     trace_id_channel_mapping = {}

    #     for comp in components:
    #         if comp in channels:
    #             stream = channels[comp][0]
    #             # Populate trace_id_channel_mapping dictionary
    #             for tr in stream:
    #                 location_code = tr.stats.location if tr.stats.location else ""
    #                 trace_id = f"{tr.stats.network}.{tr.stats.station}.{location_code}.{tr.stats.channel}"
    #                 #trace_id = tr.stats.network + '.' + tr.stats.station + '.'
    #                 trace_id_channel_mapping[trace_id] = tr.stats.channel
    #             # Remove traces with only 1 sample
    #             filtered_traces = []
    #             for tr in stream:
    #                 if tr.stats.npts > 1:
    #                     tr.interpolate(sampling_rate=100.0)
    #                     filtered_traces.append(tr)
    #             #filtered_traces = [tr for tr in stream if tr.stats.npts > 1]  
    #             # If the filtered traces are not empty, append them to stream_data
    #             if len(filtered_traces) > 0:
    #                 filtered_stream = obspy.Stream(traces=filtered_traces)
    #                 stream_data.append(filtered_stream)
    #                 # Add the channel mapping for this component
    #                 #trace_id = stream[0].stats.network + '.' + station_code + '.'
    #                 #trace_id_channel_mapping[trace_id] = comp

    #     # Only process the station if all three components are available
    #     if len(stream_data) == 3:
    #         merged_stream = stream_data[2] + stream_data[1] + stream_data[0]
    #         print(merged_stream)
    #         detections = loaded_model.classify(merged_stream)
    #         # Print the number of picks made by GPD
    #         print(f"Number of picks made by GPD: {len(detections)}")
            
    #         # Create the output file with the unique name based on the date
    #         date_str = directory.name
    #         output_file_path = directory / f"seisbench_gpd_{date_str}.out"
            
    #         # Create the output file
    #         with open(output_file_path, "a") as output_file:
    #             # Write header line
    #             #output_file.write("trace_id, start_time, end_time, peak_time, peak_value, phase\n")
            
    #             # Iterate through the detections and write each one to the output file
    #             for detection in detections:
    #                 # Extract the detection attributes
    #                 trace_id = detection.trace_id
    #                 #start_time = detection.start_time
    #                 #end_time = detection.end_time
    #                 peak_time = detection.peak_time
    #                 peak_value = detection.peak_value
    #                 phase = detection.phase
    #                 #channel = trace_id_channel_mapping[trace_id]
    #                 channel = None
    #                 for tr in merged_stream:
    #                     if tr.id.startswith(trace_id):
    #                         channel = tr.stats.channel
    #                         break

    #                 if channel is None:
    #                     print(f"Channel not found for trace_id: {trace_id}")
    #                     continue


    #                 # Assign the channel based on the phase
    #                 if phase == "P":
    #                     channel = "BHZ"
    #                 elif phase == "S":
    #                     if "BHN" in [tr.stats.channel for tr in merged_stream if tr.id.startswith(trace_id)]:
    #                         channel = "BHN"
    #                     else:
    #                         channel = "BHE"

    #                 # Combine trace_id with the channel
    #                 trace_id_with_channel = f"{trace_id}{channel}"

    #                 # Format the output line
    #                 output_line = f"{' '.join(trace_id_with_channel.split('.'))} {phase} {peak_time} {peak_value}\n"
            
    #                 # Write the detection attributes to the output file
    #                 output_file.write(output_line)
    #                 print(output_line.strip())
    #                 #output_file.write(f"{trace_id}, {start_time}, {end_time}, {peak_time}, {peak_value}, {phase}\n")
    #         # Add the picks to the database
    #         pick_add(dbsession=session, fileinput=output_file_path, inventory=inv)

    #     else:
    #         print(f"Skipping station {station} due to insufficient components.")
            
if __name__ == "__main__":
    main()