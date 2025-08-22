#!/usr/bin/env python3
"""
Data reader wrapper for PhaseNet TF2/Keras 3 compatibility.
"""

import os
import sys
import numpy as np

# Add the phasenet directory to the path for imports
phasenet_dir = os.path.dirname(__file__)
if phasenet_dir not in sys.path:
    sys.path.insert(0, phasenet_dir)

from data_reader import DataReader_pred, DataConfig


class DataReaderWrapper:
    """
    Wrapper to make the PhaseNet data reader compatible with our Keras 3 predictor.
    """
    
    def __init__(self, args):
        """Initialize the data reader wrapper."""
        self.args = args
        self.config = DataConfig()
        self.data_reader = None
        
        # Update config with args
        if hasattr(args, 'nt'):
            self.config.nt = args.nt
        if hasattr(args, 'dt'):
            self.config.dt = args.dt
        if hasattr(args, 'dtype'):
            self.config.dtype = args.dtype
        
        # Initialize the actual data reader
        try:
            self.data_reader = DataReader_pred(
                format=getattr(args, 'format', 'mseed'),
                data_list=args.data_list,
                data_dir=getattr(args, 'data_dir', './'),
                amplitude=True,
                config=self.config
            )
            self.data = self.data_reader.data_list
            self.num_data = len(self.data)
        except Exception as e:
            print(f"Failed to create data reader: {e}")
            # Fallback - read the data list file directly
            self.data = []
            self.num_data = 0
            if hasattr(args, 'data_list') and os.path.exists(args.data_list):
                with open(args.data_list, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            # Store all three components
                            self.data.append(parts[2])  # Use Z component as primary
                self.num_data = len(self.data)
    
    def get_next_batch(self, batch_size=1):
        """Get next batch of data (for compatibility)."""
        if self.data_reader and hasattr(self.data_reader, '__getitem__'):
            # Use the actual data reader
            batch_data = []
            batch_names = []
            
            for i in range(min(batch_size, self.num_data)):
                try:
                    sample, name = self.data_reader[i]
                    batch_data.append(sample)
                    batch_names.append(name)
                except Exception as e:
                    print(f"Error reading sample {i}: {e}")
                    continue
            
            if batch_data:
                return np.array(batch_data), batch_names
            else:
                return None, None
        else:
            return None, None


def create_data_reader_for_tf2(args):
    """
    Create a data reader that works with both the original PhaseNet and our TF2 implementation.
    """
    return DataReaderWrapper(args)


def read_mseed_files_directly(file_paths, target_length=3000, target_rate=100):
    """
    Read MSEED files directly and return processed data.
    """
    import obspy
    
    processed_data = []
    file_names = []
    
    for file_path in file_paths:
        try:
            # Check if this is a single file or part of a 3-component set
            if isinstance(file_path, str):
                # Single file - need to find the other components
                base_path = file_path.replace('.Z.', '.{}.').replace('.EHZ.', '.EH{}.').replace('.BHZ.', '.BH{}.')
                component_files = []
                
                for comp in ['E', 'N', 'Z']:
                    comp_file = base_path.format(comp)
                    if os.path.exists(comp_file):
                        component_files.append(comp_file)
                
                if len(component_files) >= 3:
                    # Read all three components
                    traces = []
                    for comp_file in component_files[:3]:
                        st = obspy.read(comp_file)
                        if len(st) > 0:
                            traces.append(st[0])
                    
                    if len(traces) == 3:
                        # Process the 3-component data
                        waveform_data = process_three_component_traces(traces, target_length, target_rate)
                        if waveform_data is not None:
                            processed_data.append(waveform_data)
                            file_names.append(os.path.basename(file_path))
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return processed_data, file_names


def process_three_component_traces(traces, target_length=3000, target_rate=100):
    """
    Process three-component traces into the format expected by PhaseNet.
    """
    try:
        # Ensure same sampling rate
        for trace in traces:
            if trace.stats.sampling_rate != target_rate:
                trace.resample(target_rate)
        
        # Find common time window
        start_times = [tr.stats.starttime for tr in traces]
        end_times = [tr.stats.endtime for tr in traces]
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        if common_end <= common_start:
            return None
        
        # Trim to common window
        for trace in traces:
            trace.trim(common_start, common_end)
        
        # Ensure same length
        min_length = min(len(tr.data) for tr in traces)
        for trace in traces:
            trace.data = trace.data[:min_length]
        
        # Preprocess
        for trace in traces:
            trace.detrend('linear')
            trace.filter('bandpass', freqmin=1.0, freqmax=45.0, zerophase=True)
        
        # Create waveform array (time, channels)
        waveform = np.column_stack([tr.data for tr in traces])
        
        # Pad or truncate to target length
        if waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            waveform = np.pad(waveform, ((0, padding), (0, 0)), mode='constant')
        elif waveform.shape[0] > target_length:
            waveform = waveform[:target_length]
        
        # Normalize each channel
        for i in range(waveform.shape[1]):
            channel_data = waveform[:, i]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            if std_val > 0:
                waveform[:, i] = (channel_data - mean_val) / std_val
            else:
                waveform[:, i] = channel_data - mean_val
        
        # Add batch dimension and convert to correct format
        return np.expand_dims(waveform, axis=0).astype(np.float32)
        
    except Exception as e:
        print(f"Error processing traces: {e}")
        return None


if __name__ == "__main__":
    # Test the data reader wrapper
    import argparse
    
    # Create test args
    args = argparse.Namespace()
    args.data_list = "/tmp/phasenet_test/dayfile.in"
    args.format = "mseed"
    args.data_dir = "./"
    args.nt = 3000
    args.dt = 0.01
    args.dtype = "float32"
    
    # Test the wrapper
    wrapper = DataReaderWrapper(args)
    print(f"Data reader created with {wrapper.num_data} files")
    print(f"Files: {wrapper.data[:3]}")
    
    # Test reading data
    if wrapper.num_data > 0:
        batch_data, batch_names = wrapper.get_next_batch(1)
        if batch_data is not None:
            print(f"Successfully read batch: {batch_data.shape}")
        else:
            print("Failed to read batch, trying direct MSEED reading...")
            processed_data, file_names = read_mseed_files_directly(wrapper.data[:1])
            if processed_data:
                print(f"Direct reading successful: {processed_data[0].shape}")
            else:
                print("Direct reading also failed")
