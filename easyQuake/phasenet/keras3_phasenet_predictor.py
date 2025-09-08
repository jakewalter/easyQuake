#!/usr/bin/env python3
"""
PhaseNet TF2/Keras 3 predictor using the existing converted model.
This implementation uses the working phasenet_tf2_converted.keras model.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import obspy
from obspy import UTCDateTime
import logging


def load_phasenet_tf2_converted():
    """Load the existing converted PhaseNet TF2 model."""
    model_path = os.path.join(os.path.dirname(__file__), "phasenet_tf2_converted.keras")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logging.info(f"Loaded PhaseNet TF2 model: {model.count_params()} parameters")
        return model
    except Exception as e:
        logging.error(f"Failed to load PhaseNet TF2 model: {e}")
        return None


def preprocess_waveform(waveform, sampling_rate=100, target_length=3000):
    """
    Preprocess waveform data for PhaseNet prediction.
    
    Args:
        waveform: numpy array of shape (channels, samples) or (samples, channels)
        sampling_rate: sampling rate in Hz
        target_length: target length in samples
    
    Returns:
        Preprocessed waveform of shape (1, target_length, channels)
    """
    # Ensure correct shape (samples, channels)
    if waveform.shape[0] == 3 and waveform.shape[1] > 3:
        waveform = waveform.T  # Transpose from (channels, samples) to (samples, channels)
    
    # Pad or truncate to target length
    if waveform.shape[0] < target_length:
        # Pad with zeros
        padding = target_length - waveform.shape[0]
        waveform = np.pad(waveform, ((0, padding), (0, 0)), mode='constant')
    elif waveform.shape[0] > target_length:
        # Truncate
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
    
    # Add batch dimension
    return np.expand_dims(waveform, axis=0).astype(np.float32)


def predict_phases(model, waveform, sampling_rate=100, p_threshold=0.5, s_threshold=0.5):
    """
    Predict P and S phases using PhaseNet model.
    
    Args:
        model: Loaded PhaseNet model
        waveform: Preprocessed waveform data
        sampling_rate: Sampling rate in Hz
        p_threshold: Threshold for P-phase detection
        s_threshold: Threshold for S-phase detection
    
    Returns:
        List of picks with times and probabilities
    """
    # Get predictions
    predictions = model.predict(waveform, verbose=0)
    
    # Extract probability arrays
    noise_prob = predictions[0, :, 0]
    p_prob = predictions[0, :, 1]
    s_prob = predictions[0, :, 2]
    
    picks = []
    
    # Find P-phase picks
    p_indices = find_peaks(p_prob, threshold=p_threshold)
    for idx in p_indices:
        picks.append({
            'phase': 'P',
            'sample': idx,
            'time': idx / sampling_rate,
            'probability': float(p_prob[idx])
        })
    
    # Find S-phase picks
    s_indices = find_peaks(s_prob, threshold=s_threshold)
    for idx in s_indices:
        picks.append({
            'phase': 'S',
            'sample': idx,
            'time': idx / sampling_rate,
            'probability': float(s_prob[idx])
        })
    
    return picks


def find_peaks(probability_array, threshold=0.3, min_distance=50):
    """
    Find peaks in probability array above threshold.
    
    Args:
        probability_array: 1D array of probabilities
        threshold: Minimum probability threshold
        min_distance: Minimum distance between peaks in samples
    
    Returns:
        List of peak indices
    """
    peaks = []
    
    for i in range(1, len(probability_array) - 1):
        if (probability_array[i] > threshold and
            probability_array[i] > probability_array[i-1] and
            probability_array[i] > probability_array[i+1]):
            
            # Check minimum distance from previous peaks
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    
    return peaks


def process_mseed_file(file_path, model, component_order=['E', 'N', 'Z']):
    """
    Process a single MSEED file and generate picks.
    
    Args:
        file_path: Path to MSEED file
        model: Loaded PhaseNet model
        component_order: Expected component order
    
    Returns:
        List of picks
    """
    try:
        # Read MSEED file
        st = obspy.read(file_path)
        
        if len(st) == 0:
            return []
        
        # Group traces by station
        stations = {}
        for trace in st:
            net_sta = f"{trace.stats.network}.{trace.stats.station}"
            if net_sta not in stations:
                stations[net_sta] = {}
            
            # Get component from channel code
            component = trace.stats.channel[-1]
            stations[net_sta][component] = trace
        
        all_picks = []
        
        # Process each station
        for net_sta, components in stations.items():
            station_picks = process_station(net_sta, components, model, component_order)
            all_picks.extend(station_picks)
        
        return all_picks
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return []


def process_station(net_sta, components, model, component_order=['E', 'N', 'Z']):
    """
    Process 3-component data for a single station.
    """
    try:
        # Get traces for each component
        traces = []
        for comp in component_order:
            if comp in components:
                traces.append(components[comp])
            else:
                # Try common variations
                variations = {'E': ['E', '1'], 'N': ['N', '2'], 'Z': ['Z']}
                found = False
                for var in variations.get(comp, [comp]):
                    if var in components:
                        traces.append(components[var])
                        found = True
                        break
                
                if not found:
                    logging.warning(f"Missing {comp} component for {net_sta}")
                    return []
        
        if len(traces) != 3:
            return []
        
        # Preprocess traces
        traces = preprocess_traces(traces)
        if traces is None:
            return []
        
        # Create waveform array (samples, channels)
        waveform = np.column_stack([tr.data for tr in traces])
        
        # Process in windows
        picks = process_waveform_windows(waveform, model, traces[0], net_sta)
        
        return picks
    
    except Exception as e:
        logging.error(f"Error processing station {net_sta}: {e}")
        return []


def preprocess_traces(traces):
    """
    Preprocess traces for consistency.
    """
    try:
        # Ensure same sampling rate
        sample_rates = [tr.stats.sampling_rate for tr in traces]
        if len(set(sample_rates)) > 1:
            target_rate = 100.0
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
        
        # Basic preprocessing
        for trace in traces:
            trace.detrend('linear')
            trace.filter('bandpass', freqmin=1.0, freqmax=45.0, zerophase=True)
        
        return traces
    
    except Exception as e:
        logging.error(f"Error preprocessing traces: {e}")
        return None


def process_waveform_windows(waveform, model, reference_trace, station_id):
    """
    Process waveform in overlapping windows.
    """
    picks = []
    window_length = 3000  # samples
    overlap = 1500  # samples
    step = window_length - overlap
    
    sampling_rate = reference_trace.stats.sampling_rate
    start_time = reference_trace.stats.starttime
    
    for i in range(0, len(waveform) - window_length + 1, step):
        window_data = waveform[i:i + window_length]
        
        if window_data.shape[0] != window_length:
            continue
        
        # Preprocess window
        processed_window = preprocess_waveform(window_data.T, sampling_rate, window_length)
        
        # Predict
        window_picks = predict_phases(model, processed_window, sampling_rate, p_threshold=0.5, s_threshold=0.5)
        
        # Convert to absolute times and add station info
        for pick in window_picks:
            absolute_time = start_time + (i + pick['sample']) / sampling_rate
            picks.append({
                'station': station_id,
                'phase': pick['phase'],
                'time': absolute_time,
                'probability': pick['probability'],
                'sample_offset': i + pick['sample']
            })
    
    return picks


def main_phasenet_tf2_predict(args, data_reader, log_dir=None):
    """
    Main prediction function using the converted PhaseNet TF2 model.
    """
    logging.info("Using converted PhaseNet TF2 model...")
    
    # Load model
    model = load_phasenet_tf2_converted()
    if model is None:
        logging.error("Failed to load PhaseNet TF2 model")
        return []
    
    all_picks = []
    
    # Handle different data reader types
    try:
        if hasattr(data_reader, 'data'):
            # Standard data reader with .data attribute
            file_paths = data_reader.data
            logging.info(f"Using data reader with {len(file_paths)} files")
        elif hasattr(data_reader, 'data_list'):
            # PhaseNet data reader
            file_paths = data_reader.data_list
            logging.info(f"Using PhaseNet data reader with {len(file_paths)} files")
        else:
            # Try to get files from dayfile.in
            file_paths = []
            if hasattr(args, 'data_list') and os.path.exists(args.data_list):
                with open(args.data_list, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            # Process all three components together
                            file_paths.append(parts)  # Store as [E, N, Z] components
                logging.info(f"Loaded {len(file_paths)} 3-component sets from dayfile")
            else:
                logging.warning("No data list file found")
                return []
        
        # Process each file or file set
        for file_item in file_paths:
            if isinstance(file_item, list) and len(file_item) >= 3:
                # 3-component set
                picks = process_three_component_set(file_item, model)
                if picks:
                    all_picks.extend(picks)
                    logging.info(f"Found {len(picks)} picks in 3-component set")
            elif isinstance(file_item, str):
                # Single file - try to find components
                picks = process_mseed_file(file_item, model)
                if picks:
                    all_picks.extend(picks)
                    logging.info(f"Found {len(picks)} picks in {os.path.basename(file_item)}")
    
    except Exception as e:
        logging.error(f"Error in main prediction: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    # Convert to expected format for easyQuake
    if all_picks:
        # Convert to list format expected by the framework
        formatted_picks = []
        for pick in all_picks:
            # Calculate relative time from start of day
            pick_time = pick['time']
            if hasattr(pick_time, 'timestamp'):
                # Convert UTCDateTime to seconds since start of trace
                phase_time = pick.get('sample_offset', 0) / 100.0  # Assume 100 Hz
            else:
                phase_time = float(pick_time)
            
            # Create pick dictionary in expected format
            formatted_picks.append({
                'file_name': pick.get('file_name', 'unknown'),
                'station_id': pick['station'].split('.')[-1] if '.' in pick['station'] else pick['station'],
                'begin_time': 0,
                'phase_time': phase_time,
                'phase_score': pick['probability'],
                'phase_type': pick['phase'],
                'phase_index': int(phase_time * 100)  # Sample index
            })
        
        logging.info(f"Total picks found: {len(formatted_picks)}")
        return formatted_picks
    else:
        logging.warning("No picks found")
        return []


def process_three_component_set(file_list, model):
    """
    Process a 3-component file set [E, N, Z] and generate picks.
    """
    try:
        import obspy
        
        # Read the three components
        traces = []
        for file_path in file_list[:3]:  # E, N, Z
            if os.path.exists(file_path):
                st = obspy.read(file_path)
                if len(st) > 0:
                    traces.append(st[0])
        
        if len(traces) != 3:
            logging.warning(f"Could not read all 3 components from {file_list}")
            return []
        
        # Preprocess traces
        traces = preprocess_traces(traces)
        if traces is None:
            return []
        
        # Get station name from first trace
        station_name = f"{traces[0].stats.network}.{traces[0].stats.station}"
        
        # Create waveform array (samples, channels)
        waveform = np.column_stack([tr.data for tr in traces])
        
        # Process in windows
        picks = process_waveform_windows(waveform, model, traces[0], station_name)
        
        # Add file name to picks
        for pick in picks:
            pick['file_name'] = os.path.basename(file_list[2])  # Use Z component file name
        
        return picks
    
    except Exception as e:
        logging.error(f"Error processing 3-component set {file_list}: {e}")
        return []


if __name__ == "__main__":
    # Test with available data
    test_files = [
        '/home/jwalter/easyQuake/easyQuake/gpd_predict/testdata/E.mseed',
        '/home/jwalter/easyQuake/easyQuake/gpd_predict/testdata/N.mseed',
        '/home/jwalter/easyQuake/easyQuake/gpd_predict/testdata/Z.mseed'
    ]
    
    # Create a simple test
    if all(os.path.exists(f) for f in test_files):
        model = load_phasenet_tf2_converted()
        if model:
            # Load test data
            traces = []
            for file_path in test_files:
                trace = obspy.read(file_path)[0]
                traces.append(trace)
            
            # Create waveform array
            waveform = np.column_stack([tr.data for tr in traces])
            
            # Preprocess
            processed = preprocess_waveform(waveform.T, 100, 3000)
            
            # Predict
            picks = predict_phases(model, processed, 100, p_threshold=0.1, s_threshold=0.1)
            
            print(f"Test prediction: {len(picks)} picks found")
            for pick in picks[:10]:
                print(f"  {pick}")
    else:
        print("Test files not found")
