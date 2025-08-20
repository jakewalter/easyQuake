#!/usr/bin/env python3
"""Direct test of GPD picker with debug output and plotting."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import obspy
from obspy import UTCDateTime
sys.path.insert(0, '/home/jwalter/easyQuake')

# Import GPD directly
from easyQuake.gpd_predict import process_dayfile


def extract_gpd_probabilities(dayfile_path, base_dir='/home/jwalter/easyQuake/easyQuake/gpd_predict'):
    """
    Extract GPD probabilities for plotting without writing picks to file.
    Returns raw data, timestamps, and probabilities.
    """
    import tensorflow as tf
    
    # Load the model (similar to process_dayfile)
    try:
        import keras
    except ImportError:
        print("Keras not available for GPD model loading")
        return None, None, None, None
    
    # Model loading logic from gpd_predict.py
    _CACHED_MODEL = None
    model = _CACHED_MODEL
    keras_path = os.path.join(base_dir, 'model_pol_new.keras')
    h5_path = os.path.join(base_dir, 'model_pol_legacy.h5')
    
    if model is None:
        try:
            model = keras.models.load_model(keras_path)
            print(f"Loaded model from: {keras_path}")
        except Exception as e_keras:
            print(f"Failed to load .keras model ({keras_path}): {e_keras}")
            if os.path.isfile(h5_path):
                try:
                    model = keras.models.load_model(h5_path)
                    print(f"Loaded legacy HDF5 model from: {h5_path}")
                except Exception as e_h5:
                    print(f"Failed to load fallback HDF5 model ({h5_path}): {e_h5}")
                    return None, None, None, None
            else:
                return None, None, None, None
    
    # Read dayfile
    fdir = []
    with open(dayfile_path, 'r') as f:
        for line in f:
            tmp = line.split()
            if len(tmp) >= 3:
                fdir.append([tmp[0], tmp[1], tmp[2]])
    
    if len(fdir) == 0:
        print("No valid entries in dayfile")
        return None, None, None, None
    
    # Process first station only for plotting
    files = fdir[0]
    print(f"Processing files: {files}")
    
    # Load data
    st = obspy.Stream()
    st += obspy.read(files[0])  # N
    st += obspy.read(files[1])  # E  
    st += obspy.read(files[2])  # Z
    
    print(f"Loaded {len(st)} traces")
    
    # GPD preprocessing (from gpd_predict.py)
    freq_min = 3.0
    freq_max = 20.0
    half_dur = 2.00
    only_dt = 0.01
    n_win = int(half_dur/only_dt)
    n_shift = 10
    batch_size = 1000*3
    
    # Sort traces by channel (Z, N, E) - GPD expects this order
    st.sort(['channel'])
    
    # Filter and process
    st.filter('highpass', freq=freq_min, corners=2, zerophase=True)
    st.filter('lowpass', freq=freq_max, corners=2, zerophase=True)
    st.detrend('demean')
    st.detrend('linear')
    
    # Resample to 100 Hz if needed
    for tr in st:
        if tr.stats.sampling_rate != 100.0:
            tr.resample(100.0)
    
    # Get Z channel for plotting
    z_trace = None
    for tr in st:
        if tr.stats.channel.endswith('Z'):
            z_trace = tr.copy()
            break
    
    if z_trace is None:
        print("No Z channel found")
        return None, None, None, None
    
    # Prepare data for GPD prediction using n_feat (not n_win)
    n_feat = 2 * n_win  # This is the actual window size used in GPD
    net = st[0].stats.network
    sta = st[0].stats.station
    
    print(f"Data length: {len(st[0].data)}, Window size (n_feat): {n_feat}, Step size: {n_shift}")
    
    # Check if we have enough data
    if len(st[0].data) < n_feat:
        print(f"Error: Data length ({len(st[0].data)}) is shorter than window size ({n_feat})")
        return None, None, None, None
    
    # Create sliding windows (from gpd_predict.py sliding_window function)
    def sliding_window(data, size, stepsize=1, axis=-1, copy=True):
        if axis >= data.ndim:
            raise ValueError("Axis value out of range")
        if stepsize < 1:
            raise ValueError("Stepsize may not be zero or negative")
        if size > data.shape[axis]:
            raise ValueError("Sliding window size may not exceed size of selected axis")
        
        shape = list(data.shape)
        shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
        shape.append(size)
        
        strides = list(data.strides)
        strides[axis] *= stepsize
        strides.append(data.strides[axis])
        
        strided = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        
        if copy:
            return strided.copy()
        else:
            return strided
    
    # Create sliding windows for each channel separately (like original GPD code)
    print("Creating sliding windows for each channel...")
    sliding_N = sliding_window(st[0].data, n_feat, stepsize=n_shift)  # First channel (N)
    sliding_E = sliding_window(st[1].data, n_feat, stepsize=n_shift)  # Second channel (E)  
    sliding_Z = sliding_window(st[2].data, n_feat, stepsize=n_shift)  # Third channel (Z)
    
    print(f"Sliding window shapes: N={sliding_N.shape}, E={sliding_E.shape}, Z={sliding_Z.shape}")
    
    # Stack the sliding windows (shape: n_windows, n_feat, n_channels)
    tr_win = np.zeros((sliding_N.shape[0], n_feat, 3))
    tr_win[:, :, 0] = sliding_N
    tr_win[:, :, 1] = sliding_E  
    tr_win[:, :, 2] = sliding_Z
    
    print(f"Final window tensor shape: {tr_win.shape}")
    
    # Create time array
    tt = []
    for i in range(tr_win.shape[0]):
        tt.append(i * n_shift * only_dt)
    tt = np.array(tt)
    
    # Normalize windows
    tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
    tt = tt[:tr_win.shape[0]]
    
    # Predict
    print(f"Running prediction on {tr_win.shape[0]} windows...")
    ts = model.predict(tr_win, verbose=False, batch_size=batch_size)
    
    prob_P = ts[:,0]
    prob_S = ts[:,1] 
    prob_N = ts[:,2]
    
    # Create timestamps for probabilities
    prob_times = []
    start_time = st[0].stats.starttime
    for t in tt:
        prob_times.append(start_time + t)
    
    print(f"Extracted {len(prob_P)} probability predictions")
    print(f"P-wave probability range: {np.min(prob_P):.4f} - {np.max(prob_P):.4f}")
    
    return z_trace, prob_times, prob_P, prob_S


def plot_gpd_results(z_trace, prob_times, prob_P, prob_S, picks_file=None):
    """
    Plot the Z channel data and GPD probabilities.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Raw Z channel data
    times = z_trace.times('matplotlib')
    ax1.plot(times, z_trace.data, 'k-', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Amplitude\n(Z channel)', fontsize=12)
    ax1.set_title(f'GPD Analysis: {z_trace.stats.network}.{z_trace.stats.station}.{z_trace.stats.channel}\n'
                  f'Duration: {z_trace.stats.endtime - z_trace.stats.starttime:.0f}s, '
                  f'Sampling Rate: {z_trace.stats.sampling_rate}Hz', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GPD probabilities
    prob_times_mpl = [t.matplotlib_date for t in prob_times]
    ax2.plot(prob_times_mpl, prob_P, 'r-', linewidth=1.5, label='P-wave probability', alpha=0.8)
    ax2.plot(prob_times_mpl, prob_S, 'b-', linewidth=1.5, label='S-wave probability', alpha=0.8)
    
    # Add threshold line
    min_proba = 0.25  # Current threshold
    ax2.axhline(y=min_proba, color='gray', linestyle='--', alpha=0.7, label=f'Threshold ({min_proba})')
    
    # Add probability statistics to the plot
    p_stats = f'P-wave: min={np.min(prob_P):.3f}, max={np.max(prob_P):.3f}, mean={np.mean(prob_P):.3f}'
    s_stats = f'S-wave: min={np.min(prob_S):.3f}, max={np.max(prob_S):.3f}, mean={np.mean(prob_S):.3f}'
    ax2.text(0.02, 0.98, f'{p_stats}\n{s_stats}', transform=ax2.transAxes, 
             verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_xlabel('Time (UTC)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add picks if available
    pick_count = 0
    if picks_file and os.path.exists(picks_file):
        with open(picks_file, 'r') as f:
            picks = f.readlines()
        
        for pick_line in picks:
            parts = pick_line.strip().split()
            if len(parts) >= 4:
                pick_time_str = parts[4]
                phase = parts[3]
                try:
                    pick_time = UTCDateTime(pick_time_str)
                    pick_time_mpl = pick_time.matplotlib_date
                    
                    # Add vertical line to both plots
                    color = 'red' if phase == 'P' else 'blue'
                    ax1.axvline(x=pick_time_mpl, color=color, linestyle='-', alpha=0.8, linewidth=2, 
                               label=f'{phase} pick' if pick_count == 0 or (pick_count == 1 and phase == 'S') else '')
                    ax2.axvline(x=pick_time_mpl, color=color, linestyle='-', alpha=0.8, linewidth=2)
                    pick_count += 1
                    
                except Exception as e:
                    print(f"Could not parse pick time: {pick_time_str}, error: {e}")
    
    # Add legend for picks if any were plotted
    if pick_count > 0:
        ax1.legend(loc='upper left')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = '/tmp/gpd_analysis_plot.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    # Don't show plot in headless environment
    # plt.show()
    plt.close()  # Close the figure to free memory


def test_gpd_direct():
    """Test GPD directly on test data with plotting."""
    print("Testing GPD directly with plotting...")
    
    # Use correct test dayfile
    dayfile = "/tmp/test_dayfile.in"
    output = "/tmp/gpd_test_direct.out"
    
    try:
        print(f"Processing dayfile: {dayfile}")
        print(f"Output file: {output}")
        
        # First extract probabilities for plotting
        print("\n=== Extracting probabilities for plotting ===")
        z_trace, prob_times, prob_P, prob_S = extract_gpd_probabilities(dayfile)
        
        if z_trace is None:
            print("Failed to extract probabilities, skipping plot")
            return
        
        # Run GPD to generate picks
        print("\n=== Running GPD picker ===")
        process_dayfile(
            dayfile, 
            output, 
            base_dir='/home/jwalter/easyQuake/easyQuake/gpd_predict',
            verbose=True, 
            plot=False
        )
        
        # Check results
        if os.path.exists(output):
            with open(output, 'r') as f:
                picks = f.readlines()
            print(f"GPD generated {len(picks)} picks")
            for pick in picks[:5]:  # Show first 5
                print(f"  {pick.strip()}")
        else:
            print("No output file generated")
        
        # Create plot
        print("\n=== Creating plot ===")
        plot_gpd_results(z_trace, prob_times, prob_P, prob_S, output)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpd_direct()
