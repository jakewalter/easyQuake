"""
TensorFlow 2.x / Keras 3 compatible PhaseNet prediction script.
This module provides earthquake phase picking using modern TensorFlow.
"""

import argparse
import logging
import os
import pickle
import time
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from obspy.signal.trigger import trigger_onset
import matplotlib
matplotlib.use('Agg')

# Imports for phasenet submodules
from data_reader import DataReader_mseed_array, DataReader_pred, DataConfig
from model_tf2 import ModelConfigTF2, UNetTF2
from postprocess import (
    extract_amplitude,
    extract_picks,
    save_picks,
    save_picks_json,
    save_prob_h5,
)
from detect_peaks import detect_peaks
from tqdm import tqdm


def set_gpu_config():
    """Configure GPU settings for TensorFlow 2.x"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            logging.warning(f"GPU configuration error: {e}")
    else:
        logging.info("No GPUs found, using CPU")


def load_model_tf2(model_path, config=None):
    """
    Load a PhaseNet model compatible with TensorFlow 2.x
    
    Args:
        model_path: Path to model directory or file
        config: ModelConfigTF2 instance
        
    Returns:
        Loaded Keras model
    """
    if config is None:
        config = ModelConfigTF2()
    
    try:
        # Try loading as SavedModel format
        if os.path.isdir(model_path):
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Loaded SavedModel from {model_path}")
            return model
    except Exception as e:
        logging.warning(f"Could not load as SavedModel: {e}")
    
    try:
        # Try loading as .h5 format
        h5_path = model_path if model_path.endswith('.h5') else model_path + '.h5'
        if os.path.exists(h5_path):
            model = tf.keras.models.load_model(h5_path)
            logging.info(f"Loaded H5 model from {h5_path}")
            return model
    except Exception as e:
        logging.warning(f"Could not load as H5 model: {e}")
    
    try:
        # Try loading as .keras format
        keras_path = model_path if model_path.endswith('.keras') else model_path + '.keras'
        if os.path.exists(keras_path):
            model = tf.keras.models.load_model(keras_path)
            logging.info(f"Loaded Keras model from {keras_path}")
            return model
    except Exception as e:
        logging.warning(f"Could not load as Keras model: {e}")
    
    # If all loading attempts fail, create a new model
    logging.warning("Could not load pre-trained model, creating new model with random weights")
    model = UNetTF2(config)
    return model


def predict_tf2(model, data_reader, config, result_dir="results", 
                result_fname=None, plot_figure=False, save_prob=False,
                batch_size=1):
    """
    Perform phase prediction using TensorFlow 2.x model
    
    Args:
        model: Loaded Keras model
        data_reader: DataReader instance
        config: ModelConfigTF2 instance
        result_dir: Output directory
        result_fname: Output filename
        plot_figure: Whether to save figures
        save_prob: Whether to save probability outputs
        batch_size: Batch size for prediction
    """
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if result_fname is None:
        result_fname = "picks.csv"
    
    # Initialize results storage
    picks = []
    
    logging.info(f"Starting prediction on {len(data_reader)} samples")
    
    # Process data in batches
    for i in tqdm(range(len(data_reader)), desc="Predicting"):
        try:
            # Get data sample
            if hasattr(data_reader, 'amplitude') and data_reader.amplitude:
                sample, raw_amp, fname, t0, station_id = data_reader[i]
            else:
                sample, fname, t0, station_id = data_reader[i]
                raw_amp = None
            
            # Add batch dimension
            sample_batch = np.expand_dims(sample, axis=0)
            
            # Run prediction
            pred_batch = model(sample_batch, training=False).numpy()
            pred = pred_batch[0]  # Remove batch dimension
            
            # Extract picks from predictions
            sample_picks = extract_picks_tf2(
                pred, fname, t0, station_id, config,
                raw_amp=raw_amp if raw_amp is not None else None
            )
            
            picks.extend(sample_picks)
            
            # Save probability outputs if requested
            if save_prob:
                prob_file = os.path.join(result_dir, f"prob_{i:06d}.npz")
                np.savez(prob_file, prob=pred, fname=fname, t0=t0)
            
            # Plot figure if requested
            if plot_figure and i < 10:  # Limit plotting to first 10 samples
                plot_prediction_tf2(sample, pred, sample_picks, 
                                  os.path.join(result_dir, f"fig_{i:06d}.png"))
        
        except Exception as e:
            logging.error(f"Error processing sample {i}: {e}")
            continue
    
    # Save picks to CSV
    if picks:
        picks_df = pd.DataFrame(picks)
        output_path = os.path.join(result_dir, result_fname)
        picks_df.to_csv(output_path, index=False)
        logging.info(f"Saved {len(picks)} picks to {output_path}")
    else:
        logging.warning("No picks found")
    
    return picks


def extract_picks_tf2(pred, fname, t0, station_id, config, raw_amp=None):
    """
    Extract phase picks from prediction probabilities
    
    Args:
        pred: Prediction array (nt, 1, 3) - noise, P, S probabilities
        fname: Filename
        t0: Start time
        station_id: Station identifier
        config: ModelConfigTF2 instance
        raw_amp: Raw amplitude data (optional)
        
    Returns:
        List of pick dictionaries
    """
    picks = []
    
    # Extract P and S phase probabilities
    prob_p = pred[:, 0, 1]  # P phase probability
    prob_s = pred[:, 0, 2]  # S phase probability
    
    # Detect peaks for P phase
    p_peaks = detect_peaks(prob_p, mph=config.P_threshold, mpd=100)
    
    # Detect peaks for S phase  
    s_peaks = detect_peaks(prob_s, mph=config.S_threshold, mpd=100)
    
    # Convert to picks format
    for peak_idx in p_peaks:
        pick_time = pd.to_datetime(t0) + pd.Timedelta(seconds=peak_idx * config.dt)
        
        pick = {
            'fname': fname,
            'station_id': station_id,
            'phase_index': peak_idx,
            'phase_time': pick_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
            'phase_prob': prob_p[peak_idx],
            'phase_type': 'P'
        }
        
        if raw_amp is not None:
            pick['phase_amp'] = np.max(np.abs(raw_amp[peak_idx-10:peak_idx+10]))
        
        picks.append(pick)
    
    for peak_idx in s_peaks:
        pick_time = pd.to_datetime(t0) + pd.Timedelta(seconds=peak_idx * config.dt)
        
        pick = {
            'fname': fname,
            'station_id': station_id,
            'phase_index': peak_idx,
            'phase_time': pick_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
            'phase_prob': prob_s[peak_idx],
            'phase_type': 'S'
        }
        
        if raw_amp is not None:
            pick['phase_amp'] = np.max(np.abs(raw_amp[peak_idx-10:peak_idx+10]))
        
        picks.append(pick)
    
    return picks


def plot_prediction_tf2(sample, pred, picks, output_path):
    """
    Plot waveform data with predictions and picks
    
    Args:
        sample: Input waveform data
        pred: Prediction probabilities
        picks: List of picks
        output_path: Output file path
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        
        # Plot waveforms
        for i, comp in enumerate(['E', 'N', 'Z']):
            if i < sample.shape[-1]:
                axes[i].plot(sample[:, 0, i], 'k-', linewidth=0.5)
                axes[i].set_ylabel(f'{comp} component')
                axes[i].grid(True, alpha=0.3)
        
        # Plot probabilities
        axes[3].plot(pred[:, 0, 1], 'r-', label='P phase', linewidth=1)
        axes[3].plot(pred[:, 0, 2], 'b-', label='S phase', linewidth=1)
        axes[3].set_ylabel('Probability')
        axes[3].set_xlabel('Sample')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Mark picks
        for pick in picks:
            idx = pick['phase_index']
            color = 'red' if pick['phase_type'] == 'P' else 'blue'
            for ax in axes:
                ax.axvline(idx, color=color, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.warning(f"Could not create plot: {e}")


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='PhaseNet TF2 Prediction')
    parser.add_argument('--model_dir', type=str, required=False,
                       help='Path to trained model directory (TF2 style)')
    parser.add_argument('--model', type=str, required=False,
                       help='Path to trained model (legacy PhaseNet style)')
    parser.add_argument('--data_dir', type=str, required=False,
                       help='Path to input data directory (optional)')
    parser.add_argument('--data_list', type=str, required=True,
                       help='Path to data list file')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--result_fname', type=str, default='picks.csv',
                       help='Output filename')
    parser.add_argument('--format', type=str, default='mseed',
                       choices=['mseed', 'sac', 'numpy', 'hdf5'],
                       help='Input data format')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for prediction')
    parser.add_argument('--min_p_prob', type=float, default=0.1,
                       help='Minimum probability threshold for P picks')
    parser.add_argument('--min_s_prob', type=float, default=0.1,
                       help='Minimum probability threshold for S picks')
    parser.add_argument('--plot_figure', action='store_true',
                       help='Plot figures')
    parser.add_argument('--save_prob', action='store_true',
                       help='Save probability outputs')
    parser.add_argument('--amplitude', action='store_true',
                       help='Extract amplitude information')
    parser.add_argument('--highpass_filter', type=float, default=0.0,
                       help='Highpass filter frequency')

    args = parser.parse_args()

    # Backward compatibility: if --model_dir is not provided, use --model or default
    if not args.model_dir:
        if args.model:
            args.model_dir = args.model
        else:
            # Default to legacy model path if not provided
            args.model_dir = os.path.join(os.path.dirname(__file__), 'model', '190703-214543')

    # Backward compatibility: if --data_dir is not provided, try to infer from data_list
    if not args.data_dir:
        # Try to use the directory containing the first file in data_list
        try:
            with open(args.data_list, 'r') as f:
                first_line = f.readline().strip()
                if os.path.isfile(first_line):
                    args.data_dir = os.path.dirname(first_line)
                else:
                    args.data_dir = os.path.dirname(args.data_list)
        except Exception:
            args.data_dir = '.'
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure GPU
    set_gpu_config()
    
    # Create configuration
    config = ModelConfigTF2()
    config.P_threshold = args.min_p_prob
    config.S_threshold = args.min_s_prob
    
    # Load model
    logging.info(f"Loading model from {args.model_dir}")
    model = load_model_tf2(args.model_dir, config)
    
    # Create data reader
    logging.info(f"Setting up data reader for format: {args.format}")
    
    data_config = DataConfig()
    if args.highpass_filter > 0:
        data_config.highpass_filter = args.highpass_filter
    
    if args.format == 'mseed':
        data_reader = DataReader_pred(
            format=args.format,
            data_dir=args.data_dir,
            data_list=args.data_list,
            amplitude=args.amplitude,
            config=data_config,
            highpass_filter=args.highpass_filter
        )
    else:
        # Add support for other formats as needed
        raise NotImplementedError(f"Format {args.format} not yet implemented in TF2 version")
    
    # Run prediction
    logging.info("Starting phase prediction...")
    start_time = time.time()
    
    picks = predict_tf2(
        model=model,
        data_reader=data_reader,
        config=config,
        result_dir=args.result_dir,
        result_fname=args.result_fname,
        plot_figure=args.plot_figure,
        save_prob=args.save_prob,
        batch_size=args.batch_size
    )
    
    end_time = time.time()
    logging.info(f"Prediction completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Found {len(picks)} total picks")


if __name__ == "__main__":
    main()
