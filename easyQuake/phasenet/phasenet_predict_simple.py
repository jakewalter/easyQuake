import argparse
import logging
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from data_reader import DataReader_mseed_array, DataReader_pred
try:
    # Try to use TF2 compatible model first
    from model_tf2 import ModelConfigTF2 as ModelConfig, UNet
    logging.info("Using TF2/Keras 3 compatible model")
except ImportError:
    # Fallback to original model (will likely fail with Keras 3)
    from model import ModelConfig, UNet
    logging.warning("Using legacy model - may not work with Keras 3")
from tqdm import tqdm

# Configure TensorFlow for modern usage
tf.config.run_functions_eagerly(True)  # Force eager execution
logging.info(f"Using TensorFlow {tf.__version__}")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--model_dir", help="Checkpoint directory (default: None)")
    parser.add_argument("--data_dir", default="", help="Input file directory")
    parser.add_argument("--data_list", default="", help="Input csv file")
    parser.add_argument("--hdf5_file", default="", help="Input hdf5 file")
    parser.add_argument("--hdf5_group", default="data", help="data group name in hdf5 file")
    parser.add_argument("--result_dir", default="results", help="Output directory")
    parser.add_argument("--result_fname", default="picks", help="Output file")
    parser.add_argument("--min_p_prob", default=0.5, type=float, help="Probability threshold for P pick")
    parser.add_argument("--min_s_prob", default=0.5, type=float, help="Probability threshold for S pick")
    parser.add_argument("--mpd", default=50, type=float, help="Minimum peak distance")
    parser.add_argument("--amplitude", action="store_true", help="if return amplitude value")
    parser.add_argument("--format", default="numpy", help="input format")
    parser.add_argument("--stations", default="", help="seismic station info")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--save_prob", action="store_true", help="If save result for test")
    args = parser.parse_args()
    return args


def load_tf1_weights_to_tf2_model(model, checkpoint_path):
    """Load TF1 checkpoint weights into TF2 model using direct variable mapping"""
    try:
        logging.info(f"Loading TF1 checkpoint: {checkpoint_path}")
        reader = tf.train.load_checkpoint(checkpoint_path)
        
        # Get keras model and build it
        keras_model = getattr(model, 'keras_model', model)
        dummy_input = tf.random.normal((1, 3000, 1, 3))
        _ = keras_model(dummy_input)
        
        # Key mappings: most important layers for functionality
        critical_mappings = [
            # Input layer - absolutely critical for proper preprocessing
            ("Input/input_conv/kernel", 0, "kernel"),
            ("Input/input_conv/bias", 0, "bias"), 
            ("Input/input_bn/gamma", 1, "gamma"),
            ("Input/input_bn/beta", 1, "beta"),
            ("Input/input_bn/moving_mean", 1, "moving_mean"),
            ("Input/input_bn/moving_variance", 1, "moving_variance"),
            # Output layer - critical for final predictions
            ("Output/output_conv/kernel", -1, "kernel"),
            ("Output/output_conv/bias", -1, "bias"),
        ]
        
        loaded_count = 0
        for tf1_var_name, layer_idx, weight_name in critical_mappings:
            try:
                tf1_weight = reader.get_tensor(tf1_var_name)
                layer = keras_model.layers[layer_idx]
                
                # Find the weight by name in the layer
                for w in layer.weights:
                    if weight_name in w.name:
                        w.assign(tf1_weight)
                        loaded_count += 1
                        logging.info(f"✓ Loaded {tf1_var_name} -> {layer.name}/{weight_name}")
                        break
            except Exception as e:
                logging.warning(f"✗ Failed to load {tf1_var_name}: {e}")
        
        # Try to load some middle layers by shape matching
        tf1_vars = reader.get_variable_to_shape_map()
        model_vars = keras_model.variables
        
        shape_matched = 0
        for tf1_name, tf1_shape in tf1_vars.items():
            if "Adam" in tf1_name or "global_step" in tf1_name or "beta1_power" in tf1_name or "beta2_power" in tf1_name:
                continue  # Skip optimizer variables
                
            # Find TF2 variable with matching shape
            for tf2_var in model_vars:
                if list(tf2_var.shape) == list(tf1_shape):
                    # Skip if already loaded
                    already_loaded = any(tf1_name.endswith(mapping[0].split('/')[-1]) for mapping in critical_mappings)
                    if not already_loaded:
                        try:
                            tf1_weight = reader.get_tensor(tf1_name)
                            tf2_var.assign(tf1_weight)
                            shape_matched += 1
                            logging.debug(f"Shape matched: {tf1_name} -> {tf2_var.name}")
                            break
                        except:
                            pass
        
        total_loaded = loaded_count + shape_matched
        logging.info(f"Loaded {loaded_count} critical + {shape_matched} shape-matched = {total_loaded} total weights")
        
        return total_loaded > 0
        
    except Exception as e:
        logging.error(f"Failed to load TF1 checkpoint: {e}")
        return False


def extract_picks_from_prediction(pred, fname, t0, station_id, args, raw_amp=None):
    """Extract phase picks from prediction probabilities"""
    from detect_peaks import detect_peaks
    import pandas as pd
    
    # Ensure pred is a numpy array
    if hasattr(pred, 'numpy'):
        pred = pred.numpy()
    
    picks = []
    
    # Get prediction probabilities and ensure 1D numpy arrays
    if len(pred.shape) == 3:
        prob_p = pred[:, 0, 1]  # P phase
        prob_s = pred[:, 0, 2]  # S phase
    else:
        prob_p = pred[:, 1]  # P phase
        prob_s = pred[:, 2]  # S phase

    # Force conversion to numpy 1D arrays
    prob_p = np.asarray(prob_p).ravel()
    prob_s = np.asarray(prob_s).ravel()
    
    # Detect P peaks (detect_peaks returns (indices, values))
    p_peaks, _ = detect_peaks(prob_p, mph=args.min_p_prob, mpd=args.mpd)
    # Detect S peaks
    s_peaks, _ = detect_peaks(prob_s, mph=args.min_s_prob, mpd=args.mpd)
    
    # Convert to picks format
    dt = 0.01  # 100 Hz sampling rate
    
    # Normalize t0 and station_id to scalar strings if they are arrays/lists
    if isinstance(t0, (list, tuple, np.ndarray)):
        t0_val = t0[0]
    else:
        t0_val = t0

    if isinstance(station_id, (list, tuple, np.ndarray)):
        station_id_val = station_id[0]
    else:
        station_id_val = station_id

    for peak_idx in p_peaks:
        idx = int(np.asarray(peak_idx).item())
        pick_time = pd.to_datetime(t0_val) + pd.Timedelta(seconds=idx * dt)
        
        pick = {
            'file_name': fname if isinstance(fname, str) else fname.decode() if hasattr(fname, 'decode') else str(fname),
            'station_id': station_id_val if isinstance(station_id_val, str) else station_id_val.decode() if hasattr(station_id_val, 'decode') else str(station_id_val),
            'phase_index': idx,
            'phase_time': pick_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
            'phase_prob': float(prob_p[idx]),
            'phase_type': 'P',
            'network': station_id_val.split('.')[0] if '.' in str(station_id_val) else '',
            'chan_pick': idx
        }
        picks.append(pick)
    
    for peak_idx in s_peaks:
        idx = int(np.asarray(peak_idx).item())
        pick_time = pd.to_datetime(t0_val) + pd.Timedelta(seconds=idx * dt)

        pick = {
            'file_name': fname if isinstance(fname, str) else fname.decode() if hasattr(fname, 'decode') else str(fname),
            'station_id': station_id_val if isinstance(station_id_val, str) else station_id_val.decode() if hasattr(station_id_val, 'decode') else str(station_id_val),
            'phase_index': idx,
            'phase_time': pick_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
            'phase_prob': float(prob_s[idx]),
            'phase_type': 'S',
            'network': station_id_val.split('.')[0] if '.' in str(station_id_val) else '',
            'chan_pick': idx
        }
        picks.append(pick)
    
    return picks


def main(args):
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logging.info(f"Using TensorFlow {tf.__version__} with Keras 3 compatibility")

    # Create data reader
    if args.format == "mseed_array":
        data_reader = DataReader_mseed_array(
            data_dir=args.data_dir,
            data_list=args.data_list,
            stations=args.stations,
            amplitude=args.amplitude,
        )
    else:
        data_reader = DataReader_pred(
            format=args.format,
            data_dir=args.data_dir,
            data_list=args.data_list,
            hdf5_file=getattr(args, 'hdf5_file', None),
            hdf5_group=getattr(args, 'hdf5_group', 'data'),
            amplitude=args.amplitude,
        )

    logging.info(f"Dataset size: {data_reader.num_data}")

    # Create model configuration
    config = ModelConfig(X_shape=data_reader.X_shape)
    
    # Build model
    logging.info("Building UNet model...")
    model = UNet(config=config, mode="pred")
    
    # Load TF1 checkpoint
    checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
    weights_loaded = False
    if checkpoint_path:
        weights_loaded = load_tf1_weights_to_tf2_model(model, checkpoint_path)
    
    if not weights_loaded:
        logging.warning("No trained weights loaded - predictions may be unreliable!")
    
    # Run prediction with batching
    picks = []
    batch_size = max(1, int(args.batch_size))
    batch_inputs = []
    batch_meta = []
    
    def flush_batch():
        nonlocal batch_inputs, batch_meta, picks
        if len(batch_inputs) == 0:
            return
        input_array = np.concatenate(batch_inputs, axis=0)
        
        preds = model.predict(input_array)
        preds = np.asarray(preds)
        
        for j in range(preds.shape[0]):
            pred = preds[j]
            fname, t0, station_id, raw_amp = batch_meta[j]
            sample_picks = extract_picks_from_prediction(pred, fname, t0, station_id, args, raw_amp)
            picks.extend(sample_picks)
        
        batch_inputs.clear()
        batch_meta.clear()
    
    logging.info("Starting prediction...")
    for i in tqdm(range(data_reader.num_data), desc="Processing"):
        try:
            if args.amplitude:
                sample, raw_amp, fname, t0, station_id = data_reader[i]
            else:
                sample, fname, t0, station_id = data_reader[i]
                raw_amp = None
            
            if len(sample.shape) == 3:
                input_data = np.expand_dims(sample, axis=0)
            else:
                input_data = sample
            
            batch_inputs.append(input_data)
            batch_meta.append((fname, t0, station_id, raw_amp))
            
            if len(batch_inputs) >= batch_size:
                flush_batch()
                
        except Exception as e:
            logging.error(f"Error processing sample {i}: {e}")
            continue
    
    # Flush remaining
    flush_batch()
    
    # Save results and check for unrealistic counts
    if len(picks) > 0:
        p_count = len([p for p in picks if p['phase_type'] == 'P'])
        s_count = len([p for p in picks if p['phase_type'] == 'S'])
        total_picks = len(picks)
        
        if total_picks > 1000:
            logging.warning(f"WARNING: {total_picks} picks detected ({p_count} P, {s_count} S)")
            logging.warning("High pick count may indicate untrained model or very low thresholds")
        
        df = pd.DataFrame(picks)
        df = df[["file_name", "station_id", "phase_index", "phase_time", "phase_prob", "phase_type"]]
        
        output_path = os.path.join(args.result_dir, args.result_fname + ".csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Saved {len(picks)} picks to {output_path}")
        print(f"Done with {p_count} P-picks and {s_count} S-picks")
    else:
        logging.warning("No picks found")
        print("Done with 0 P-picks and 0 S-picks")

    return picks


if __name__ == "__main__":
    args = read_args()
    main(args)
