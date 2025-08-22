import tensorflow as tf
import argparse
import logging
import os
import time
import numpy as np
import pandas as pd
from data_reader import DataReader_mseed_array, DataReader_pred
# Conditional import: use TF2 model if H5 files exist, otherwise TF1
import os
import glob
def get_model_imports():
    # Check if we're in a directory with H5 models - look in multiple possible locations
    possible_dirs = [
        os.path.join('model', '190703-214543'),  # from phasenet/ directory
        os.path.join('phasenet', 'model', '190703-214543'),  # from root
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', '190703-214543'))  # absolute path
    ]
    
    h5_files = []
    for model_dir in possible_dirs:
        if os.path.exists(model_dir):
            h5_files = glob.glob(os.path.join(model_dir, '*.h5'))
            if h5_files:
                break
    
    if h5_files:
        from model_tf2 import ModelConfigTF2 as ModelConfig, UNet
        return ModelConfig, UNet, True  # TF2 mode
    else:
        from model import ModelConfig, UNet
        return ModelConfig, UNet, False  # TF1 mode

ModelConfig, UNet, USE_TF2 = get_model_imports()
from postprocess import extract_amplitude, extract_picks, save_picks, save_picks_json, save_prob_h5
from tqdm import tqdm

# Configure TensorFlow for modern usage - but allow TF1 fallback
logging.info(f"Using TensorFlow {tf.__version__}")
EAGER_ENABLED_AT_START = tf.executing_eagerly()
# Try to enable eager execution if available
try:
    if hasattr(tf.config, 'run_functions_eagerly'):
        tf.config.run_functions_eagerly(True)  # Force eager execution for functions
except:
    pass
# Enable mixed precision if available
try:
    if hasattr(tf.keras, 'mixed_precision'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logging.info("Mixed precision enabled")
except:
    logging.info("Mixed precision not available")


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
    parser.add_argument("--min_p_prob", default=0.3, type=float, help="Probability threshold for P pick")
    parser.add_argument("--min_s_prob", default=0.3, type=float, help="Probability threshold for S pick")
    parser.add_argument("--mpd", default=50, type=float, help="Minimum peak distance")
    parser.add_argument("--amplitude", action="store_true", help="if return amplitude value")
    parser.add_argument("--format", default="numpy", help="input format")
    parser.add_argument("--stations", default="", help="seismic station info")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--save_prob", action="store_true", help="If save result for test")
    parser.add_argument("--pre_sec", default=1, type=float, help="Window length before pick")
    parser.add_argument("--post_sec", default=4, type=float, help="Window length after pick")
    parser.add_argument("--highpass_filter", default=0.0, type=float, help="Highpass filter frequency")
    parser.add_argument("--response_xml", default=None, type=str, help="response xml file")
    parser.add_argument("--sampling_rate", default=100, type=float, help="sampling rate")
    args = parser.parse_args()

    # Normalize result filename: strip any extension so we append a single ".out" later
    try:
        # os.path.splitext removes the last extension (e.g., '.out', '.csv')
        args.result_fname = os.path.splitext(args.result_fname)[0]
    except Exception:
        pass

    return args


def load_tf1_checkpoint_to_tf2_model(model, checkpoint_path):
    """
    Load TF1 checkpoint by creating a temporary TF1 model and copying weights
    """
    try:
        # Get the underlying Keras model
        keras_model = getattr(model, 'keras_model', model)
        
        # Import TF1 model temporarily
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Temporarily disable eager execution to load TF1 model
            if tf.executing_eagerly():
                tf.compat.v1.disable_eager_execution()
                eager_was_enabled = True
            else:
                eager_was_enabled = False
                
            try:
                from model import ModelConfig as ModelConfigTF1, UNet as UNetTF1
                
                # Create TF1 model with same config
                config_tf1 = ModelConfigTF1(X_shape=[3000, 1, 3])
                
                # Create placeholders and TF1 model
                X = tf.compat.v1.placeholder(tf.float32, shape=[None] + config_tf1.X_shape, name="X")
                model_tf1 = UNetTF1(config=config_tf1, mode="pred", input=X)
                
                # Create session and restore checkpoint
                sess = tf.compat.v1.Session()
                saver = tf.compat.v1.train.Saver()
                saver.restore(sess, checkpoint_path)
                
                logging.info("Successfully loaded TF1 checkpoint")
                
                # Get all trainable variables from TF1 model
                tf1_vars = tf.compat.v1.trainable_variables()
                tf1_values = sess.run(tf1_vars)
                
                # Clean up TF1 session
                sess.close()
                
                # Re-enable eager execution if it was enabled
                if eager_was_enabled:
                    tf.compat.v1.enable_eager_execution()
                
                # Now map TF1 weights to TF2 model
                # This is a simplified mapping - we'll map by shape and position
                tf2_vars = keras_model.trainable_variables
                
                # Map weights by matching shapes
                loaded_count = 0
                for tf1_val, tf1_var in zip(tf1_values, tf1_vars):
                    tf1_shape = tf1_val.shape
                    tf1_name = tf1_var.name
                    
                    # Find matching TF2 variable by shape
                    for tf2_var in tf2_vars:
                        if tf2_var.shape.as_list() == list(tf1_shape):
                            tf2_var.assign(tf1_val)
                            loaded_count += 1
                            break
                
                logging.info(f"Mapped {loaded_count} weights from TF1 to TF2 model")
                return loaded_count > 0
                
            except Exception as e:
                logging.error(f"Error loading TF1 model: {e}")
                if eager_was_enabled:
                    tf.compat.v1.enable_eager_execution()
                return False
                
    except Exception as e:
        logging.error(f"Error in checkpoint conversion: {e}")
        return False


def pred_fn(args, data_reader, figure_dir=None, prob_dir=None, log_dir=None):
    current_time = time.strftime("%y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(args.result_dir, "pred", current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if (args.plot_figure == True) and (figure_dir is None):
        figure_dir = os.path.join(log_dir, "figures")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
    if (args.save_prob == True) and (prob_dir is None):
        prob_dir = os.path.join(log_dir, "probs")
        if not os.path.exists(prob_dir):
            os.makedirs(prob_dir)
    
    logging.info("Pred log: %s" % log_dir)
    logging.info("Dataset size: {}".format(data_reader.num_data))

    # Create model configuration
    config = ModelConfig(X_shape=data_reader.X_shape)
    with open(os.path.join(log_dir, "config.log"), "w") as fp:
        fp.write("\n".join("%s: %s" % item for item in vars(config).items()))

    # Build model using new functional UNet (TF2/Keras 3 only)
    logging.info("Building TF2/Keras 3 functional UNet model...")
    model = UNet(config=config, mode="pred")

    # Debug: print model summary and weight stats
    keras_model = getattr(model, 'keras_model', model)
    try:
        keras_model.model.summary(print_fn=logging.info)
    except Exception as e:
        logging.info(f"Could not print model summary: {e}")
    total_weights = keras_model.model.count_params()
    weights = keras_model.model.get_weights()
    if weights:
        mean_weight = np.mean([np.mean(np.abs(w)) for w in weights if w.size > 0])
        max_weight = np.max([np.max(np.abs(w)) for w in weights if w.size > 0])
        min_weight = np.min([np.min(w) for w in weights if w.size > 0])
        logging.info(f"Model weights: total={total_weights}, mean(abs)={mean_weight:.4g}, max(abs)={max_weight:.4g}, min={min_weight:.4g}")
    else:
        logging.info("Model has no weights loaded.")

    # Check for checkpoint loading (TF2/Keras3 only: look for .h5 or .keras in model_dir)
    loaded_checkpoint = False
    if hasattr(args, 'model_dir') and args.model_dir:
        import glob
        ckpt_files = glob.glob(os.path.join(args.model_dir, '*.h5')) + glob.glob(os.path.join(args.model_dir, '*.keras'))
        if ckpt_files:
            ckpt_path = ckpt_files[0]
            try:
                keras_model.model.load_weights(ckpt_path)
                loaded_checkpoint = True
                logging.info(f"Loaded model weights from {ckpt_path}")
            except Exception as e:
                logging.warning(f"Failed to load weights from {ckpt_path}: {e}")
        else:
            logging.info(f"No .h5 or .keras checkpoint found in {args.model_dir}")
    else:
        logging.info("No model_dir specified; using randomly initialized weights.")
    logging.info(f"Checkpoint loaded: {loaded_checkpoint}")

    # Prepare for prediction
    picks = []
    amps = [] if args.amplitude else None

    logging.info("Starting prediction with TF2/Keras 3...")

    batch_size = max(1, int(args.batch_size))
    batch_inputs = []
    batch_meta = []  # store tuples (fname, t0, station_id, raw_amp)

    def flush_batch():
        nonlocal batch_inputs, batch_meta, picks
        if len(batch_inputs) == 0:
            return
        input_array = np.concatenate(batch_inputs, axis=0)
        # If input_array is too long, window it into (N, 3000, 1, 3)
        if input_array.shape[1] > 3000:
            n_windows = input_array.shape[1] // 3000
            input_array = input_array[:, :n_windows*3000, :, :]
            input_array = input_array.reshape(-1, 3000, 1, 3)
            # Repeat meta for each window
            batch_meta_expanded = []
            for meta in batch_meta:
                batch_meta_expanded.extend([meta]*n_windows)
            batch_meta = batch_meta_expanded
        try:
            if not isinstance(input_array, np.ndarray):
                input_array = np.array(input_array)
            preds = model.predict(input_array)
        except Exception as e:
            logging.error(f"Batch predict failed: {e}")
            preds = []
            for arr in batch_inputs:
                if arr.shape[0] > 3000:
                    n_windows = arr.shape[0] // 3000
                    arr = arr[:n_windows*3000, :, :].reshape(-1, 3000, 1, 3)
                    for window in arr:
                        pred = model.predict(window[np.newaxis, ...])
                        preds.append(pred[0])
                else:
                    pred = model.predict(arr[np.newaxis, ...])
                    preds.append(pred[0])
            preds = np.array(preds)

        preds = np.asarray(preds)

        for j in range(preds.shape[0]):
            pred = preds[j]
            fname, t0, station_id, raw_amp = batch_meta[j]
            sample_picks = extract_picks_from_prediction(pred, fname, t0, station_id, args, raw_amp)
            picks.extend(sample_picks)

        batch_inputs = []
        batch_meta = []

    for i in tqdm(range(data_reader.num_data), desc="Predicting"):
        try:
            # Get single sample
            if args.amplitude:
                sample, raw_amp, fname, t0, station_id = data_reader[i]
            else:
                sample, fname, t0, station_id = data_reader[i]
                raw_amp = None

            # Prepare input - ensure correct shape
            if len(sample.shape) == 3:
                # Add batch dimension: (nt, nsta, nch) -> (1, nt, nsta, nch)
                input_data = np.expand_dims(sample, axis=0)
            else:
                input_data = sample

            batch_inputs.append(input_data)
            batch_meta.append((fname, t0, station_id, raw_amp))

            if len(batch_inputs) >= batch_size:
                flush_batch()

        except Exception as e:
            logging.error(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # flush remaining
    flush_batch()
    
    # No TF1 session to clean up
    
    # Save results
    if len(picks) > 0:
        df = pd.DataFrame(picks)
        
        # Sanity check: if too many picks detected, likely using untrained model
        p_count = len([p for p in picks if p['phase_type'] == 'P'])
        s_count = len([p for p in picks if p['phase_type'] == 'S'])
        total_picks = len(picks)
        
        # Rough heuristic: more than 1000 picks per hour of data suggests untrained model
        if total_picks > 1000:
            logging.warning(f"WARNING: Detected {total_picks} picks ({p_count} P, {s_count} S)")
            logging.warning("This unusually high count suggests the model may be untrained or checkpoint loading failed.")
            logging.warning("Consider: 1) Check if checkpoint loaded properly, 2) Increase --min_p_prob and --min_s_prob thresholds")
        
        # Save in EQTransformer format: NETWORK STATION COMPONENT PHASE TIMESTAMP
        output_path = os.path.join(args.result_dir, args.result_fname + ".out")
        with open(output_path, 'w') as f:
            for pick in picks:
                # Parse station_id to extract network, station, and component
                # Expected format: "O2.WILZ..EH" or similar
                station_parts = pick['station_id'].split('.')
                network = station_parts[0] if len(station_parts) > 0 else "NET"
                station = station_parts[1] if len(station_parts) > 1 else "STA"
                
                # For component, we need to determine which channel this pick came from
                # Default to EHZ for now, but this could be improved by tracking the channel
                component = "EHZ"
                if pick['phase_type'] == 'S':
                    # S waves are often better detected on horizontal components
                    component = "EHE"
                
                # Format timestamp to match EQTransformer format (6 decimal places)
                timestamp = pick['phase_time'].strftime('%Y-%m-%dT%H:%M:%S.%f')  # Keep all 6 microsecond digits
                
                # Write in EQTransformer format: NETWORK STATION COMPONENT PHASE TIMESTAMP
                f.write(f"{network} {station} {component} {pick['phase_type']} {timestamp}\n")
        
        logging.info(f"Saved {len(picks)} picks to {output_path}")
    else:
        logging.warning("No picks found")
    
    return picks


def extract_picks_from_prediction(pred, fname, t0, station_id, args, raw_amp=None):
    """
    Extract phase picks from prediction probabilities using TF2/Keras 3
    """
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
        picks.append({
            'file_name': fname,
            'station_id': station_id_val,
            'phase_index': idx,
            'phase_time': pick_time,
            'phase_prob': float(prob_p[idx]),
            'phase_type': 'P',
        })

    for peak_idx in s_peaks:
        idx = int(np.asarray(peak_idx).item())
        pick_time = pd.to_datetime(t0_val) + pd.Timedelta(seconds=idx * dt)
        picks.append({
            'file_name': fname,
            'station_id': station_id_val,
            'phase_index': idx,
            'phase_time': pick_time,
            'phase_prob': float(prob_s[idx]),
            'phase_type': 'S',
        })

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
            highpass_filter=args.highpass_filter,
        )
    else:
        data_reader = DataReader_pred(
            format=args.format,
            data_dir=args.data_dir,
            data_list=args.data_list,
            hdf5_file=getattr(args, 'hdf5_file', None),
            hdf5_group=getattr(args, 'hdf5_group', 'data'),
            amplitude=args.amplitude,
            highpass_filter=args.highpass_filter,
        )

    # Run prediction
    picks = pred_fn(args, data_reader, log_dir=args.result_dir)
    
    # Print summary
    if picks:
        p_picks = len([p for p in picks if p['phase_type'] == 'P'])
        s_picks = len([p for p in picks if p['phase_type'] == 'S'])
        print(f"Done with {p_picks} P-picks and {s_picks} S-picks")
    else:
        print("Done with 0 P-picks and 0 S-picks")

    return picks


if __name__ == "__main__":
    args = read_args()
    main(args)
