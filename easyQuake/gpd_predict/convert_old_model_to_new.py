import os
import argparse
import sys
import traceback

# Try to import TensorFlow; fail gracefully with instructions if not present.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_IMPORT_ERR = None
except Exception as e:
    tf = None
    TF_AVAILABLE = False
    TF_IMPORT_ERR = e

# Compatibility shim for legacy Keras backend functions used in old model configs
def _ensure_legacy_keras_backend_functions():
    """Ensure legacy backend functions (e.g. slice) exist on common backend modules.

    Some legacy Keras model JSON refer to backend functions via different module
    paths (tf.keras.backend and tensorflow.python.keras.backend). Patch all
    likely targets so model deserialization can find them.
    """
    import types
    # Primary tf.keras backend
    try:
        kb = tf.keras.backend
        if not hasattr(kb, 'slice'):
            def _kb_slice(x, start, size):
                return tf.slice(x, start, size)
            setattr(kb, 'slice', _kb_slice)
    except Exception:
        pass

    # Also patch tensorflow.python.keras.backend if present
    try:
        import tensorflow.python.keras.backend as tpkb
        if not hasattr(tpkb, 'slice'):
            def _tpkb_slice(x, start, size):
                return tf.slice(x, start, size)
            setattr(tpkb, 'slice', _tpkb_slice)
    except Exception:
        pass

    # If standalone keras is installed, patch keras.backend as well
    try:
        import keras.backend as kb2
        if not hasattr(kb2, 'slice'):
            def _kb2_slice(x, start, size):
                return tf.slice(x, start, size)
            setattr(kb2, 'slice', _kb2_slice)
    except Exception:
        pass

# Ensure shim is present when TensorFlow is available
# (we call this later in main after checking TF availability)

def update_keras_model_from_single_file(old_model_path, new_model_path):
    """
    Loads a Keras model from a single .h5 file and saves it in the
    newer, recommended .keras format.

    Args:
        old_model_path (str): The file path of the old Keras model (.h5).
        new_model_path (str): The file path to save the updated Keras model (.keras).
    """
    if not os.path.exists(old_model_path):
        print(f"Error: The file '{old_model_path}' was not found.")
        return

    print(f"\nLoading model from: '{old_model_path}'")
    try:
        print('file size:', os.path.getsize(old_model_path))
    except Exception:
        pass
    # Try tf.keras load first
    try:
        print('Attempting to load with tf.keras...')
        model = tf.keras.models.load_model(old_model_path)
        print("Model loaded successfully with tf.keras.")
        print(f"Saving updated model to: '{new_model_path}'")
        model.save(new_model_path)
        print("Model successfully updated and saved in the new format.")
        try:
            print('post-save exists:', os.path.exists(new_model_path), 'size:', os.path.getsize(new_model_path) if os.path.exists(new_model_path) else 'N/A')
        except Exception:
            pass
        return True
    except Exception as e_tf:
        print("tf.keras failed to load H5:")
        traceback.print_exc()
        # Try standalone keras (legacy) if available
        try:
            import keras
            print("Attempting to load with standalone keras...")
            # dump h5 structure for debugging
            try:
                import h5py
                with h5py.File(old_model_path,'r') as hf:
                    print('HDF5 top keys:', list(hf.keys()))
            except Exception:
                pass
            kmodel = keras.models.load_model(old_model_path)
            print("Standalone Keras loaded model; re-saving through tf.keras...")
            import tempfile
            tmp_h5 = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
            tmp_h5_path = tmp_h5.name
            tmp_h5.close()
            kmodel.save(tmp_h5_path)
            print('tmp h5 saved to', tmp_h5_path, 'size', os.path.getsize(tmp_h5_path))
            try:
                model = tf.keras.models.load_model(tmp_h5_path)
                model.save(new_model_path)
                print('Saved new model to', new_model_path)
                try:
                    print('post-save exists:', os.path.exists(new_model_path), 'size:', os.path.getsize(new_model_path) if os.path.exists(new_model_path) else 'N/A')
                except Exception:
                    pass
                return True
            finally:
                try:
                    os.remove(tmp_h5_path)
                except Exception:
                    pass
        except Exception as e_ks:
            print('Standalone Keras also failed:')
            traceback.print_exc()
            print("Conversion failed. Consider converting in a Keras 2.x environment using the helper script.")
            return False

def update_keras_model_from_json(json_path, weights_path, new_model_path):
    """
    Loads a model from a JSON architecture file and H5 weights file,
    then saves it in the new .keras format.

    Args:
        json_path (str): The path to the model architecture (.json).
        weights_path (str): The path to the model weights (.h5).
        new_model_path (str): The path to save the updated model (.keras).
    """
    if not os.path.exists(json_path):
        print(f"Error: The JSON file was not found at '{json_path}'.")
        return
    if not os.path.exists(weights_path):
        print(f"Error: The weights file was not found at '{weights_path}'.")
        return

    print(f"\nLoading model architecture from: '{json_path}'")
    try:
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        print('Loaded JSON length:', len(loaded_model_json))
        
        # Try different approaches to load the model
        custom_objects = {'tf': tf}  # For Lambda layers that use tf
        
        # First try: standard model_from_json
        try:
            model = tf.keras.models.model_from_json(loaded_model_json, custom_objects=custom_objects)
            print("Model architecture loaded successfully with standard loader.")
        except Exception as e1:
            print(f"Standard loader failed: {e1}")
            
            # Second try: try with additional custom objects
            try:
                from tensorflow.keras.utils import get_custom_objects
                custom_objects.update(get_custom_objects())
                model = tf.keras.models.model_from_json(loaded_model_json, custom_objects=custom_objects)
                print("Model architecture loaded successfully with extended custom objects.")
            except Exception as e2:
                print(f"Extended loader failed: {e2}")
                
                # Third try: create a simple legacy-compatible environment
                try:
                    import tensorflow.keras.utils as utils
                    with utils.custom_object_scope(custom_objects):
                        model = tf.keras.models.model_from_json(loaded_model_json)
                    print("Model architecture loaded successfully with custom object scope.")
                except Exception as e3:
                    print(f"All loading attempts failed. Final error: {e3}")
                    print("\nThis model likely requires Keras 2.x to load properly.")
                    print("Consider using a Keras 2.x environment to convert to .h5 format first.")
                    return False

        print(f"Loading model weights from: '{weights_path}'")
        try:
            print('Attempting to load weights directly...')
            model.load_weights(weights_path)
            print("Model weights loaded successfully.")
        except Exception as e_w:
            print('Direct model.load_weights failed:')
            traceback.print_exc()
            try:
                print('Attempting load_weights(by_name=True)')
                model.load_weights(weights_path, by_name=True)
                print("Model weights loaded with by_name=True")
            except Exception as e_by:
                print('load_weights by_name also failed:')
                traceback.print_exc()
                # dump weight h5 contents for debugging
                try:
                    import h5py
                    with h5py.File(weights_path,'r') as wf:
                        print('weights HDF5 top keys:', list(wf.keys()))
                except Exception:
                    pass
                # Try standalone keras path: build in keras, save temp h5, load with tf.keras
                try:
                    import keras
                    from keras.models import model_from_json as k_model_from_json
                    print('Attempting JSON+weights load using standalone keras...')
                    with open(json_path,'r') as jf:
                        jm = jf.read()
                    kmodel = k_model_from_json(jm)
                    kmodel.load_weights(weights_path)
                    import tempfile
                    tmp_h5 = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
                    tmp_h5_path = tmp_h5.name
                    tmp_h5.close()
                    kmodel.save(tmp_h5_path)
                    print('Standalone keras temporary h5 saved to', tmp_h5_path)
                    try:
                        model = tf.keras.models.load_model(tmp_h5_path)
                        model.save(new_model_path)
                        print('Converted JSON+weights via standalone keras and saved to new format.')
                        try:
                            print('post-save exists:', os.path.exists(new_model_path), 'size:', os.path.getsize(new_model_path) if os.path.exists(new_model_path) else 'N/A')
                        except Exception:
                            pass
                        return True
                    finally:
                        try:
                            os.remove(tmp_h5_path)
                        except Exception:
                            pass
                except Exception as e_final:
                    print('Standalone keras JSON+weights conversion failed:')
                    traceback.print_exc()
                    return False

        # Use a more generic compilation that's likely to work
        try:
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        except:
            # If compilation fails, try without it (weights are already loaded)
            print("Model compilation skipped (weights already loaded).")

        print(f"Saving updated model to: '{new_model_path}'")
        model.save(new_model_path)
        print("Model successfully updated and saved in the new format.")
        return True

    except Exception as e:
        print(f"An error occurred during the model update process: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. This model may require Keras 2.x to load properly")
        print("2. Try creating a Keras 2.x environment and saving as .h5 format")
        print("3. Then use the h5 mode of this script to convert to .keras")
        return False


def verify_model(model_path):
    """Loads and prints the summary of a model to verify it's working."""
    if os.path.exists(model_path):
        print(f"\nVerifying the updated model: {model_path}")
        try:
            loaded_model = tf.keras.models.load_model(model_path)
            loaded_model.summary()
            print(f"Verification successful: '{model_path}' can be loaded.")
        except Exception as e:
            print(f"Verification failed. Could not load the model: {e}")

def create_keras2_helper_script():
    """Creates a helper script for converting models in a Keras 2.x environment."""
    script_content = '''#!/usr/bin/env python
"""
Helper script to convert legacy Keras models in a Keras 2.x environment.
Run this in an environment with Keras 2.x and TensorFlow 1.x or early 2.x.

Usage:
    python keras2_converter.py model_pol.json model_pol_best.hdf5 model_pol_legacy.h5
"""
import sys
import os

def convert_json_to_h5(json_path, weights_path, output_h5_path):
    try:
        # Import Keras 2.x (not tf.keras)
        from keras.models import model_from_json
        
        print(f"Loading model architecture from: {json_path}")
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        
        model = model_from_json(loaded_model_json)
        print("Model architecture loaded successfully.")
        
        print(f"Loading model weights from: {weights_path}")
        model.load_weights(weights_path)
        print("Model weights loaded successfully.")
        
        print(f"Saving complete model to: {output_h5_path}")
        model.save(output_h5_path)
        print("Model successfully saved in .h5 format!")
        
        print("\\nNow you can use this .h5 file in your modern environment with:")
        print(f"python convert_old_model_to_new.py --mode h5 --h5_file {output_h5_path} --output_file model_pol_new.keras")
        
    except ImportError:
        print("Error: This script requires Keras 2.x (not tf.keras).")
        print("Install with: pip install keras==2.3.1 tensorflow==1.15")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python keras2_converter.py <json_file> <weights_file> <output_h5_file>")
        sys.exit(1)
    
    json_path, weights_path, output_h5_path = sys.argv[1:4]
    convert_json_to_h5(json_path, weights_path, output_h5_path)
'''
    
    script_path = os.path.join(os.path.dirname(__file__), 'keras2_converter.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\nCreated helper script: {script_path}")
    print("This script can be used in a Keras 2.x environment to convert your model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update legacy Keras models (.h5 or .json/.h5) to the new .keras format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        choices=['h5', 'json'], 
        required=True, 
        help="The conversion mode.\n"
             "'h5': for a single .h5 model file.\n"
             "'json': for a .json architecture file and a .h5 weights file."
    )
    parser.add_argument('--h5_file', type=str, help="Path to the input legacy Keras model file (.h5).")
    parser.add_argument('--json_file', type=str, help="Path to the input model architecture file (.json).")
    parser.add_argument('--weights_file', type=str, help="Path to the input model weights file (.h5).")
    parser.add_argument('--output_file', type=str, required=True, help="Path for the output updated model file (.keras).")

    args = parser.parse_args()

    if not TF_AVAILABLE:
        print("\nTensorFlow is not importable in this environment.")
        print("Import error:", TF_IMPORT_ERR)
        print("\nTo run this conversion script you need TensorFlow available (typically in a conda env or virtualenv).")
        print("Suggested quick setup (conda):")
        print("  conda create -n keras2 python=3.8")
        print("  conda activate keras2")
        print("  pip install keras==2.3.1 tensorflow==1.15")
        print("Then run the included helper `keras2_converter.py` in that env to produce a .h5, and re-run this script in your modern env to create a .keras file.")
        print("\nAlternatively, if you already have a legacy .h5 file, run this script in an environment with TensorFlow available.")
        # still create helper script for user's convenience
        create_keras2_helper_script()
        sys.exit(1)

    # If TF is available, ensure legacy shim functions are present
    _ensure_legacy_keras_backend_functions()

    success = False
    if args.mode == 'h5':
        if not args.h5_file:
            parser.error("--h5_file is required for 'h5' mode.")
        success = update_keras_model_from_single_file(args.h5_file, args.output_file)
    
    elif args.mode == 'json':
        if not args.json_file or not args.weights_file:
            parser.error("--json_file and --weights_file are required for 'json' mode.")
        success = update_keras_model_from_json(args.json_file, args.weights_file, args.output_file)
        
        # If conversion failed, create the helper script
        if not success:
            print("\nCreating Keras 2.x helper script...")
            create_keras2_helper_script()

    if success:
        verify_model(args.output_file)

    # Create helper script for Keras 2.x environment
    create_keras2_helper_script()