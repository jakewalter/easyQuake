import os
import argparse
import tensorflow as tf

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

# Ensure shim is present when module is imported
_ensure_legacy_keras_backend_functions()

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
        # Load the model from the old path.
        # Keras can automatically handle models saved in the .h5 format.
        model = tf.keras.models.load_model(old_model_path)
        print("Model loaded successfully.")

        # Save the model to the new path. By using the .keras extension,
        # you are saving it in the new, recommended format.
        print(f"Saving updated model to: '{new_model_path}'")
        model.save(new_model_path)
        print("Model successfully updated and saved in the new format.")
        return True

    except Exception as e:
        print(f"An error occurred during the model update process: {e}")
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
        model.load_weights(weights_path)
        print("Model weights loaded successfully.")

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

    # --- USAGE EXAMPLES ---
    #
    # To convert a single .h5 file:
    # python your_script_name.py --mode h5 --h5_file /path/to/my_model.h5 --output_file /path/to/new_model.keras
    #
    # To convert from a .json and .h5 weights file:
    # python convert_old_model_to_new.py --mode json --json_file model_pol.json --weights_file model_pol_best.hdf5 --output_file model_pol_best.keras

    # Create helper script for Keras 2.x environment
    create_keras2_helper_script()