#!/usr/bin/env python
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
        
        print("\nNow you can use this .h5 file in your modern environment with:")
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
