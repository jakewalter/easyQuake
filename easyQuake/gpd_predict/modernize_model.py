#!/usr/bin/env python
"""
Converts a legacy Keras .h5 model file to the modern .keras format.

This script should be run in a modern environment (TensorFlow 2.x/3.x).
It assumes you have already combined your legacy model's .json and weights 
into a single .h5 file using a Keras 2.x environment.

Usage:
    python modernize_model.py <input_legacy_h5_file> <output_new_keras_file>

Example:
    python modernize_model.py model_pol_legacy.h5 model_pol_new.keras
"""
import tensorflow as tf
import sys
import os

def convert_h5_to_keras(h5_path, keras_path):
    """
    Loads a model from a legacy .h5 file and saves it in the new .keras format.
    """
    if not os.path.exists(h5_path):
        print(f"❌ Error: Input file not found at '{h5_path}'")
        return

    print(f"▶️  Loading legacy model from: '{h5_path}'...")
    
    try:
        # Load the model without its optimizer. This is the key to avoiding
        # most legacy compatibility issues.
        model = tf.keras.models.load_model(h5_path, compile=False)
        print("✅ Model loaded successfully.")

        # The model can now be re-compiled with a modern optimizer if needed, e.g.:
        # model.compile(optimizer='adam', loss='your_loss_function')

        print(f"💾 Saving model in modern .keras format to: '{keras_path}'...")
        model.save(keras_path)
        print(f"✅ Model successfully saved to '{keras_path}'")

        # --- Verification Step ---
        print("\n🔎 Verifying the new model...")
        verified_model = tf.keras.models.load_model(keras_path)
        verified_model.summary()
        print("\n✅ Verification successful! The new model is ready to use.")

    except Exception as e:
        print(f"\n❌ An error occurred during conversion: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the .h5 file was correctly created in a Keras 2.x environment.")
        print("2. If your model uses custom layers, you may need to define them and use the 'custom_objects' argument:")
        print("   from your_custom_layers import MyCustomLayer")
        print(f"   model = tf.keras.models.load_model('{h5_path}', custom_objects={{'MyCustomLayer': MyCustomLayer}}, compile=False)")

def main():
    if len(sys.argv) != 3:
        print("Usage: python modernize_model.py <input_legacy_h5_file> <output_new_keras_file>")
        sys.exit(1)
    
    input_h5_path = sys.argv[1]
    output_keras_path = sys.argv[2]
    
    convert_h5_to_keras(input_h5_path, output_keras_path)

if __name__ == "__main__":
    main()
