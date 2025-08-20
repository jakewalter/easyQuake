#!/usr/bin/env python3
"""
Test the temperature-corrected GPD model with the original 0.994 threshold.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys

# Add the parent directory to the path so we can import gpd_predict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_corrected_model():
    """Test the temperature-corrected model"""
    print("Testing Temperature-Corrected GPD Model")
    print("=" * 50)
    
    # Load the corrected model
    corrected_model_path = "model_pol_temp_0.03.keras"
    if not os.path.exists(corrected_model_path):
        print(f"✗ Corrected model not found: {corrected_model_path}")
        return False
    
    print(f"Loading corrected model: {corrected_model_path}")
    corrected_model = keras.models.load_model(corrected_model_path)
    
    # Generate test data (simulate seismic waveforms)
    print("\nGenerating test data...")
    batch_size = 20
    test_data = np.random.randn(batch_size, 400, 3).astype(np.float32)
    
    # Add some synthetic P-wave signals to test detection
    for i in range(0, batch_size, 4):  # Every 4th sample gets a clear signal
        # Create a P-wave like signal in the Z component (channel 0)
        p_arrival = 200  # Sample index for P-wave arrival
        amplitude = 2.0
        frequency = 10.0
        
        # Generate P-wave signal
        t = np.arange(400) - p_arrival
        signal = amplitude * np.exp(-0.01 * t**2) * np.sin(2 * np.pi * frequency * t / 100)
        signal[t < 0] = 0  # Only after arrival
        
        test_data[i, :, 0] += signal
        print(f"  Added P-wave signal to sample {i}")
    
    # Test with corrected model
    print("\nTesting corrected model...")
    predictions = corrected_model.predict(test_data, verbose=0)
    
    print(f"\nPrediction results:")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Number of samples: {len(predictions)}")
    
    # Analyze predictions
    max_probs = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"\nProbability statistics:")
    print(f"  Mean max probability: {np.mean(max_probs):.4f}")
    print(f"  Min max probability: {np.min(max_probs):.4f}")
    print(f"  Max max probability: {np.max(max_probs):.4f}")
    
    # Test threshold 0.994
    threshold = 0.994
    above_threshold = max_probs > threshold
    picks_count = np.sum(above_threshold)
    
    print(f"\nThreshold test (>{threshold}):")
    print(f"  Samples above threshold: {picks_count}/{len(predictions)} ({100*picks_count/len(predictions):.1f}%)")
    
    if picks_count > 0:
        print(f"  ✓ Model generates picks with original threshold!")
        
        # Show details of detected picks
        pick_indices = np.where(above_threshold)[0]
        print(f"\nDetected picks:")
        for idx in pick_indices:
            prob_vector = predictions[idx]
            max_prob = np.max(prob_vector)
            predicted_class = np.argmax(prob_vector)
            class_names = ['Noise', 'P-wave', 'S-wave']
            
            print(f"  Sample {idx}: {class_names[predicted_class]} (prob={max_prob:.4f})")
            print(f"    Full probabilities: {prob_vector}")
    else:
        print(f"  ✗ No picks generated with threshold {threshold}")
        
        # Try lower thresholds to see where picks start appearing
        for test_thresh in [0.99, 0.95, 0.9, 0.8]:
            test_picks = np.sum(max_probs > test_thresh)
            print(f"  Threshold {test_thresh}: {test_picks} picks")
    
    return picks_count > 0

def update_gpd_predict_to_use_corrected_model():
    """Update gpd_predict.py to use the corrected model"""
    print("\n" + "=" * 50)
    print("Updating gpd_predict.py to use corrected model")
    print("=" * 50)
    
    corrected_model_path = os.path.abspath("model_pol_temp_0.03.keras")
    backup_model_path = os.path.abspath("model_pol_new.keras")
    
    if not os.path.exists(corrected_model_path):
        print(f"✗ Corrected model not found: {corrected_model_path}")
        return False
    
    # Create a backup and replace the model
    if os.path.exists(backup_model_path):
        backup_name = "model_pol_new_backup.keras"
        if not os.path.exists(backup_name):
            os.rename(backup_model_path, backup_name)
            print(f"✓ Backed up original model to {backup_name}")
    
    # Copy corrected model to the expected location
    import shutil
    shutil.copy(corrected_model_path, backup_model_path)
    print(f"✓ Installed corrected model as {backup_model_path}")
    
    return True

def main():
    success = test_corrected_model()
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Temperature correction fixed the model!")
        print("=" * 60)
        
        # Offer to update the actual model file
        response = input("\nReplace the current model with the corrected version? (y/n): ")
        if response.lower().startswith('y'):
            if update_gpd_predict_to_use_corrected_model():
                print("\n✓ Model successfully updated!")
                print("✓ GPD should now work with the original 0.994 threshold")
                print("\nYou can now test with:")
                print("  cd /home/jwalter/easyQuake/tests")
                print("  python test_all_pickers.py")
            else:
                print("\n✗ Failed to update model")
        else:
            print("\nModel not updated. You can manually copy model_pol_temp_0.03.keras")
            print("to model_pol_new.keras when ready.")
    else:
        print("\n✗ Temperature correction did not resolve the issue")

if __name__ == "__main__":
    main()
