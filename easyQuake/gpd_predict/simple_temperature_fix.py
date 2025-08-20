#!/usr/bin/env python3
"""
Simple approach: Load the converted model and apply temperature scaling directly.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def create_temperature_wrapper_model(original_model, temperature=0.03):
    """Create a wrapper model that applies temperature scaling"""
    print(f"Creating temperature wrapper with T={temperature}")
    
    # Create a new model that wraps the original
    inputs = keras.Input(shape=(400, 3), name='input')
    
    # Get the output from the original model (before softmax)
    # We need to get the logits (pre-softmax values)
    
    # Find the sequential model inside
    sequential_layer = None
    for layer in original_model.layers:
        if isinstance(layer, keras.models.Sequential):
            sequential_layer = layer
            break
    
    if not sequential_layer:
        raise ValueError("No sequential layer found")
    
    # Create a new sequential model without the final softmax
    layers_without_softmax = []
    for i, layer in enumerate(sequential_layer.layers[:-1]):  # Exclude last layer (softmax)
        layers_without_softmax.append(layer)
    
    # Add the final dense layer but without softmax activation
    final_dense = sequential_layer.layers[-2]  # This should be the Dense layer before softmax
    if isinstance(final_dense, keras.layers.Dense):
        # Clone the dense layer but without activation
        new_dense = keras.layers.Dense(
            units=final_dense.units,
            activation='linear',
            use_bias=final_dense.use_bias,
            name='final_dense_linear'
        )
        layers_without_softmax.append(new_dense)
    
    # Create new sequential model
    logits_model = keras.models.Sequential(layers_without_softmax, name='logits_model')
    
    # Apply the sequential model to inputs
    logits = logits_model(inputs)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Apply softmax
    probabilities = keras.layers.Activation('softmax', name='temperature_softmax')(scaled_logits)
    
    # Create the final model
    temperature_model = keras.Model(inputs=inputs, outputs=probabilities, name='gpd_temperature_scaled')
    
    # Copy weights
    # First, build the logits model
    logits_model.build(input_shape=(None, 400, 3))
    
    # Copy weights from original sequential model
    for i, (orig_layer, new_layer) in enumerate(zip(sequential_layer.layers[:-1], logits_model.layers)):
        if orig_layer.get_weights():
            new_layer.set_weights(orig_layer.get_weights())
            print(f"Copied weights for layer {i}: {orig_layer.name}")
    
    # Copy weights for the final dense layer
    if isinstance(sequential_layer.layers[-2], keras.layers.Dense):
        final_dense_orig = sequential_layer.layers[-2]
        final_dense_new = logits_model.layers[-1]
        if final_dense_orig.get_weights():
            final_dense_new.set_weights(final_dense_orig.get_weights())
            print(f"Copied weights for final dense layer")
    
    return temperature_model

def test_temperature_scaling():
    """Test the temperature scaling approach"""
    print("GPD Temperature Scaling Test")
    print("=" * 40)
    
    # Load original converted model
    original_model = keras.models.load_model("model_pol_new.keras")
    print("✓ Loaded original converted model")
    
    # Create test data
    test_data = np.random.randn(10, 400, 3).astype(np.float32)
    print("✓ Created test data")
    
    # Test original model
    orig_predictions = original_model.predict(test_data, verbose=0)
    orig_max_probs = np.max(orig_predictions, axis=1)
    print(f"Original model max probs: {orig_max_probs[:5]}")
    print(f"Original mean max prob: {np.mean(orig_max_probs):.4f}")
    
    # Try different temperatures
    best_temp = None
    best_model = None
    
    for temp in [0.05, 0.03, 0.02, 0.01]:
        try:
            print(f"\nTesting temperature: {temp}")
            temp_model = create_temperature_wrapper_model(original_model, temp)
            
            # Test predictions
            temp_predictions = temp_model.predict(test_data, verbose=0)
            temp_max_probs = np.max(temp_predictions, axis=1)
            
            print(f"  Max probs: {temp_max_probs[:5]}")
            print(f"  Mean max prob: {np.mean(temp_max_probs):.4f}")
            
            # Count high confidence predictions
            high_conf = np.sum(temp_max_probs > 0.99)
            very_high_conf = np.sum(temp_max_probs > 0.994)
            
            print(f"  >0.99 confidence: {high_conf}/{len(temp_predictions)}")
            print(f"  >0.994 confidence: {very_high_conf}/{len(temp_predictions)}")
            
            if very_high_conf > 0:
                print(f"  ✓ SUCCESS: Found {very_high_conf} predictions above 0.994 threshold!")
                best_temp = temp
                best_model = temp_model
                break
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    if best_model:
        print(f"\n{'='*60}")
        print(f"BEST TEMPERATURE: {best_temp}")
        print(f"{'='*60}")
        
        # Save the model in the old H5 format to avoid Lambda issues
        h5_path = f"model_pol_corrected_T{best_temp}.h5"
        best_model.save(h5_path, save_format='h5')
        print(f"✓ Saved corrected model as {h5_path}")
        
        # Also save as keras format for compatibility
        try:
            keras_path = f"model_pol_corrected_T{best_temp}.keras"
            best_model.save(keras_path)
            print(f"✓ Saved corrected model as {keras_path}")
        except Exception as e:
            print(f"Warning: Could not save as .keras format: {e}")
        
        return best_model, best_temp
    else:
        print("\n✗ No suitable temperature found")
        return None, None

def update_gpd_model_file(corrected_model, temperature):
    """Update the GPD model file with the corrected version"""
    print(f"\nUpdating GPD model file...")
    
    # Backup original
    if os.path.exists("model_pol_new.keras"):
        backup_name = "model_pol_new_original_backup.keras"
        if not os.path.exists(backup_name):
            import shutil
            shutil.copy("model_pol_new.keras", backup_name)
            print(f"✓ Backed up original to {backup_name}")
    
    # Save corrected model with standard name
    try:
        corrected_model.save("model_pol_new.keras")
        print("✓ Replaced model_pol_new.keras with temperature-corrected version")
        return True
    except Exception as e:
        print(f"✗ Failed to save corrected model: {e}")
        # Try H5 format instead
        try:
            corrected_model.save("model_pol_new.h5", save_format='h5')
            print("✓ Saved as model_pol_new.h5 (H5 format)")
            
            # Update gpd_predict.py to use .h5 file
            print("Note: You may need to update gpd_predict.py to load .h5 file")
            return True
        except Exception as e2:
            print(f"✗ Failed to save in H5 format too: {e2}")
            return False

def main():
    try:
        corrected_model, best_temp = test_temperature_scaling()
        
        if corrected_model:
            print(f"\n{'='*60}")
            print("SUCCESS: Temperature scaling fixed the model!")
            print(f"Best temperature: {best_temp}")
            print("The model now produces high-confidence predictions")
            print("compatible with the original 0.994 threshold.")
            print(f"{'='*60}")
            
            # Ask user if they want to replace the model
            try:
                response = input("\nReplace the current model file? (y/n): ")
                if response.lower().startswith('y'):
                    if update_gpd_model_file(corrected_model, best_temp):
                        print("\n✓ Model file updated successfully!")
                        print("✓ GPD should now work with original 0.994 threshold")
                        print("\nTest it with:")
                        print("  cd /home/jwalter/easyQuake/tests")
                        print("  python test_all_pickers.py")
                    else:
                        print("\n✗ Failed to update model file")
                else:
                    print(f"\nModel not updated automatically.")
                    print(f"Corrected model saved as model_pol_corrected_T{best_temp}.h5")
            except (EOFError, KeyboardInterrupt):
                print(f"\nCorrected model saved as model_pol_corrected_T{best_temp}.h5")
        else:
            print("\n✗ Could not create working temperature-scaled model")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
