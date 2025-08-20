#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

# Load a converted model and examine its final layer
model = tf.keras.models.load_model('model_pol_new.keras')
print("Model loaded successfully")
print("\nModel summary:")
model.summary()

print("\nExamining final layers:")
for i, layer in enumerate(model.layers):
    if 'dense' in layer.name.lower() or 'activation' in layer.name.lower():
        print(f"Layer {i}: {layer.name} - {layer.__class__.__name__}")
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                print(f"  Weights shape: {[w.shape for w in weights]}")
                if 'dense' in layer.name.lower() and len(weights) >= 2:
                    # Check final dense layer weights and biases
                    W, b = weights[0], weights[1]
                    print(f"  Weight range: [{np.min(W):.4f}, {np.max(W):.4f}]")
                    print(f"  Bias range: [{np.min(b):.4f}, {np.max(b):.4f}]")
                    print(f"  Weight magnitude: {np.mean(np.abs(W)):.4f}")
                    
# Create some test input to see what the model produces
test_input = np.random.randn(1, 400, 3)
print(f"\nTest input shape: {test_input.shape}")

# Get logits before final activation
for i, layer in enumerate(model.layers):
    if 'dense' in layer.name.lower() and layer.units == 3:
        # This should be the final dense layer
        logit_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        logits = logit_model.predict(test_input, verbose=0)
        print(f"\nFinal dense layer ({layer.name}) logits: {logits[0]}")
        
        # Apply softmax manually
        probs = tf.nn.softmax(logits).numpy()
        print(f"Softmax probabilities: {probs[0]}")
        
        # Check what happens with temperature scaling
        for temp in [0.1, 0.05, 0.01]:
            scaled_logits = logits / temp
            scaled_probs = tf.nn.softmax(scaled_logits).numpy()
            print(f"Temperature {temp}: {scaled_probs[0]} (max: {np.max(scaled_probs[0]):.4f})")
        break

# Also check final model output
final_output = model.predict(test_input, verbose=0)
print(f"\nFinal model output: {final_output[0]} (max: {np.max(final_output[0]):.4f})")
