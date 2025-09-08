Converting Old Keras Models to Work with Modern Keras/TensorFlow

PROBLEM:
Old Keras models (especially with Lambda layers) often fail to load in modern 
Keras/TensorFlow environments. You might see errors like:
- "ValueError: File not found: filepath=model.keras"
- "EOF read where object expected" 
- Lambda layer deserialization errors

QUICK SOLUTION:
If you have a model.json + weights.h5 pair, use the simple_rebuild.py script:

1. Run this command:
   python gpd_predict/simple_rebuild.py --json model.json --weights weights.h5 --output new_model.keras

2. The script will:
   - Read your model architecture from the JSON
   - Skip problematic Lambda layers  
   - Build a simplified version
   - Load your trained weights
   - Save as a modern .keras file

WHAT IT DOES:
- Removes complex Lambda layer slicing (common source of errors)
- Keeps all your trained CNN/Dense layer weights
- Creates a clean model that works with modern Keras
- Same input/output shapes as original

FILES YOU NEED:
- model.json (model architecture)
- weights.h5 (trained weights)

FILES IT CREATES:
- new_model.keras (working modern format)

TESTING:
After conversion, test with:
python -c "import tensorflow as tf; model = tf.keras.models.load_model('new_model.keras'); print('Success!')"

This approach works for most CNN-based models that had Lambda layer compatibility issues.
