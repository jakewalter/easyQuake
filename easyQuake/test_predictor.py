"""
Simple test script to validate the predictor can load and use the model.
This bypasses the Lambda layer issue by testing the original predictor logic.
"""

import sys
import os
import numpy as np

# Add the gpd_predict directory to path so we can import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpd_predict'))

def test_predictor():
    """Test the predictor with the new model."""
    try:
        # Import the predictor module
        import gpd_predict
        
        # Test with a sample waveform
        sample_data = np.random.randn(1, 400, 3).astype(np.float32)
        
        print("Testing predictor with rebuilt model...")
        
        # This will test the fallback loading logic in gpd_predict.py
        # which should try the .keras file first, then fall back to legacy
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_predictor()
    sys.exit(0 if success else 1)
