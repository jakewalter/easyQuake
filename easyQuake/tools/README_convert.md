Conversion helper for EQTransformer EqT_model.h5

This repository includes legacy Keras HDF5 models that may not load under modern
TensorFlow/Keras versions. The `mseed_predictor.py` module includes a `convert_h5_to_keras`
helper that reconstructs the model and saves a modern `.keras` file.

Requirements:
- Python 3.8+ (use your project's environment)
- TensorFlow installed (the same major version you're running, e.g., TF 2.20)

How to run:

```bash
python3 tools/convert_eqt_model.py --h5 EQTransformer/EqT_model.h5
```

This will produce `EQTransformer/EqT_model.keras` next to the input HDF5 file.

If you run into import errors for missing packages (e.g., TensorFlow) run in a
virtualenv/conda environment that has them installed.
