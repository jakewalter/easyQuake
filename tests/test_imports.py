def test_imports():
    """Test that all main modules can be imported."""
    import easyQuake
    import easyQuake.gpd_predict.gpd_predict
    import easyQuake.EQTransformer.mseed_predictor
    import easyQuake.phasenet.phasenet_predict
    import easyQuake.seisbench.run_seisbench
    # Add more imports as needed for coverage

import importlib
import pytest

def test_imports():
    """Test that main modules can be imported; skip when optional deps are missing."""
    import easyQuake  # ensure package itself imports

    modules = [
        'easyQuake.gpd_predict.gpd_predict',
        'easyQuake.EQTransformer.mseed_predictor',
        'easyQuake.phasenet.phasenet_predict',
        'easyQuake.seisbench.run_seisbench',
    ]

    for mod in modules:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            # Skip the specific import when a required optional dependency
            # (e.g., obspy, torch, tensorflow) isn't installed in this env.
            pytest.skip(f"Skipping import {mod} due to missing dependency: {e.name}")
