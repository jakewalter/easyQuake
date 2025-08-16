def test_imports():
    """Test that all main modules can be imported."""
    import easyQuake
    import easyQuake.gpd_predict.gpd_predict
    import easyQuake.EQTransformer.mseed_predictor
    import easyQuake.phasenet.phasenet_predict
    import easyQuake.seisbench.run_seisbench
    # Add more imports as needed for coverage
