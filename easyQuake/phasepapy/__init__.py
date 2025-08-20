# Minimal imports to avoid dependency issues
# Individual modules can be imported as needed
try:
    from . import fbpicker
    from . import tables1D
    from . import assoc1D
    from . import tt_stations_1D
except ImportError:
    # If obspy or other dependencies are missing, modules won't be available
    # but the package import won't fail
    pass
