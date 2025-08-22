"""easyQuake package initializer.

Keep the top-level import lightweight but provide lazy access to the
primary API functions defined in `easyQuake.easyQuake`.

This avoids importing heavy dependencies (obspy, tensorflow, torch)
at import time while preserving the public API surface used by
legacy scripts and tests (for example `easyQuake.detection_continuous`).
"""

__version__ = "1.4.0"

# Public names we expose lazily from easyQuake.easyQuake
_LAZY_EXPORTS = [
    'detection_continuous',
    'detection_association_event',
    'detection_continuous_event',
    'download_mseed',
    'combine_associated',
    # add other commonly used functions here as needed
]

def __getattr__(name: str):
    """Lazily import attributes from `easyQuake.easyQuake` on demand.

    This keeps `import easyQuake` cheap but allows `easyQuake.detection_continuous`
    to work as callers expect.
    """
    if name in _LAZY_EXPORTS:
        try:
            from . import easyQuake as _mod
        except Exception:
            # Re-raise with a clearer message to aid debugging in CI/test runs
            raise
        try:
            val = getattr(_mod, name)
        except AttributeError:
            raise AttributeError(f"module 'easyQuake' has no attribute '{name}'")
        globals()[name] = val
        return val
    raise AttributeError(f"module 'easyQuake' has no attribute '{name}'")

def __dir__():
    return sorted(list(globals().keys()) + _LAZY_EXPORTS)

__all__ = _LAZY_EXPORTS
