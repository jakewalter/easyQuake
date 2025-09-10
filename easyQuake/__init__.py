"""easyQuake package initializer.

WARNING: This package is designed to prefer GPU usage for ML components (e.g., Seisbench, EQTransformer)
but falls back to CPU if GPU is unavailable or incompatible. CI tests are CPU-only due to hosted runner
limitations. For GPU support, ensure compatible CUDA drivers and install GPU-enabled ML frameworks.

Keep the top-level import lightweight but provide lazy access to the
primary API functions defined in `easyQuake.easyQuake`.

This avoids importing heavy dependencies (obspy, tensorflow, torch)
at import time while preserving the public API surface used by
legacy scripts and tests (for example `easyQuake.detection_continuous`).
"""

__version__ = "1.4.0"

# Public names we expose lazily from easyQuake.easyQuake
_LAZY_EXPORTS = [
    # common, historically-exposed names kept as a small hint for tools.
    'detection_continuous',
    'detection_association_event',
    'detection_continuous_event',
    'download_mseed',
    'combine_associated',
]

def __getattr__(name: str):
    """Lazily import public attributes from `easyQuake.easyQuake` on demand.

    Any non-private attribute lookup (not starting with '_') will attempt a lazy
    import of the implementation module and return the attribute if present.

    This keeps `import easyQuake` cheap but allows callers to do things like
    `from easyQuake import detection_continuous` or `import easyQuake; easyQuake.foo`.
    """
    # Don't try to expose dunder/private attributes lazily
    if name.startswith("_"):
        raise AttributeError(f"module 'easyQuake' has no attribute '{name}'")

    import sys, importlib
    impl_name = __name__ + '.easyQuake'
    if impl_name in sys.modules:
        _mod = sys.modules[impl_name]
    else:
        try:
            _mod = importlib.import_module(impl_name)
        except Exception:
            # If importing the implementation fails, propagate error so callers see why.
            raise

    # Legacy alias: allow callers using the old name to continue to work.
    if name == 'detection_continuous_event' and not hasattr(_mod, name):
        if hasattr(_mod, 'detection_continuous'):
            val = getattr(_mod, 'detection_continuous')
            globals()[name] = val
            return val
        # fall through to normal error handling

    if hasattr(_mod, name):
        val = getattr(_mod, name)
        globals()[name] = val
        return val

    raise AttributeError(f"module 'easyQuake' has no attribute '{name}'")

def __dir__():
    """Return available attributes: combine globals with what's provided by the impl.

    We try to avoid importing the implementation at module import time, but
    `__dir__` is allowed to import it lazily to provide a complete listing.
    """
    names = set(globals().keys())
    names.update(_LAZY_EXPORTS)
    try:
        import sys, importlib
        impl_name = __name__ + '.easyQuake'
        if impl_name in sys.modules:
            _mod = sys.modules[impl_name]
        else:
            _mod = importlib.import_module(impl_name)
        for n in dir(_mod):
            if not n.startswith("_"):
                names.add(n)
    except Exception:
        # If import fails, fall back to the lightweight listing
        pass
    return sorted(names)

# Expose a minimal __all__ by default; tools that import after installation
# will be able to access the full list through __dir__ and attribute access.
__all__ = list(_LAZY_EXPORTS) + ['__version__']
