"""Lightweight package initializer for the EQTransformer subpackage.

Avoid importing heavy optional dependencies (like TensorFlow) at
package import time. Submodules should be imported explicitly by callers.
"""

__all__ = [
    'mseed_predictor',
    'EqT_utils',
]

