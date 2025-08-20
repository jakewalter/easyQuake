"""Small compatibility shim to provide TF1-style helpers when running PhaseNet
under TF2 / Keras3. This file is intentionally small and defensive; it only
monkeypatches symbols if missing.
"""

def ensure_compat():
    try:
        import tensorflow as tf
    except Exception:
        return

    # Try to switch TF into v1-style graph mode (disable eager) first
    try:
        if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
            try:
                tf.compat.v1.disable_eager_execution()
            except Exception:
                # best-effort; if it fails, continue and other shims may help
                pass
    except Exception:
        pass

    # Ensure placeholder exists
    try:
        if not hasattr(tf, 'placeholder'):
            tf.placeholder = tf.compat.v1.placeholder
    except Exception:
        pass

    # Provide a conv2d shim if tf.compat.v1.layers.conv2d is not available
    try:
        _ = tf.compat.v1.layers.conv2d
    except Exception:
        try:
            from tensorflow.keras.layers import Conv2D
            from tensorflow.keras import activations

            def _conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, name=None, data_format=None, dilation_rate=(1,1), **kwargs):
                # Map simple args to Keras Conv2D. This is a pragmatic shim and may not
                # preserve all subtle behavior of the original tf.compat.v1.layers.conv2d.
                layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, name=name)
                x = layer(inputs)
                if activation:
                    try:
                        return activations.get(activation)(x)
                    except Exception:
                        return x
                return x

            try:
                tf.compat.v1.layers.conv2d = _conv2d
            except Exception:
                # If layers namespace is missing entirely, create it
                try:
                    tf.compat.v1.layers = type('obj', (), {})()
                    tf.compat.v1.layers.conv2d = _conv2d
                except Exception:
                    pass
        except Exception:
            pass

    # best-effort: set learning phase to 0 if available
    try:
        try:
            tf.compat.v1.keras.backend.set_learning_phase(0)
        except Exception:
            from tensorflow.keras import backend as _kb
            try:
                _kb.set_learning_phase(0)
            except Exception:
                pass
    except Exception:
        pass

    # Provide batch_normalization shim if missing
    try:
        _ = tf.compat.v1.layers.batch_normalization
    except Exception:
        try:
            from tensorflow.keras.layers import BatchNormalization

            def _batch_normalization(inputs, training=False, name=None, **kwargs):
                # Create a Keras BatchNormalization layer and apply it. This is a
                # pragmatic shim and may not perfectly match TF1 behavior.
                layer = BatchNormalization(name=name)
                try:
                    return layer(inputs, training=training)
                except Exception:
                    # Fallback: call without training flag
                    return layer(inputs)

            try:
                if not hasattr(tf.compat.v1, 'layers'):
                    tf.compat.v1.layers = type('obj', (), {})()
                tf.compat.v1.layers.batch_normalization = _batch_normalization
            except Exception:
                pass
        except Exception:
            pass

    # Provide dropout shim if missing
    try:
        _ = tf.compat.v1.layers.dropout
    except Exception:
        try:
            from tensorflow.keras.layers import Dropout

            def _dropout(inputs, rate, training=False, name=None, **kwargs):
                # Keras Dropout's 'rate' matches TF's 'rate' (fraction to drop)
                layer = Dropout(rate=rate, name=name)
                try:
                    return layer(inputs, training=training)
                except Exception:
                    # Fallback: call without training flag
                    return layer(inputs)

            try:
                if not hasattr(tf.compat.v1, 'layers'):
                    tf.compat.v1.layers = type('obj', (), {})()
                tf.compat.v1.layers.dropout = _dropout
            except Exception:
                pass
        except Exception:
            pass
