import logging
import numpy as _np
import tensorflow as tf


class CropAndConcat(tf.keras.layers.Layer):
    """Center-crop two tensors to the same temporal length and concatenate on channels."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        s_shape, n_shape = input_shape
        batch = s_shape[0]
        time = None  # dynamic time dimension
        width = s_shape[2] if len(s_shape) > 2 else 1
        ch_s = s_shape[3] if len(s_shape) > 3 else 1
        ch_n = n_shape[3] if len(n_shape) > 3 else 1
        ch = ch_s + ch_n  # sum channels
        return (batch, time, width, ch)

    def call(self, inputs):
        s, n = inputs
        s_shape = tf.shape(s)
        n_shape = tf.shape(n)
        s_nt = s_shape[1]
        n_nt = n_shape[1]
        target = tf.minimum(s_nt, n_nt)

        s_start = (s_nt - target) // 2
        n_start = (n_nt - target) // 2

        s_slice = tf.slice(s, [0, s_start, 0, 0], [-1, target, -1, -1])
        n_slice = tf.slice(n, [0, n_start, 0, 0], [-1, target, -1, -1])
        
        # Match original crop_and_concat behavior - crop n to match s size
        result = tf.concat([s_slice, n_slice], axis=-1)
        
        # Set static shape like original (important for shape inference)
        s_static_shape = s.get_shape().as_list()
        n_static_shape = n.get_shape().as_list()
        if s_static_shape[-1] is not None and n_static_shape[-1] is not None:
            result.set_shape([None, None, None, s_static_shape[-1] + n_static_shape[-1]])
            
        return result


class ModelConfigTF2:
    def __init__(self, **kwargs):
        self.batch_size = 20
        self.depths = 5
        self.filters_root = 8
        self.kernel_size = (7, 1)
        self.pool_size = (4, 1)
        self.dilation_rate = (1, 1)
        self.drop_rate = 0.0
        self.weight_decay = 0.0
        self.X_shape = [3000, 1, 3]
        self.n_channel = self.X_shape[-1]
        self.Y_shape = [3000, 1, 3]
        self.n_class = self.Y_shape[-1]
        # Use the same initializer as original PhaseNet
        self.initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        # Add regularizer support like original
        if self.weight_decay:
            self.regularizer = tf.keras.regularizers.l2(l=0.5 * self.weight_decay)
        else:
            self.regularizer = None
        for k, v in kwargs.items():
            setattr(self, k, v)
            # Update regularizer if weight_decay was changed
            if k == 'weight_decay':
                if v:
                    self.regularizer = tf.keras.regularizers.l2(l=0.5 * v)
                else:
                    self.regularizer = None


class UNetTF2(tf.keras.Model):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        if config is None:
            config = ModelConfigTF2()
        self._config = config
        self.model = self.build_phasenet_unet_functional(config)

    def build_phasenet_unet_functional(self, config):
        inputs = tf.keras.Input(shape=(config.X_shape[0], config.X_shape[1], config.X_shape[2]))

        net = tf.keras.layers.Conv2D(
            filters=config.filters_root,
            kernel_size=config.kernel_size,
            padding='same',
            dilation_rate=config.dilation_rate,
            kernel_initializer=config.initializer,
            kernel_regularizer=config.regularizer,
            name="input_conv",
        )(inputs)
        net = tf.keras.layers.BatchNormalization(name="input_bn")(net)
        net = tf.keras.layers.Activation('relu', name="input_relu")(net)
        net = tf.keras.layers.Dropout(config.drop_rate, name="input_dropout")(net)

        convs = []
        for depth in range(config.depths):
            filters = config.filters_root * (2 ** depth)
            net = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=config.kernel_size,
                padding='same',
                dilation_rate=config.dilation_rate,
                kernel_initializer=config.initializer,
                kernel_regularizer=config.regularizer,
                use_bias=False,
                name=f"down_conv1_{depth+1}",
            )(net)
            net = tf.keras.layers.BatchNormalization(name=f"down_bn1_{depth+1}")(net)
            net = tf.keras.layers.Activation('relu', name=f"down_relu1_{depth+1}")(net)
            net = tf.keras.layers.Dropout(config.drop_rate, name=f"down_dropout1_{depth+1}")(net)
            convs.append(net)
            if depth < config.depths - 1:
                net = tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=config.kernel_size,
                    padding='same',
                    dilation_rate=config.dilation_rate,
                    strides=config.pool_size,
                    kernel_initializer=config.initializer,
                    kernel_regularizer=config.regularizer,
                    use_bias=False,
                    name=f"down_conv3_{depth+1}",
                )(net)
                net = tf.keras.layers.BatchNormalization(name=f"down_bn3_{depth+1}")(net)
                net = tf.keras.layers.Activation('relu', name=f"down_relu3_{depth+1}")(net)
                net = tf.keras.layers.Dropout(config.drop_rate, name=f"down_dropout3_{depth+1}")(net)

        for depth in reversed(range(config.depths - 1)):
            filters = config.filters_root * (2 ** depth)
            net = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=config.kernel_size,
                strides=config.pool_size,
                padding='same',
                use_bias=False,
                kernel_initializer=config.initializer,
                kernel_regularizer=config.regularizer,
                name=f"up_conv0_{depth+1}",
            )(net)
            net = tf.keras.layers.BatchNormalization(name=f"up_bn0_{depth+1}")(net)
            net = tf.keras.layers.Activation('relu', name=f"up_relu0_{depth+1}")(net)
            net = tf.keras.layers.Dropout(config.drop_rate, name=f"up_dropout0_{depth+1}")(net)
            net = CropAndConcat()([convs[depth], net])
            net = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=config.kernel_size,
                padding='same',
                dilation_rate=config.dilation_rate,
                kernel_initializer=config.initializer,
                kernel_regularizer=config.regularizer,
                use_bias=False,
                name=f"up_conv1_{depth+1}",
            )(net)
            net = tf.keras.layers.BatchNormalization(name=f"up_bn1_{depth+1}")(net)
            net = tf.keras.layers.Activation('relu', name=f"up_relu1_{depth+1}")(net)
            net = tf.keras.layers.Dropout(config.drop_rate, name=f"up_dropout1_{depth+1}")(net)

        output = tf.keras.layers.Conv2D(
            filters=config.n_class, 
            kernel_size=(1, 1), 
            padding='same', 
            kernel_initializer=config.initializer,
            kernel_regularizer=config.regularizer,
            name="output_conv")(net)
        output = tf.keras.layers.Activation('softmax', name="output_softmax")(output)
        model = tf.keras.Model(inputs=inputs, outputs=output, name="PhaseNetUNetFunctional")
        return model

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)


class UNet:
    def __init__(self, config=None, mode='pred'):
        if config is None:
            config = ModelConfigTF2()
        elif hasattr(config, 'X_shape'):
            new_config = ModelConfigTF2()
            for attr in ['X_shape', 'Y_shape', 'n_channel', 'n_class', 'depths', 'filters_root', 'kernel_size', 'pool_size', 'dilation_rate']:
                if hasattr(config, attr):
                    setattr(new_config, attr, getattr(config, attr))
            config = new_config
        self.config = config
        self.keras_model = UNetTF2(config)
        dummy_input = tf.zeros([1, config.X_shape[0], config.X_shape[1], config.X_shape[2]])
        _ = self.keras_model(dummy_input, training=False)
        logging.info(f"Legacy UNet wrapper created with {self.keras_model.count_params()} parameters")

    def predict(self, inputs):
        arr = inputs
        if isinstance(arr, tf.Tensor):
            try:
                arr = arr.numpy()
            except Exception:
                arr = _np.array(arr)
        arr = _np.asarray(arr)
        if arr.ndim == 3:
            arr = _np.expand_dims(arr, axis=0)
        if arr.ndim != 4:
            raise ValueError(f"Input to predict must be 4D (batch, nt, 1, ch) or 3D (nt,1,ch); got {arr.shape}")
        keras_model = getattr(self.keras_model, 'model', self.keras_model)
        try:
            preds = keras_model.predict(arr)
        except Exception as e:
            logging.error(f"keras_model.predict failed: {e}. Falling back to direct call.")
            out = keras_model(arr, training=False)
            preds = out.numpy() if hasattr(out, 'numpy') else _np.array(out)
        return preds
