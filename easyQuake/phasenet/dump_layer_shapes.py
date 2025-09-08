"""Dump per-layer output shapes for TF1 and TF2 UNet implementations.

Runs a single dummy batch through each model and prints op/layer output shapes
so we can locate where the time dimension changes.
"""
import os
import numpy as np


def dump_tf2():
    print("\n=== TF2 / Keras model layer output shapes ===\n")
    from phasenet.model_tf2 import ModelConfigTF2, UNet
    config = ModelConfigTF2(X_shape=[3000,1,3])
    wrapper = UNet(config=config, mode='pred')
    # underlying keras model
    keras_obj = getattr(wrapper, 'keras_model', wrapper)
    keras_model = getattr(keras_obj, 'model', keras_obj)

    try:
        keras_model.summary(print_fn=lambda s: None)
    except Exception:
        pass

    # Print each layer's name and output shape
    for layer in keras_model.layers:
        try:
            shape = layer.output_shape
        except Exception:
            shape = getattr(layer, 'output_shape', None)
        print(f"{layer.name:40s} -> {shape}")

    # Run a dummy forward to ensure runtime shapes
    x = np.zeros((1,3000,1,3), dtype=np.float32)
    out = keras_model.predict(x)
    print(f"Keras model runtime output shape: {np.asarray(out).shape}\n")



def dump_tf1():
    print("\n=== TF1 model op output shapes (selected ops) ===\n")
    import tensorflow.compat.v1 as tf1
    tf1.disable_eager_execution()
    from phasenet.phasenet_original.phasenet.model import ModelConfig as ModelConfigTF1, UNet as UNetTF1

    config = ModelConfigTF1(X_shape=[3000,1,3])
    X = tf1.placeholder(tf1.float32, shape=[None, 3000, 1, 3], name='X')
    # pass input_batch as tuple so model stores self.X correctly
    model = UNetTF1(config=config, input_batch=(X,), mode='pred')

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        feed = {X: np.zeros((1,3000,1,3), dtype=np.float32)}

        g = sess.graph
        # Select ops likely related to conv/pool/up operations
        interesting = []
        for op in g.get_operations():
            name = op.name.lower()
            if any(k in name for k in ('input_conv','down_conv','down_conv3','down_conv1','up_conv0','up_conv1','encoder_pool','conv2d','conv2d_transpose','up_conv','pool')):
                interesting.append(op)

        # De-duplicate and sort
        seen = set()
        filtered = []
        for op in interesting:
            if op.name in seen:
                continue
            seen.add(op.name)
            filtered.append(op)

        # Print op name, type and runtime shape of first output
        for op in filtered:
            try:
                t = op.outputs[0]
                shape_val = sess.run(tf1.shape(t), feed_dict=feed)
                print(f"{op.name:60s} ({op.type:20s}) -> runtime shape {tuple(shape_val)}")
            except Exception as e:
                print(f"{op.name:60s} ({op.type:20s}) -> error getting shape: {e}")


if __name__ == '__main__':
    dump_tf2()
    dump_tf1()
