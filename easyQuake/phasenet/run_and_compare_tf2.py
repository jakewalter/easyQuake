import numpy as np
import sys
import os

# --- User must set these paths ---
INPUT_NPY = "phasenet/input_waveform.npy"
# TF1 checkpoint is only needed when running the TF1 block in a TF1 environment
TF1_CKPT = "path/to/tf1/model.ckpt"
# Default to the converted TF2 .h5 if present in the repo; edit if you want a different file
TF2_H5 = "/home/jwalter/easyQuake/easyQuake/phasenet/model/190703-214543/model_95_converted.h5"

# --- TF2 Model Import ---
def run_tf2_model(input_array):
    from phasenet.model_tf2 import ModelConfigTF2, UNet
    config = ModelConfigTF2(X_shape=[3000, 1, 3])
    # Create the compatibility wrapper UNet which delegates to the Keras model
    unet = UNet(config=config, mode="pred")

    # Resolve the underlying Keras model instance robustly. The wrapper stores
    # the TF2 implementation at `unet.keras_model` which itself has `.model`.
    keras_obj = getattr(unet, 'keras_model', None) or getattr(unet, 'model', None) or unet
    keras_inner = getattr(keras_obj, 'model', None) or keras_obj

    # Load weights onto the resolved Keras model
    try:
        keras_inner.load_weights(TF2_H5)
    except Exception as e:
        # Provide a clearer error message for debugging weight loading issues
        raise RuntimeError(f"Failed to load TF2 weights from '{TF2_H5}': {e}")

    # Use the wrapper's predict() which handles numpy conversion and batching
    output = unet.predict(input_array)
    return output

if __name__ == "__main__":
    arr = np.load(INPUT_NPY)
    print(f"Loaded input: {arr.shape}")
    # --- TF2 ---
    print("Running TF2 model...")
    out_tf2 = run_tf2_model(arr)
    np.save("output_tf2.npy", out_tf2)
    print("TF2 output saved to output_tf2.npy")
    # --- TF1 ---
    # To run TF1, activate the easyquake environment and run the following code as a separate script:
    # import numpy as np
    # import sys, os
    # arr = np.load('input_waveform.npy')
    # import tensorflow.compat.v1 as tf1
    # tf1.disable_eager_execution()
    # sys.path.append(os.path.dirname(__file__))
    # from phasenet.phasenet_original.phasenet.model import ModelConfig as ModelConfigTF1, UNet as UNetTF1
    # config = ModelConfigTF1(X_shape=[3000, 1, 3])
    # X = tf1.placeholder(tf.float32, shape=[None, 3000, 1, 3], name="X")
    # model = UNetTF1(config=config, mode="pred", input=X)
    # saver = tf1.train.Saver()
    # with tf1.Session() as sess:
    #     saver.restore(sess, 'path/to/tf1/model.ckpt')
    #     output = sess.run(model.pred, feed_dict={X: arr})
    # np.save('output_tf1.npy', output)
    # print('TF1 output saved to output_tf1.npy')
    # --- Compare ---
    try:
        out_tf1 = np.load("output_tf1.npy")
        print("Comparing outputs...")
        print("Shape TF1:", out_tf1.shape, "Shape TF2:", out_tf2.shape)
        # Align time dimension by center-cropping to minimum length
        def center_crop_time(a, target_nt):
            # a: (batch, nt, 1, classes) or (batch, nt, classes)
            if a.ndim == 4 and a.shape[2] == 1:
                nt = a.shape[1]
                start = (nt - target_nt) // 2
                return a[:, start:start+target_nt, :, :]
            elif a.ndim == 3:
                nt = a.shape[1]
                start = (nt - target_nt) // 2
                return a[:, start:start+target_nt, :]
            else:
                raise ValueError(f"Unsupported array shape for center cropping: {a.shape}")

        tf1_nt = out_tf1.shape[1]
        tf2_nt = out_tf2.shape[1]
        min_nt = min(tf1_nt, tf2_nt)
        if min_nt <= 0:
            raise RuntimeError("One of the model outputs has zero time length")

        out_tf1_c = center_crop_time(out_tf1, min_nt) if tf1_nt != min_nt else out_tf1
        out_tf2_c = center_crop_time(out_tf2, min_nt) if tf2_nt != min_nt else out_tf2

        # Save aligned outputs for further inspection
        np.save('output_tf1_aligned.npy', out_tf1_c)
        np.save('output_tf2_aligned.npy', out_tf2_c)

        # Compute diffs over the aligned arrays
        diff = np.abs(out_tf1_c - out_tf2_c)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        allclose = bool(np.allclose(out_tf1_c, out_tf2_c, atol=1e-5))
        print("Max abs diff:", max_abs)
        print("Mean abs diff:", mean_abs)
        print("Allclose (1e-5):", allclose)

        # Find location of largest difference
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        # idx is a tuple (batch, t, 1, class) or (batch, t, class)
        with open('tf1_tf2_diff_summary.txt', 'w') as fp:
            fp.write(f"TF1 shape: {out_tf1.shape}\n")
            fp.write(f"TF2 shape: {out_tf2.shape}\n")
            fp.write(f"Aligned time length: {min_nt}\n")
            fp.write(f"Max abs diff: {max_abs}\n")
            fp.write(f"Mean abs diff: {mean_abs}\n")
            fp.write(f"Allclose (1e-5): {allclose}\n")
            fp.write(f"Index of max diff: {idx}\n")

        # Also save the diff array (may be large)
        np.save('output_tf_diff.npy', diff)
        print("Saved aligned arrays and diff to output_tf1_aligned.npy, output_tf2_aligned.npy, output_tf_diff.npy and tf1_tf2_diff_summary.txt")
    except Exception as e:
        print("TF1 output not found or error loading. Run the TF1 block in the easyquake environment first.")
        print(e)
