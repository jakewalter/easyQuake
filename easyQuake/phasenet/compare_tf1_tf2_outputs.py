import numpy as np
import tensorflow as tf
import sys
import os

# --- User must set these paths ---

# Path pattern for three miniseed files (E, N, Z)
MSEED_PATTERN = "/tmp/O2.WILZ.EH?.mseed"
# Path to TF1 checkpoint directory (should contain .meta, .index, .data-*)
TF1_CKPT = "path/to/tf1/model.ckpt"
# Path to TF2/Keras3 .h5 or .keras file
TF2_H5 = "path/to/tf2/model.h5"

import glob
from obspy import read

# --- Helper: Load and normalize miniseed files ---
def load_and_normalize_mseed(pattern):
    files = sorted(glob.glob(pattern))
    if len(files) != 3:
        raise ValueError(f"Expected 3 miniseed files, found: {files}")
    # Read and stack traces in E, N, Z order
    comps = {'E': None, 'N': None, 'Z': None}
    for f in files:
        st = read(f)
        for tr in st:
            c = tr.stats.channel[-1]
            if c in comps:
                comps[c] = tr.data.astype(np.float32)
    arrs = [comps[c] for c in 'ENZ']
    if any(a is None for a in arrs):
        raise ValueError(f"Missing one or more components in {files}")
    # Stack to shape (nt, 3)
    data = np.stack(arrs, axis=-1)
    # Normalize (mean/std per channel)
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True) + 1e-6
    normed = (data - mean) / std
    # Reshape to (nt, 1, 3)
    normed = normed[:, np.newaxis, :]
    # If longer than 3000, take first 3000
    if normed.shape[0] > 3000:
        normed = normed[:3000, :, :]
    elif normed.shape[0] < 3000:
        # Pad with zeros
        pad = np.zeros((3000 - normed.shape[0], 1, 3), dtype=np.float32)
        normed = np.concatenate([normed, pad], axis=0)
    return normed[np.newaxis, ...]  # Add batch dim

# --- TF1 Model Import ---
def run_tf1_model(input_array):
    import tensorflow.compat.v1 as tf1
    tf1.disable_eager_execution()
    sys.path.append(os.path.dirname(__file__))
    from phasenet.phasenet_original.phasenet.model import ModelConfig as ModelConfigTF1, UNet as UNetTF1
    config = ModelConfigTF1(X_shape=[3000, 1, 3])
    X = tf1.placeholder(tf.float32, shape=[None, 3000, 1, 3], name="X")
    model = UNetTF1(config=config, mode="pred", input=X)
    saver = tf1.train.Saver()
    with tf1.Session() as sess:
        saver.restore(sess, TF1_CKPT)
        output = sess.run(model.pred, feed_dict={X: input_array})
    return output

# --- TF2 Model Import ---
def run_tf2_model(input_array):
    from phasenet.model_tf2 import ModelConfigTF2, UNet
    config = ModelConfigTF2(X_shape=[3000, 1, 3])
    model = UNet(config=config, mode="pred")
    model.model.load_weights(TF2_H5)
    output = model.model.predict(input_array)
    return output

if __name__ == "__main__":
    # Load and normalize miniseed
    arr = load_and_normalize_mseed(MSEED_PATTERN)
    print(f"Input shape: {arr.shape}")
    # Run TF1
    print("Running TF1 model...")
    out_tf1 = run_tf1_model(arr)
    np.save("output_tf1.npy", out_tf1)
    print("TF1 output saved to output_tf1.npy")
    # Run TF2
    print("Running TF2 model...")
    out_tf2 = run_tf2_model(arr)
    np.save("output_tf2.npy", out_tf2)
    print("TF2 output saved to output_tf2.npy")
    # Compare
    print("Comparing outputs...")
    print("Shape TF1:", out_tf1.shape, "Shape TF2:", out_tf2.shape)
    print("Max abs diff:", np.max(np.abs(out_tf1 - out_tf2)))
    print("Mean abs diff:", np.mean(np.abs(out_tf1 - out_tf2)))
    print("Allclose (1e-5):", np.allclose(out_tf1, out_tf2, atol=1e-5))
