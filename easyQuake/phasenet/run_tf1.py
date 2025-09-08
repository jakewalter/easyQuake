import numpy as np
import sys, os
arr = np.load('phasenet/input_waveform.npy')
print("arr.shape:", arr.shape)
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()
# Check for TF1-appropriate APIs that are unavailable under TF2/Keras3.
try:
    _ = tf1.layers
except Exception:
    raise RuntimeError(
        "This script requires the original TensorFlow 1.x environment.\n"
        "Activate the 'easyquake' conda environment (which contains TF1) and rerun this script.\n"
        "For example:\n  conda activate easyquake\n  python phasenet/run_tf1.py\n"
        "Ensure the TF1 checkpoint path in this script is set to the TF1 checkpoint file (not a directory)."
    )
sys.path.append(os.path.dirname(__file__))
from phasenet.phasenet_original.phasenet.model import ModelConfig as ModelConfigTF1, UNet as UNetTF1
config = ModelConfigTF1(X_shape=[3000, 1, 3])
# Create placeholder with batch dim
X = tf1.placeholder(tf1.float32, shape=[None, 3000, 1, 3], name="X")
# Pass input_batch as a sequence so the TF1 UNet assigns self.X = input_batch[0]
model = UNetTF1(config=config, input_batch=(X,), mode="pred")
saver = tf1.train.Saver()
with tf1.Session() as sess:
    # Resolve checkpoint: if the path is a directory, find the latest checkpoint inside it
    ckpt_path = '/home/jwalter/easyQuake/easyQuake/phasenet/model/190703-214543/'
    if os.path.isdir(ckpt_path):
        ckpt = tf1.train.latest_checkpoint(ckpt_path)
        if ckpt is None:
            raise RuntimeError(f"No TensorFlow 1.x checkpoint found in directory: {ckpt_path}.\n"
                               "Place the TF1 checkpoint files (.ckpt-xxxx and .meta) there or edit this script to point to the exact checkpoint file.")
    else:
        ckpt = ckpt_path

    saver.restore(sess, ckpt)
    # Ensure dropout/is_training placeholders are set for prediction
    feed = {model.X: arr, model.drop_rate: 0.0, model.is_training: False}
    # TF1 model stores predictions in `preds` (plural)
    output = sess.run(model.preds, feed_dict=feed)

np.save('output_tf1.npy', output)
print('TF1 output saved to output_tf1.npy')