import os
import re

def update_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Replace standalone keras imports with tf.keras
    content = re.sub(r'^\s*import keras\b', 'import tensorflow.keras as keras', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*from keras\b', 'from tensorflow.keras', content, flags=re.MULTILINE)
    # Replace keras. references with tf.keras. (if not already handled)
    content = re.sub(r'(?<!tensorflow\.)keras\.', 'tf.keras.', content)
    # If keras is used as a namespace, ensure keras = tf.keras is present
    if re.search(r'keras\.', content) and 'keras = tf.keras' not in content:
        # Add alias after first tf import
        content = re.sub(r'(import tensorflow as tf\s*)', r'\1\nkeras = tf.keras\n', content, count=1)
    # Remove tf.compat.v1.disable_eager_execution()
    content = re.sub(r'^\s*tf\.compat\.v1\.disable_eager_execution\(\)\s*', '', content, flags=re.MULTILINE)
    # Remove tf.Session() and tf.ConfigProto()
    content = re.sub(r'tf\.Session\(\)', '', content)
    content = re.sub(r'tf\.ConfigProto\(\)', '', content)
    # Remove tf.global_variables_initializer()
    content = re.sub(r'tf\.global_variables_initializer\(\)', '', content)

    with open(filepath, "w") as f:
        f.write(content)

def update_codebase(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                update_file(filepath)
                print(f"Updated {filepath}")

if __name__ == "__main__":
    # Change '.' to your project root if needed
    update_codebase("..")
