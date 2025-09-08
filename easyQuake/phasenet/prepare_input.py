import numpy as np
import glob
from obspy import read

# Path pattern for three miniseed files (E, N, Z)
MSEED_PATTERN = "/tmp/O2.WILZ.EH?.mseed"

# Output file for normalized input
OUT_NPY = "input_waveform.npy"

def load_and_normalize_mseed(pattern):
    files = sorted(glob.glob(pattern))
    if len(files) != 3:
        raise ValueError(f"Expected 3 miniseed files, found: {files}")
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
    data = np.stack(arrs, axis=-1)
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True) + 1e-6
    normed = (data - mean) / std
    normed = normed[:, np.newaxis, :]
    if normed.shape[0] > 3000:
        normed = normed[:3000, :, :]
    elif normed.shape[0] < 3000:
        pad = np.zeros((3000 - normed.shape[0], 1, 3), dtype=np.float32)
        normed = np.concatenate([normed, pad], axis=0)
    return normed[np.newaxis, ...]

if __name__ == "__main__":
    arr = load_and_normalize_mseed(MSEED_PATTERN)
    print(f"Saving input waveform: {arr.shape} -> {OUT_NPY}")
    np.save(OUT_NPY, arr)
    print("Done.")
