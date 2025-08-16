import pytest
import numpy as np
import h5py
import tempfile
from easyQuake.EQTransformer import EqT_utils

def test_DataGenerator_basic():
    # Create a fake HDF5 file with minimal structure
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        with h5py.File(tmp.name, 'w') as f:
            grp = f.create_group('data')
            dset = grp.create_dataset('test_EV', data=np.random.randn(6000, 3))
            dset.attrs['p_arrival_sample'] = 1000
            dset.attrs['s_arrival_sample'] = 2000
            dset.attrs['coda_end_sample'] = 3000
            dset.attrs['snr_db'] = np.array([20.0, 20.0, 20.0])
            dset.attrs['trace_category'] = 'earthquake_local'
        gen = EqT_utils.DataGenerator(
            list_IDs=['test_EV'],
            file_name=tmp.name,
            dim=6000,
            batch_size=1,
            n_channels=3,
            phase_window=40,
            shuffle=False,
            norm_mode='max',
            label_type='gaussian',
            augmentation=False
        )
        X, y1, y2, y3 = gen.__getitem__(0)[0]['input'], gen.__getitem__(0)[1]['detector'], gen.__getitem__(0)[1]['picker_P'], gen.__getitem__(0)[1]['picker_S']
        assert X.shape == (1, 6000, 3)
        assert y1.shape == (1, 6000, 1)
        assert y2.shape == (1, 6000, 1)
        assert y3.shape == (1, 6000, 1)
        # Check normalization
        assert np.allclose(np.max(X), 1.0, atol=1e-2) or np.allclose(np.max(X), 0.0, atol=1e-2)

def test_label_function():
    gen = EqT_utils.DataGenerator(
        list_IDs=[], file_name='', dim=40, batch_size=1, n_channels=3
    )
    label = gen._label(0, 20, 40)
    assert isinstance(label, np.ndarray)
    assert label.shape[0] == 41
    assert np.all(label >= 0)
    assert np.all(label <= 1)
