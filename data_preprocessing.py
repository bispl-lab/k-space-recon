import h5py
import numpy as np
from pathlib import Path


"""
Code to transform the original dataset into a form that has faster data IO.
This is especially important for non-SSD storage.
"""


def make_compressed_dataset(data_folder, save_dir, **kwargs):
    data_path = Path(data_folder)
    files = data_path.glob('*.h5')

    save_path = Path(save_dir) / ('new_' + data_path.stem)
    save_path.mkdir()

    for file in files:
        print(f'Processing {file}')
        with h5py.File(file, mode='r') as old_hf:
            attrs = dict(old_hf.attrs)
            kspace = np.asarray(old_hf['kspace'])
            try:
                esc = np.asarray(old_hf['reconstruction_esc'])
                # rss = np.asarray(old_hf['reconstruction_rss'])
                test_set = False
            except KeyError:
                test_set = True

        if kspace.ndim == 3:  # Single-coil case
            chunk = (1, kspace.shape[-2], kspace.shape[-1])
        elif kspace.ndim == 4:
            chunk = (1, 1, kspace.shape[-2], kspace.shape[-1])
        else:
            raise TypeError('Invalid dimensions of input k-space data')

        with h5py.File(save_path / (file.stem + '.h5'), mode='x', libver='latest') as new_hf:
            new_hf.attrs.update(attrs)
            new_hf.create_dataset('kspace', data=kspace, chunks=chunk, **kwargs)
            if not test_set:
                new_hf.create_dataset('reconstruction_esc', data=esc, chunks=(1, 320, 320), **kwargs)


def check_same(old_folder, new_folder):
    old_path = Path(old_folder)
    new_path = Path(new_folder)

    old_paths = list(old_path.glob('*.h5'))
    new_paths = list(new_path.glob('*.h5'))

    assert len(old_paths) == len(new_paths)

    for old, new in zip(old_paths, new_paths):
        print(f'Checking {new}')
        with h5py.File(old, mode='r') as old_hf, h5py.File(new, mode='r') as new_hf:
            assert dict(new_hf.attrs) == dict(old_hf.attrs)

            for key, value in new_hf.values():
                assert np.all(old_hf['key'] == value)
    else:
        print('All is well!')


if __name__ == '__main__':
    train_dir = '/media/veritas/D/fastMRI/singlecoil_train'
    val_dir = '/media/veritas/D/fastMRI/singlecoil_val'
    test_dir = '/media/veritas/D/fastMRI/singlecoil_test'

    data_root = '/media/veritas/D/fastMRI'

    gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)

    make_compressed_dataset(test_dir, data_root, **gzip)  # Use compression if storing on hard drive, not SSD.


