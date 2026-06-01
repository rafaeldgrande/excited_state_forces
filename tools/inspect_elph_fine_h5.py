"""Print all groups, datasets, and attributes in an elph_fine.h5 file."""
import sys
import numpy as np
import h5py

fname = sys.argv[1] if len(sys.argv) > 1 else 'elph_fine.h5'

def print_item(name, obj):
    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        shape_str = str(arr.shape)
        dtype_str = str(arr.dtype)
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
            val_str = f'  min={arr.min():.6g}  max={arr.max():.6g}'
        elif arr.size <= 50:
            val_str = f'  values={arr.tolist()}'
        else:
            val_str = ''
        print(f'  DATASET  {name:<45} shape={shape_str:<20} dtype={dtype_str}{val_str}')
        for k, v in obj.attrs.items():
            print(f'           .attrs[{k!r}] = {v!r}')
    elif isinstance(obj, h5py.Group):
        print(f'  GROUP    {name}')
        for k, v in obj.attrs.items():
            print(f'           .attrs[{k!r}] = {v!r}')

with h5py.File(fname, 'r') as f:
    print(f'File: {fname}')
    print('File-level attributes:')
    for k, v in f.attrs.items():
        print(f'  {k!r} = {v!r}')
    print()
    print('Contents:')
    f.visititems(print_item)
