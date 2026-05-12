"""
elph_in_ph_basis.py

Converts electron-phonon matrix elements from the displacement-pattern basis
(as stored in elph_coeffs.h5 by excited_forces.py) to the phonon eigenvector
basis (as read from a modes.axsf file).

The transformation is:
    elph_ph[μ, k, i, j] = Σ_ν  T[μ, ν] · elph_h5[ν, k, i, j]

where T[μ, ν] = Σ_{I,α} e_mw_norm[μ, I, α] · dp[ν, I, α]

and e_mw_norm are the mass-weighted, normalized phonon eigenvectors from modes.axsf,
dp are the (already orthonormal) displacement patterns from elph_coeffs.h5.

Usage:
    python elph_in_ph_basis.py [--elph_file ELPH_FILE] [--modes_file MODES_FILE]
                                [--output OUTPUT] [--atom_masses M1 M2 ...]

Defaults:
    --elph_file   elph_coeffs.h5
    --modes_file  modes.axsf
    --output      elph_in_ph_basis.h5
    --atom_masses  (read from elph_coeffs.h5 if present, otherwise must be provided)
"""

import argparse
import numpy as np
import h5py
import sys

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--elph_file',   default='elph_coeffs.h5',
                    help='HDF5 file with electron-phonon matrix elements in displacement-pattern basis (default: elph_coeffs.h5)')
parser.add_argument('--modes_file',  default='modes.axsf',
                    help='AXSF file containing phonon eigenvectors (ANIMSTEPS format; default: modes.axsf)')
parser.add_argument('--output',      default='elph_in_ph_basis.h5',
                    help='Output HDF5 file name (default: elph_in_ph_basis.h5)')
parser.add_argument('--atom_masses', type=float, nargs='+', default=None,
                    help='Atomic masses in amu, one per atom (e.g. 12.011 12.011 ... 1.008 1.008 ...)')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Parse modes.axsf  →  eigvecs (nmodes, natoms, 3)
# ---------------------------------------------------------------------------
def parse_axsf(filename):
    """
    Read phonon eigenvectors from an AXSF ANIMSTEPS file.
    Returns array of shape (nmodes, natoms, 3).
    The vectors are the displacement part (columns 4-6) in each PRIMCOORD block.
    """
    eigvecs = []
    symbols = []
    with open(filename) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('PRIMCOORD'):
            i += 1
            natoms_block = int(lines[i].split()[0])
            i += 1
            evec = []
            syms = []
            for _ in range(natoms_block):
                parts = lines[i].split()
                syms.append(parts[0])
                evec.append([float(parts[4]), float(parts[5]), float(parts[6])])
                i += 1
            eigvecs.append(evec)
            if not symbols:
                symbols = syms
        else:
            i += 1
    return np.array(eigvecs), symbols   # (nmodes, natoms, 3), list of symbols


print(f'Parsing phonon eigenvectors from {args.modes_file} ...')
ev_axsf, atom_symbols = parse_axsf(args.modes_file)
nmodes_axsf, natoms, _ = ev_axsf.shape
print(f'  Found {nmodes_axsf} modes, {natoms} atoms')
print(f'  Atom symbols: {atom_symbols}')


# ---------------------------------------------------------------------------
# 2. Determine atomic masses
# ---------------------------------------------------------------------------
# Default mass table (amu)
MASS_TABLE = {
    'H':  1.00794,  'He': 4.00260,
    'Li': 6.94100,  'Be': 9.01218,  'B': 10.81100,  'C': 12.01100,
    'N': 14.00670,  'O': 15.99940,  'F': 18.99840,  'Ne': 20.17900,
    'Si': 28.08550, 'P': 30.97376,  'S': 32.06000,  'Cl': 35.45300,
    'Br': 79.90400, 'I': 126.90450,
}

if args.atom_masses is not None:
    if len(args.atom_masses) != natoms:
        sys.exit(f'ERROR: --atom_masses has {len(args.atom_masses)} entries but {natoms} atoms found in modes file.')
    masses = np.array(args.atom_masses)
    print(f'  Using masses from command line: {masses}')
else:
    try:
        masses = np.array([MASS_TABLE[s] for s in atom_symbols])
        print(f'  Auto-determined masses from symbols: {masses}')
    except KeyError as e:
        sys.exit(f'ERROR: Unknown element {e}. Please provide --atom_masses.')


# ---------------------------------------------------------------------------
# 3. Mass-weight and normalize eigenvectors
# ---------------------------------------------------------------------------
# ev_mw[μ, I, α] = ev_axsf[μ, I, α] * sqrt(M_I)
sqrt_masses = np.sqrt(masses)                              # (natoms,)
ev_mw = ev_axsf * sqrt_masses[np.newaxis, :, np.newaxis]  # (nmodes, natoms, 3)

# Normalize each mode
norms = np.linalg.norm(ev_mw.reshape(nmodes_axsf, -1), axis=1)  # (nmodes,)
print(f'  Mass-weighted eigenvector norms (before normalization): min={norms.min():.4f}, max={norms.max():.4f}')
ev_mw_norm = ev_mw / norms[:, np.newaxis, np.newaxis]            # (nmodes, natoms, 3)


# ---------------------------------------------------------------------------
# 4. Read elph_coeffs.h5
# ---------------------------------------------------------------------------
print(f'\nReading electron-phonon data from {args.elph_file} ...')
with h5py.File(args.elph_file, 'r') as hf:
    dp = hf['displacement_patterns'][()]          # (nmodes_h5, natoms, 3)
    elph_val  = hf['elph_val'][()]                # (nmodes_h5, nk, nv, nv)
    elph_cond = hf['elph_cond'][()]               # (nmodes_h5, nk, nc, nc)

    # Optional not-renormalized datasets
    has_notrenorm = ('elph_val_not_renorm' in hf) and ('elph_cond_not_renorm' in hf)
    if has_notrenorm:
        elph_val_nr  = hf['elph_val_not_renorm'][()]
        elph_cond_nr = hf['elph_cond_not_renorm'][()]

    # Copy all scalar attributes for documentation
    attrs = dict(hf.attrs)

nmodes_h5 = dp.shape[0]
print(f'  displacement_patterns shape : {dp.shape}')
print(f'  elph_val shape              : {elph_val.shape}')
print(f'  elph_cond shape             : {elph_cond.shape}')
if has_notrenorm:
    print(f'  elph_val_not_renorm shape   : {elph_val_nr.shape}')
    print(f'  elph_cond_not_renorm shape  : {elph_cond_nr.shape}')

if nmodes_axsf != nmodes_h5:
    sys.exit(f'ERROR: modes.axsf has {nmodes_axsf} modes but elph_coeffs.h5 has {nmodes_h5} displacement patterns.')
if natoms != dp.shape[1]:
    sys.exit(f'ERROR: modes.axsf has {natoms} atoms but elph_coeffs.h5 has {dp.shape[1]} atoms.')


# ---------------------------------------------------------------------------
# 5. Build transformation matrix T[μ_axsf, ν_h5]
# ---------------------------------------------------------------------------
# T[μ, ν] = Σ_{I,α}  e_mw_norm[μ, I, α] · dp[ν, I, α]
T = np.einsum('nia, mia -> nm', ev_mw_norm, dp)   # (nmodes, nmodes)

# Sanity check: T should be nearly unitary
TT = T @ T.T
deviation = np.max(np.abs(TT - np.eye(nmodes_h5)))
print(f'\nTransformation matrix T shape: {T.shape}')
print(f'Unitarity check  max|T·Tᵀ - I| = {deviation:.3e}')
if deviation > 0.05:
    print('WARNING: T is not very unitary — check mass assignments and eigenvector normalization.')


# ---------------------------------------------------------------------------
# 6. Transform electron-phonon matrix elements
# ---------------------------------------------------------------------------
# elph_ph[μ, ...] = Σ_ν  T[μ, ν] · elph_h5[ν, ...]
print('\nTransforming electron-phonon matrix elements to phonon basis ...')

def transform(T, arr):
    """Apply mode transformation T[μ,ν] to first axis of arr[ν,...]."""
    return np.einsum('mn, n... -> m...', T, arr)

elph_val_ph  = transform(T, elph_val)
elph_cond_ph = transform(T, elph_cond)
print(f'  elph_val_ph  shape: {elph_val_ph.shape}')
print(f'  elph_cond_ph shape: {elph_cond_ph.shape}')

if has_notrenorm:
    elph_val_nr_ph  = transform(T, elph_val_nr)
    elph_cond_nr_ph = transform(T, elph_cond_nr)
    print(f'  elph_val_not_renorm_ph  shape: {elph_val_nr_ph.shape}')
    print(f'  elph_cond_not_renorm_ph shape: {elph_cond_nr_ph.shape}')


# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
print(f'\nSaving results to {args.output} ...')
with h5py.File(args.output, 'w') as hf:

    # Transformed e-ph matrix elements
    hf.create_dataset('elph_val',  data=elph_val_ph,  compression='gzip')
    hf.create_dataset('elph_cond', data=elph_cond_ph, compression='gzip')
    if has_notrenorm:
        hf.create_dataset('elph_val_not_renorm',  data=elph_val_nr_ph,  compression='gzip')
        hf.create_dataset('elph_cond_not_renorm', data=elph_cond_nr_ph, compression='gzip')

    # Phonon eigenvectors (mass-weighted and normalized) used for the transformation
    hf.create_dataset('phonon_eigvecs_mw_norm', data=ev_mw_norm, compression='gzip')
    hf.create_dataset('phonon_eigvecs_raw',     data=ev_axsf,    compression='gzip')

    # Transformation matrix
    hf.create_dataset('T_matrix', data=T, compression='gzip')

    # Atom info
    hf.create_dataset('atom_masses',  data=masses)
    hf.create_dataset('atom_symbols', data=np.array(atom_symbols, dtype='S4'))

    # Propagate original attributes
    for k, v in attrs.items():
        try:
            hf.attrs[k] = v
        except Exception:
            pass

    # Add provenance attributes
    hf.attrs['elph_basis']         = 'phonon_eigenvectors'
    hf.attrs['source_elph_file']   = args.elph_file
    hf.attrs['source_modes_file']  = args.modes_file
    hf.attrs['unitarity_deviation'] = deviation
    hf.attrs['description'] = (
        'Electron-phonon matrix elements transformed from the displacement-pattern basis '
        '(elph_coeffs.h5) to the phonon eigenvector basis (modes.axsf). '
        'Transformation: elph_ph[mu,...] = sum_nu T[mu,nu] * elph_h5[nu,...], '
        'where T[mu,nu] = sum_{I,alpha} e_mw_norm[mu,I,alpha] * dp[nu,I,alpha].'
    )

print('Done!')
print(f'Output datasets:')
with h5py.File(args.output, 'r') as hf:
    def print_tree(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f'  {name:40s}  shape={obj.shape}  dtype={obj.dtype}')
    hf.visititems(print_tree)
