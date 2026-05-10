
import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser(
    description=(
        'Merge one or more exc_forces.h5 files (produced by excited_forces.py with '
        'save_forces_h5 True) into a single exciton-phonon coupling file. '
        'The output has the same schema as exc_forces.h5 and can be passed directly '
        'to susceptibility_tensors_first/second_order.py.'
    )
)
parser.add_argument('--input', '-i', nargs='+', required=True,
                    help='exc_forces.h5 file(s) to merge (one or more)')
parser.add_argument('--output', '-o', default='exciton_phonon_couplings.h5',
                    help='Output HDF5 file (default: exciton_phonon_couplings.h5)')
args = parser.parse_args()

all_pairs    = []
all_rpa_diag = []
all_rpa      = []
all_rpa_k    = []
phonon_frequencies = None

for fname in args.input:
    print(f'Reading {fname}')
    with h5py.File(fname, 'r') as hf:
        pairs    = hf['exciton_pairs'][:]                   # (Npairs, 2)
        rpa_diag = hf['forces/ph/RPA_diag'][:]              # (Npairs, Nmodes)
        rpa      = hf['forces/ph/RPA'][:]                   # (Npairs, Nmodes)
        rpa_k    = hf['forces/ph/RPA_diag_plus_Kernel'][:]  # (Npairs, Nmodes)

        all_pairs.append(pairs)
        all_rpa_diag.append(rpa_diag)
        all_rpa.append(rpa)
        all_rpa_k.append(rpa_k)

        if phonon_frequencies is None and 'system/phonon_frequencies' in hf:
            phonon_frequencies = hf['system/phonon_frequencies'][:]
            print(f'  Loaded phonon frequencies ({len(phonon_frequencies)} modes) from {fname}')

all_pairs_arr    = np.concatenate(all_pairs,    axis=0)  # (Npairs_total, 2)
all_rpa_diag_arr = np.concatenate(all_rpa_diag, axis=0)  # (Npairs_total, Nmodes)
all_rpa_arr      = np.concatenate(all_rpa,      axis=0)
all_rpa_k_arr    = np.concatenate(all_rpa_k,    axis=0)

# Deduplicate: keep first occurrence of each (i, j) pair
seen  = set()
keep  = []
dupes = 0
for k, (i, j) in enumerate(all_pairs_arr.tolist()):
    key = (int(i), int(j))
    if key not in seen:
        seen.add(key)
        keep.append(k)
    else:
        dupes += 1

if dupes:
    print(f'WARNING: {dupes} duplicate pair(s) found — keeping first occurrence')
keep = np.array(keep)

pairs_out    = all_pairs_arr[keep]
rpa_diag_out = all_rpa_diag_arr[keep]
rpa_out      = all_rpa_arr[keep]
rpa_k_out    = all_rpa_k_arr[keep]

Npairs = len(pairs_out)
Nmodes = rpa_out.shape[1]
max_exc = int(pairs_out.max())

print(f'Total pairs: {Npairs},  Nmodes: {Nmodes},  max exciton index: {max_exc}')

if phonon_frequencies is None:
    print('WARNING: phonon_frequencies not found in any input file. '
          'Run excited_forces.py with save_forces_h5 True to store system/phonon_frequencies.')

with h5py.File(args.output, 'w') as hf:
    hf.create_dataset('exciton_pairs', data=pairs_out)
    hf['exciton_pairs'].attrs['description'] = 'Exciton pair (iexc, jexc) for each row (1-based)'

    grp_ph = hf.require_group('forces/ph')

    grp_ph.create_dataset('RPA_diag', data=rpa_diag_out)
    grp_ph['RPA_diag'].attrs['description'] = (
        'F_nu = -<iexc|dH/dQ_nu|jexc> in the RPA_diag approximation. '
        'Negate to get exciton-phonon matrix elements.')
    grp_ph['RPA_diag'].attrs['units'] = 'eV/ang'

    grp_ph.create_dataset('RPA', data=rpa_out)
    grp_ph['RPA'].attrs['description'] = (
        'F_nu = -<iexc|dH/dQ_nu|jexc> in the full RPA. '
        'Negate to get exciton-phonon matrix elements.')
    grp_ph['RPA'].attrs['units'] = 'eV/ang'

    grp_ph.create_dataset('RPA_diag_plus_Kernel', data=rpa_k_out)
    grp_ph['RPA_diag_plus_Kernel'].attrs['units'] = 'eV/ang'

    grp_sys = hf.require_group('system')
    grp_sys.attrs['Nmodes']              = Nmodes
    grp_sys.attrs['Npairs']              = Npairs
    grp_sys.attrs['max_exciton_index']   = max_exc

    if phonon_frequencies is not None:
        grp_sys.create_dataset('phonon_frequencies', data=phonon_frequencies)
        grp_sys['phonon_frequencies'].attrs['units'] = 'cm^-1'

print(f'Saved {Npairs} pairs to {args.output}')
