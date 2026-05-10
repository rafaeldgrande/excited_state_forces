
import sys
from pathlib import Path
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (k_B, rec_cm_to_eV, hbar, FLAVOR_DESC,
                    ignore_0_freq_modes, _downsample_idx, unpolarized_invariant)

config_dir = Path(__file__).parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

parser = argparse.ArgumentParser(description='Compute resonant Raman intensity maps')
parser.add_argument('--temperature',       type=float, default=300,
                    help='Temperature in Kelvin (default: 300)')
parser.add_argument('--first-order-file',  type=str,
                    default='susceptibility_tensors_first_order.h5',
                    help='HDF5 file from susceptibility_tensors_first_order.py')
parser.add_argument('--second-order-file', type=str,
                    default='susceptibility_tensors_second_order.h5',
                    help='HDF5 file from susceptibility_tensors_second_order.py '
                         '(required for flavors 2 and 3)')
parser.add_argument('--ipa-first-order-file', type=str,
                    default='susceptibility_tensors_first_order_IPA.h5',
                    help='HDF5 file from susceptibility_tensors_IPA.py '
                         '(required for flavors 6 and 8)')
parser.add_argument('--ipa-second-order-file', type=str,
                    default='susceptibility_tensors_second_order_IPA.h5',
                    help='HDF5 file from susceptibility_tensors_IPA.py '
                         '(required for flavors 7 and 8)')
parser.add_argument('--freqs-file',        type=str, default=None,
                    help='File with phonon frequencies in cm^-1 (optional; read from susceptibility h5 if not given)')
parser.add_argument('--flavor',            type=int, default=0,
                    choices=list(FLAVOR_DESC.keys()),
                    help='Which susceptibility to use: ' +
                         ', '.join(f'{k}={v}' for k, v in FLAVOR_DESC.items()))
parser.add_argument('--output',            type=str, default='resonant_raman_data.h5',
                    help='Output HDF5 file (default: resonant_raman_data.h5)')
parser.add_argument('--nfreq-ph',          type=int, default=None,
                    help='Number of points on the phonon frequency axis. '
                         'Default: auto-set so that step = gamma_lor / 5.')
parser.add_argument('--nfreq-exc',         type=int, default=None,
                    help='Down-sample the excitation energy axis to this many points '
                         'before building the map (reduces memory and file size). '
                         'Default: keep all points.')
parser.add_argument('--gamma-lor',         type=float, default=10.0,
                    help='Lorentzian phonon linewidth in cm^-1 (default: 10). '
                         'Set to match your experimental resolution or phonon lifetime.')
parser.add_argument('--plot-map-log-scale', action='store_true',
                    help='Plot log(max(I, 1e-4)) instead of I in the 2-D maps')
args = parser.parse_args()

T                    = args.temperature
first_order_file     = args.first_order_file
second_order_file    = args.second_order_file
ipa_first_order_file = args.ipa_first_order_file
ipa_second_order_file= args.ipa_second_order_file
freqs_file           = args.freqs_file
flavor               = args.flavor
output_file          = args.output
gamma_lor            = args.gamma_lor          # cm^-1
nfreq_exc_target     = args.nfreq_exc          # None → keep all
plot_map_log_scale   = args.plot_map_log_scale

flavor_label = FLAVOR_DESC[flavor]
print(f'Flavor {flavor}: {flavor_label}')

# Derived flags
is_ipa           = flavor in {6, 7, 8}
has_first_order  = flavor in {0, 1, 4, 5, 6, 8}
use_d2           = flavor == 0
has_second_order = flavor in {2, 3, 4, 5, 7, 8}
has_double       = flavor in {3, 5}

cart_dir = ['x', 'y', 'z']

freqs_rec_cm = None
# Try to read phonon frequencies from whichever susceptibility h5 file is available
for _h5 in [first_order_file, second_order_file, ipa_first_order_file, ipa_second_order_file]:
    try:
        with h5py.File(_h5, 'r') as _hf:
            if 'phonon_frequencies_cm' in _hf:
                freqs_rec_cm = _hf['phonon_frequencies_cm'][:]
                print(f'Phonon frequencies read from {_h5}')
                break
    except (FileNotFoundError, OSError):
        pass
if freqs_rec_cm is None:
    if freqs_file is not None:
        freqs_rec_cm = np.loadtxt(freqs_file)
        print(f'Phonon frequencies read from {freqs_file}')
    else:
        sys.exit('ERROR: phonon frequencies not found in any susceptibility h5 file and --freqs-file not provided.')
freqs_eV = freqs_rec_cm * rec_cm_to_eV
Nmodes   = len(freqs_rec_cm)

# ---------------------------------------------------------------------------
# Load susceptibility tensors
# ---------------------------------------------------------------------------
alpha_tensor_first_order = None
excitation_energies_1st  = None
if has_first_order and not is_ipa:
    print(f'Reading first-order susceptibilities from {first_order_file}')
    with h5py.File(first_order_file, 'r') as f:
        excitation_energies_1st = f['excitation_energies'][:]
        alpha_tensor_d2         = f['alpha_tensor_d2'][:]
        alpha_tensor_d3         = f['alpha_tensor_d3'][:]
    alpha_tensor_first_order = alpha_tensor_d2 if use_d2 else alpha_tensor_d3

alpha_tensor_second_order = None
excitation_energies_2nd   = None
if has_second_order and not is_ipa:
    print(f'Reading second-order susceptibilities from {second_order_file}')
    with h5py.File(second_order_file, 'r') as f:
        excitation_energies_2nd   = f['excitation_energies'][:]
        alpha_tensor_second_order = f['alpha_tensor_triple_resonance'][:]
        if has_double:
            alpha_tensor_double_res = f['alpha_tensor_double_resonance'][:]
    if has_double:
        for imode in range(Nmodes):
            alpha_tensor_second_order[:, :, imode, imode, :] += alpha_tensor_double_res[:, :, imode, :]

if flavor in {6, 8}:
    print(f'Reading IPA first-order susceptibilities from {ipa_first_order_file}')
    with h5py.File(ipa_first_order_file, 'r') as f:
        excitation_energies_1st  = f['excitation_energies'][:]
        alpha_tensor_first_order = f['susceptibility_tensor_first_order'][:]

if flavor in {7, 8}:
    print(f'Reading IPA second-order susceptibilities from {ipa_second_order_file}')
    with h5py.File(ipa_second_order_file, 'r') as f:
        excitation_energies_2nd   = f['excitation_energies'][:]
        alpha_tensor_second_order = f['susceptibility_tensor_second_order'][:]

# ---------------------------------------------------------------------------
# Optional down-sampling of the excitation energy axis
# ---------------------------------------------------------------------------
if nfreq_exc_target is not None:
    if has_second_order:
        n2 = len(excitation_energies_2nd)
        if nfreq_exc_target < n2:
            ie2 = _downsample_idx(n2, nfreq_exc_target)
            excitation_energies_2nd   = excitation_energies_2nd[ie2]
            # alpha_tensor_second_order already has double-resonance folded in
            alpha_tensor_second_order = alpha_tensor_second_order[..., ie2]
            print(f'  Excitation axis downsampled: {n2} → {nfreq_exc_target} points')
    if has_first_order:
        n1 = len(excitation_energies_1st)
        if nfreq_exc_target < n1:
            ie1 = _downsample_idx(n1, nfreq_exc_target)
            excitation_energies_1st   = excitation_energies_1st[ie1]
            alpha_tensor_first_order  = alpha_tensor_first_order[..., ie1]

# Main excitation energy grid
excitation_energies = excitation_energies_2nd if has_second_order else excitation_energies_1st

# I(alpha,beta) ∝ |Σ_i  w_i · α¹[i]  +  Σ_ij w_i·w_j · α²[i,j]|²
# where w_i = sqrt((n_i+1) · ħ/(2·ω_i))

Nfreq_1st = excitation_energies_1st.shape[0] if has_first_order else 0
Nfreq     = excitation_energies.shape[0]

safe_freqs_eV = np.maximum(freqs_eV, 1e-8)
bose_occ      = 1.0 / (np.exp(safe_freqs_eV / (k_B * T)) - 1)          # (Nmodes,)
phonon_weight = np.sqrt((bose_occ + 1) * hbar / (2 * safe_freqs_eV))    # (Nmodes,)

def is_valid_mode(imode):
    return not (freqs_rec_cm[imode] < 1e-2 and ignore_0_freq_modes)

# ---------------------------------------------------------------------------
# Build phonon frequency axis
# ---------------------------------------------------------------------------
min_vib_freq = np.min(freqs_rec_cm)
max_vib_freq = np.max(freqs_rec_cm)

freq_axis_lo = max(0.0, min_vib_freq - 5 * gamma_lor)
freq_axis_hi = (2 * max_vib_freq if has_second_order else max_vib_freq) + 5 * gamma_lor
freq_range   = freq_axis_hi - freq_axis_lo

# Auto-set Nfreq_ph so grid step <= gamma_lor / 5 (5 points per linewidth minimum).
# User override via --nfreq-ph is honoured but a warning is printed if too coarse.
Nfreq_ph_auto = max(500, int(np.ceil(freq_range / (gamma_lor / 5))))
if args.nfreq_ph is None:
    Nfreq_ph = Nfreq_ph_auto
    print(f'  gamma_lor = {gamma_lor:.1f} cm^-1  →  auto Nfreq_ph = {Nfreq_ph} '
          f'(step = {freq_range/Nfreq_ph:.2f} cm^-1)')
else:
    Nfreq_ph  = args.nfreq_ph
    step      = freq_range / Nfreq_ph
    if step > gamma_lor / 3:
        print(f'  WARNING: grid step ({step:.2f} cm^-1) > gamma_lor/3 ({gamma_lor/3:.2f} cm^-1). '
              f'Peaks will be undersampled. Consider --nfreq-ph {Nfreq_ph_auto} or larger.')

freq_axis = np.linspace(freq_axis_lo, freq_axis_hi, Nfreq_ph)

# meshgrid: x = phonon freq, y = excitation energy  → both (Nfreq, Nfreq_ph)
freq_grid, excitation_grid = np.meshgrid(freq_axis, excitation_energies)

# ---------------------------------------------------------------------------
# Pre-compute Lorentzians (shared across all polarisations)
# ---------------------------------------------------------------------------
# Valid mode mask (first-order)
valid_modes = np.array([is_valid_mode(i) for i in range(Nmodes)])

# First-order Lorentzians: (Nvalid_1st, Nfreq_ph)
if has_first_order:
    lor_1st = (gamma_lor**2 /
               ((freq_axis[np.newaxis, :] - freqs_rec_cm[valid_modes, np.newaxis])**2
                + gamma_lor**2))                                       # (Nvalid, Nfreq_ph)

# Second-order: pair frequencies and weights
if has_second_order:
    freq_pairs = (freqs_rec_cm[:, np.newaxis] +
                  freqs_rec_cm[np.newaxis, :]).ravel()                 # (Nmodes²,)
    w_pairs    = (phonon_weight[:, np.newaxis] *
                  phonon_weight[np.newaxis, :]).ravel()                # (Nmodes²,)
    if ignore_0_freq_modes:
        # Exclude any pair that contains a near-zero-frequency (acoustic) mode.
        # The simpler cut  freq_pairs >= 1e-2  only removes acoustic+acoustic pairs;
        # acoustic+optical pairs survive with a gigantic Bose-factor phonon weight
        # (w_acoustic → large as ω→0) that would dominate dummy maps and add numerical
        # noise to real ones.  Applying the per-mode filter to both indices fixes this.
        valid_pairs = (np.outer(valid_modes, valid_modes).ravel() &
                       (freq_pairs >= 1e-2))
    else:
        valid_pairs = np.ones(Nmodes**2, dtype=bool)

    freq_pairs_v = freq_pairs[valid_pairs]
    w_pairs_v    = w_pairs[valid_pairs]                                # (Npairs_v,)
    lor_2nd = (gamma_lor**2 /
               ((freq_axis[np.newaxis, :] - freq_pairs_v[:, np.newaxis])**2
                + gamma_lor**2))                                       # (Npairs_v, Nfreq_ph)

# ---------------------------------------------------------------------------
# Compute Raman intensity maps — vectorised
# raman_maps[ialpha, ibeta, iE_exc, iE_ph]
# raman_map_unpol[iE_exc, iE_ph]
# ---------------------------------------------------------------------------
raman_maps      = np.zeros((3, 3, Nfreq, Nfreq_ph))
raman_map_unpol = np.zeros((Nfreq, Nfreq_ph))

print('Computing Raman intensity maps...')
for ialpha in range(3):
    for ibeta in range(3):
        pol = f'{cart_dir[ialpha]}{cart_dir[ibeta]}'
        print(f'  {pol}')
        raman_map = np.zeros((Nfreq, Nfreq_ph))

        # --- First-order ---
        # intensity_1st: (Nvalid, Nfreq), embedded at [:Nfreq_1st]
        if has_first_order:
            alpha_v = alpha_tensor_first_order[ialpha, ibeta, valid_modes, :]  # (Nvalid, Nfreq_1st)
            int_1st = np.zeros((valid_modes.sum(), Nfreq))
            int_1st[:, :Nfreq_1st] = np.abs(phonon_weight[valid_modes, np.newaxis] * alpha_v)**2
            # raman_map += int_1st.T @ lor_1st  →  (Nfreq, Nfreq_ph)
            raman_map += int_1st.T @ lor_1st

        # --- Second-order ---
        # intensity_2nd: (Npairs_v, Nfreq)
        if has_second_order:
            alpha_pairs = (alpha_tensor_second_order[ialpha, ibeta]
                           .reshape(Nmodes**2, Nfreq)[valid_pairs])    # (Npairs_v, Nfreq)
            int_2nd = np.abs(w_pairs_v[:, np.newaxis] * alpha_pairs)**2  # (Npairs_v, Nfreq)
            raman_map += int_2nd.T @ lor_2nd

        raman_maps[ialpha, ibeta] = raman_map

        plot_data  = np.log(np.maximum(raman_map, 1e-4)) if plot_map_log_scale else raman_map
        cbar_label = 'log(Raman Intensity) (a.u.)' if plot_map_log_scale else 'Raman Intensity (a.u.)'
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.set_title(f'{pol} — flavor {flavor}: {flavor_label}', fontsize=12)
        pcm = ax.pcolormesh(freq_grid, excitation_grid, plot_data, shading='auto')
        fig.colorbar(pcm, ax=ax, label=cbar_label)
        ax.set_xlabel(r'$\omega_{\rm{ph}}$ (cm$^{-1}$)')
        ax.set_ylabel(r'$\Omega_{\rm{exc}}$ (eV)')
        plt.savefig(f'raman_map_{pol}_flavor_{flavor}.png', dpi=300)
        plt.close()

# ---------------------------------------------------------------------------
# Unpolarized map — vectorised
# ---------------------------------------------------------------------------
print('  unpolarized')

if has_first_order:
    # alpha_w: (3, 3, Nvalid, Nfreq) with zero-padding
    alpha_w = np.zeros((3, 3, valid_modes.sum(), Nfreq), dtype=complex)
    alpha_w[:, :, :, :Nfreq_1st] = (phonon_weight[np.newaxis, np.newaxis, valid_modes, np.newaxis]
                                     * alpha_tensor_first_order[:, :, valid_modes, :])
    int_1st_u = unpolarized_invariant(alpha_w)           # (Nvalid, Nfreq)
    raman_map_unpol += int_1st_u.T @ lor_1st                   # (Nfreq, Nfreq_ph)

if has_second_order:
    # alpha_w: (3, 3, Npairs_v, Nfreq)
    alpha_w = (w_pairs_v[np.newaxis, np.newaxis, :, np.newaxis]
               * alpha_tensor_second_order.reshape(3, 3, Nmodes**2, Nfreq)[:, :, valid_pairs, :])
    int_2nd_u = unpolarized_invariant(alpha_w)           # (Npairs_v, Nfreq)
    raman_map_unpol += int_2nd_u.T @ lor_2nd                   # (Nfreq, Nfreq_ph)

plot_data  = np.log(np.maximum(raman_map_unpol, 1e-4)) if plot_map_log_scale else raman_map_unpol
cbar_label = 'log(Raman Intensity) (a.u.)' if plot_map_log_scale else 'Raman Intensity (a.u.)'
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.set_title(f'Unpolarized — flavor {flavor}: {flavor_label}', fontsize=12)
pcm = ax.pcolormesh(freq_grid, excitation_grid, plot_data, shading='auto')
fig.colorbar(pcm, ax=ax, label=cbar_label)
ax.set_xlabel(r'$\omega_{\rm{ph}}$ (cm$^{-1}$)')
ax.set_ylabel(r'$\Omega_{\rm{exc}}$ (eV)')
plt.savefig(f'raman_map_unpolarized_flavor_{flavor}.png', dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------
print(f'Saving Raman maps to {output_file}')
with h5py.File(output_file, 'w') as hf:
    hf.attrs['flavor']        = flavor
    hf.attrs['flavor_label']  = flavor_label
    hf.attrs['temperature_K'] = T
    hf.create_dataset('excitation_energies',   data=excitation_energies)   # (Nfreq,) eV
    hf.create_dataset('freq_axis_cm',          data=freq_axis)             # (Nfreq_ph,) cm^-1
    hf.create_dataset('phonon_frequencies_cm', data=freqs_rec_cm)          # (Nmodes,) cm^-1
    # raman_maps[ialpha, ibeta, iE_exc, iE_ph] — float32 + gzip to keep file small
    _kw = dict(dtype=np.float32, compression='gzip', compression_opts=4)
    hf.create_dataset('raman_maps',            data=raman_maps,    **_kw)  # (3,3,Nfreq,Nfreq_ph)
    hf.create_dataset('raman_map_unpolarized', data=raman_map_unpol, **_kw) # (Nfreq,Nfreq_ph)

print('Done.')
