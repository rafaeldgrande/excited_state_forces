
"""
Plot Raman spectra (Raman shift vs intensity) at fixed excitation energies.

Reads the raw susceptibility tensors to allow arbitrary phonon broadening,
independent of the broadening used when building the 2D maps in resonant_raman.py.

Multiple --Eexc values are stacked in a single figure with a vertical offset.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import (k_B, rec_cm_to_eV, hbar, FLAVOR_DESC,
                    ignore_0_freq_modes, unpolarized_invariant)

config_dir = Path(__file__).parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

parser = argparse.ArgumentParser(
    description='Plot Raman spectra vs Raman shift at fixed excitation energies')
parser.add_argument('--raman-file',        type=str, default='resonant_raman_data.h5',
                    help='HDF5 file from resonant_raman.py (for metadata, default: resonant_raman_data.h5)')
parser.add_argument('--first-order-file',  type=str,
                    default='susceptibility_tensors_first_order.h5',
                    help='HDF5 file from susceptibility_tensors_first_order.py (default: susceptibility_tensors_first_order.h5)')
parser.add_argument('--second-order-file', type=str,
                    default='susceptibility_tensors_second_order.h5',
                    help='Required for flavors 2–5 (default: susceptibility_tensors_second_order.h5)')
parser.add_argument('--ipa-first-order-file', type=str,
                    default='susceptibility_tensors_first_order_IPA.h5',
                    help='Required for flavors 6 and 8 (default: susceptibility_tensors_first_order_IPA.h5)')
parser.add_argument('--ipa-second-order-file', type=str,
                    default='susceptibility_tensors_second_order_IPA.h5',
                    help='Required for flavors 7 and 8 (default: susceptibility_tensors_second_order_IPA.h5)')
parser.add_argument('--Eexc',              type=float, nargs='+', required=True,
                    help='One or more excitation energies in eV')
parser.add_argument('--broadening',        type=float, default=10.0,
                    help='Phonon Lorentzian broadening in meV (default: 10 meV)')
parser.add_argument('--npoints',           type=int, default=2000,
                    help='Points on the Raman shift axis (default: 2000)')
parser.add_argument('--normalize',         action='store_true',
                    help='Normalize each spectrum to its maximum before stacking')
parser.add_argument('--log-scale',         action='store_true',
                    help='Use logarithmic y-axis scale')
args = parser.parse_args()

Eexc_targets = sorted(args.Eexc)
gamma_cm     = (args.broadening * 1e-3) / rec_cm_to_eV   # meV → cm^-1
Npoints      = args.npoints
normalize    = args.normalize
log_scale    = args.log_scale

# ---------------------------------------------------------------------------
# Load metadata from resonant_raman_data.h5
# ---------------------------------------------------------------------------
print(f'Reading metadata from {args.raman_file}')
with h5py.File(args.raman_file, 'r') as hf:
    flavor        = int(hf.attrs['flavor'])
    flavor_label  = str(hf.attrs['flavor_label'])
    T             = float(hf.attrs['temperature_K'])
    phonon_frequencies_cm = hf['phonon_frequencies_cm'][:]  # (Nmodes,)

freqs_rec_cm = phonon_frequencies_cm
freqs_eV     = freqs_rec_cm * rec_cm_to_eV
Nmodes       = len(freqs_rec_cm)

# ---------------------------------------------------------------------------
# Load raw susceptibility tensors
# ---------------------------------------------------------------------------
# Derived flags
is_ipa           = flavor in {6, 7, 8}
has_first_order  = flavor in {0, 1, 4, 5, 6, 8}
use_d2           = flavor == 0
has_second_order = flavor in {2, 3, 4, 5, 7, 8}
has_double       = flavor in {3, 5}

alpha_1st               = None
excitation_energies_1st = None
if has_first_order and not is_ipa:
    print(f'Reading first-order susceptibilities from {args.first_order_file}')
    with h5py.File(args.first_order_file, 'r') as f:
        excitation_energies_1st = f['excitation_energies'][:]
        alpha_d2                = f['alpha_tensor_d2'][:]
        alpha_d3                = f['alpha_tensor_d3'][:]
    alpha_1st = alpha_d2 if use_d2 else alpha_d3            # (3,3,Nmodes,Nfreq_1st)

alpha_2nd               = None
excitation_energies_2nd = None
if has_second_order and not is_ipa:
    print(f'Reading second-order susceptibilities from {args.second_order_file}')
    with h5py.File(args.second_order_file, 'r') as f:
        excitation_energies_2nd = f['excitation_energies'][:]
        alpha_2nd               = f['alpha_tensor_triple_resonance'][:]
        if has_double:
            alpha_double = f['alpha_tensor_double_resonance'][:]
    if has_double:
        for imode in range(Nmodes):
            alpha_2nd[:, :, imode, imode, :] += alpha_double[:, :, imode, :]

if flavor in {6, 8}:
    print(f'Reading IPA first-order susceptibilities from {args.ipa_first_order_file}')
    with h5py.File(args.ipa_first_order_file, 'r') as f:
        excitation_energies_1st = f['excitation_energies'][:]
        alpha_1st               = f['susceptibility_tensor_first_order'][:]   # (3,3,Nmodes,Nfreq)

if flavor in {7, 8}:
    print(f'Reading IPA second-order susceptibilities from {args.ipa_second_order_file}')
    with h5py.File(args.ipa_second_order_file, 'r') as f:
        excitation_energies_2nd = f['excitation_energies'][:]
        alpha_2nd               = f['susceptibility_tensor_second_order'][:]  # (3,3,Nmodes,Nmodes,Nfreq)

# ---------------------------------------------------------------------------
# Phonon weights  w_i = sqrt((n_i + 1) * hbar / (2*omega_i))
# ---------------------------------------------------------------------------
safe_freqs_eV = np.maximum(freqs_eV, 1e-8)
bose_occ      = 1.0 / (np.exp(safe_freqs_eV / (k_B * T)) - 1)
phonon_weight = np.sqrt((bose_occ + 1) * hbar / (2 * safe_freqs_eV))   # (Nmodes,)

def is_valid_mode(im):
    return not (freqs_rec_cm[im] < 1e-2 and ignore_0_freq_modes)

# ---------------------------------------------------------------------------
# Raman shift axis
# ---------------------------------------------------------------------------
valid_mask   = np.array([is_valid_mode(im) for im in range(Nmodes)])
min_raman    = max(0.0, np.min(freqs_rec_cm[valid_mask]) - 5 * gamma_cm)
max_raman    = (2 * np.max(freqs_rec_cm) if has_second_order else np.max(freqs_rec_cm)) + 5 * gamma_cm
raman_axis   = np.linspace(min_raman, max_raman, Npoints)  # cm^-1

# ---------------------------------------------------------------------------
# Compute spectrum at each requested Eexc
# ---------------------------------------------------------------------------
results = []   # list of (Eexc_actual, spectrum_unpol, spectra_pol)

cart_dir = ['x', 'y', 'z']

# Reference energy grid and Eexc_actual come from whichever grid is active
ref_energies = excitation_energies_2nd if has_second_order else excitation_energies_1st

for Eexc in Eexc_targets:
    iE_ref      = int(np.argmin(np.abs(ref_energies - Eexc)))
    Eexc_actual = ref_energies[iE_ref]
    iE_1st = int(np.argmin(np.abs(excitation_energies_1st - Eexc))) if has_first_order else None
    iE_2nd = int(np.argmin(np.abs(excitation_energies_2nd - Eexc))) if has_second_order else None
    print(f'Eexc = {Eexc:.4f} eV → {Eexc_actual:.4f} eV')

    spectrum     = np.zeros(Npoints)
    spectra_pol  = np.zeros((3, 3, Npoints))

    # --- first-order ---
    if has_first_order:
        for imode in range(Nmodes):
            if not is_valid_mode(imode):
                continue
            lorentz  = gamma_cm**2 / ((raman_axis - freqs_rec_cm[imode])**2 + gamma_cm**2)
            alpha_ab = phonon_weight[imode] * alpha_1st[:, :, imode, iE_1st]   # (3,3)
            spectrum += unpolarized_invariant(alpha_ab) * lorentz
            for ialpha in range(3):
                for ibeta in range(3):
                    intensity_pol = np.abs(phonon_weight[imode] * alpha_1st[ialpha, ibeta, imode, iE_1st])**2
                    spectra_pol[ialpha, ibeta] += intensity_pol * lorentz

    # --- second-order ---
    if has_second_order:
        for imode in range(Nmodes):
            for jmode in range(Nmodes):
                raman_shift = freqs_rec_cm[imode] + freqs_rec_cm[jmode]
                if raman_shift < 1e-2 and ignore_0_freq_modes:
                    continue
                w_ij    = phonon_weight[imode] * phonon_weight[jmode]
                lorentz = gamma_cm**2 / ((raman_axis - raman_shift)**2 + gamma_cm**2)
                alpha_ab = w_ij * alpha_2nd[:, :, imode, jmode, iE_2nd]    # (3,3)
                spectrum += unpolarized_invariant(alpha_ab) * lorentz
                for ialpha in range(3):
                    for ibeta in range(3):
                        intensity_pol = np.abs(w_ij * alpha_2nd[ialpha, ibeta, imode, jmode, iE_2nd])**2
                        spectra_pol[ialpha, ibeta] += intensity_pol * lorentz

    results.append((Eexc_actual, spectrum, spectra_pol))

# ---------------------------------------------------------------------------
# Capture norm factors before any normalization (always recorded)
# ---------------------------------------------------------------------------
# factors_unpol[i]          = max of spectrum i (unpolarized)
# factors_pol[i][ia][ib]    = max of spectrum i for polarization (ia, ib)
factors_unpol = [s.max() if s.max() > 0 else 1.0 for _, s, _ in results]
factors_pol   = [[[sp[ia, ib].max() if sp[ia, ib].max() > 0 else 1.0
                   for ib in range(3)]
                  for ia in range(3)]
                 for _, _, sp in results]

# ---------------------------------------------------------------------------
# Normalize if requested
# ---------------------------------------------------------------------------
if normalize:
    normed = []
    for Eexc_actual, spectrum, spectra_pol in results:
        s = spectrum / spectrum.max() if spectrum.max() > 0 else spectrum
        sp = np.zeros_like(spectra_pol)
        for ia in range(3):
            for ib in range(3):
                mx = spectra_pol[ia, ib].max()
                sp[ia, ib] = spectra_pol[ia, ib] / mx if mx > 0 else spectra_pol[ia, ib]
        normed.append((Eexc_actual, s, sp))
    results = normed

# ---------------------------------------------------------------------------
# Helper: produce one stacked figure and save it
# ---------------------------------------------------------------------------
# colors = plt.cm.coolwarm(np.linspace(0, 1, len(results)))
colors = plt.cm.tab20(np.linspace(0, 1, len(results)))  # more discrete colors for many spectra

# Extra x-space (cm^-1) to the left for factor annotations
_x_range     = raman_axis[-1] - raman_axis[0]
_annot_pad   = 0.14 * _x_range          # 14% of range reserved for text
_annot_x     = raman_axis[0] - 0.02 * _x_range  # text right-edge just left of data

def _save_stacked(spectra_list, title, fname, factors=None):
    """
    spectra_list : list of (Eexc_actual, 1-D spectrum array)
    factors      : list of floats — the raw max of each spectrum; shown as
                   left-side annotation if provided.
    """
    offset_step = (1.1 if normalize
                   else (max(s.max() for _, s in spectra_list) * 1.1
                         if spectra_list else 1.0))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.set_title(title, fontsize=10)
    for i, ((Eexc_actual, spectrum), color) in enumerate(zip(spectra_list, colors)):
        offset = i * offset_step
        ax.plot(raman_axis, spectrum + offset, color=color,
                label=f'{Eexc_actual:.2f} eV')
        if factors is not None:
            ax.text(_annot_x, offset, f'{factors[i]:.1e}',
                    ha='right', va='bottom', fontsize=12, color=color)
    ax.set_xlabel(r'Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Raman Intensity (a.u.)')
    ax.legend(title=r'$\Omega_{\rm{exc}}$', bbox_to_anchor=(1.02, 1),
              loc='upper left', fontsize=8, title_fontsize=9)
    ax.set_yticks([])
    if factors is not None:
        ax.set_xlim(left=raman_axis[0] - _annot_pad)
    if log_scale:
        ax.set_yscale('log')
        # prevent log(0): floor at the smallest positive plotted value
        plotted = np.concatenate([s + i * offset_step
                                  for i, (_, s) in enumerate(spectra_list)])
        pos_min = plotted[plotted > 0].min() if np.any(plotted > 0) else 1e-10
        ax.set_ylim(bottom=pos_min * 0.5)
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# Plot: unpolarized
# ---------------------------------------------------------------------------
_save_stacked(
    [(E, s) for E, s, _ in results],
    f'Unpolarized Raman — flavor {flavor}: {flavor_label}\nT = {T} K, broadening = {args.broadening} meV',
    f'raman_spectra_stacked_flavor{flavor}.png',
    factors=factors_unpol,
)

# ---------------------------------------------------------------------------
# Plot: polarization-resolved (xx, xy, …, zz)
# ---------------------------------------------------------------------------
for ialpha in range(3):
    for ibeta in range(3):
        pol = f'{cart_dir[ialpha]}{cart_dir[ibeta]}'
        _save_stacked(
            [(E, sp[ialpha, ibeta]) for E, _, sp in results],
            f'Raman ({pol}) — flavor {flavor}: {flavor_label}\nT = {T} K, broadening = {args.broadening} meV',
            f'raman_spectra_stacked_{pol}_flavor{flavor}.png',
            factors=[factors_pol[i][ialpha][ibeta] for i in range(len(results))],
        )

print('Done.')
