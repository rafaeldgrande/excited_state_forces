
from pathlib import Path
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

config_dir = Path(__file__).parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

ignore_0_freq_modes = True

# constants
k_B          = 8.617333262145e-5   # Boltzmann constant in eV/K
rec_cm_to_eV = 1.239841984e-4      # cm^-1 to eV
hbar         = 6.582119569e-16     # reduced Planck constant in eV*s

FLAVOR_DESC = {
    0: 'First-order, diagonal e-ph only (d2)',
    1: 'First-order, diagonal + off-diagonal e-ph (d3)',
    2: 'First-order d3 + second-order triple-resonance',
    3: 'First-order d3 + second-order triple-resonance + double-resonance',
}

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
parser.add_argument('--freqs-file',        type=str, default='freqs.dat',
                    help='File with phonon frequencies in cm^-1 (default: freqs.dat)')
parser.add_argument('--flavor',            type=int, default=0,
                    choices=list(FLAVOR_DESC.keys()),
                    help='Which susceptibility to use:\n' +
                         '\n'.join(f'  {k}: {v}' for k, v in FLAVOR_DESC.items()))
parser.add_argument('--output',            type=str, default='resonant_raman_data.h5',
                    help='Output HDF5 file (default: resonant_raman_data.h5)')
parser.add_argument('--nfreq-ph',          type=int, default=500,
                    help='Number of points on the phonon frequency axis (default: 500)')
args = parser.parse_args()

T                  = args.temperature
first_order_file   = args.first_order_file
second_order_file  = args.second_order_file
freqs_file         = args.freqs_file
flavor             = args.flavor
output_file        = args.output
Nfreq_ph           = args.nfreq_ph

print(f'Flavor {flavor}: {FLAVOR_DESC[flavor]}')

cart_dir = ['x', 'y', 'z']

freqs_rec_cm = np.loadtxt(freqs_file)
freqs_eV     = freqs_rec_cm * rec_cm_to_eV

# ---------------------------------------------------------------------------
# Load susceptibility tensors
# ---------------------------------------------------------------------------
print(f'Reading first-order susceptibilities from {first_order_file}')
with h5py.File(first_order_file, 'r') as f:
    excitation_energies_1st = f['excitation_energies'][:]   # (Nfreq_1st,)
    alpha_tensor_d2         = f['alpha_tensor_d2'][:]       # (3, 3, Nmodes, Nfreq_1st)
    alpha_tensor_d3         = f['alpha_tensor_d3'][:]       # (3, 3, Nmodes, Nfreq_1st)

alpha_tensor_first_order = alpha_tensor_d2 if flavor == 0 else alpha_tensor_d3

excitation_energies = excitation_energies_1st  # default; overridden below for flavors 2/3
if flavor >= 2:
    print(f'Reading second-order susceptibilities from {second_order_file}')
    with h5py.File(second_order_file, 'r') as f:
        excitation_energies       = f['excitation_energies'][:]           # (Nfreq_2nd,)
        alpha_tensor_second_order = f['alpha_tensor_triple_resonance'][:] # (3,3,Nmodes,Nmodes,Nfreq_2nd)
        if flavor == 3:
            alpha_tensor_double_res = f['alpha_tensor_double_resonance'][:]  # (3,3,Nmodes,Nfreq_2nd)

Nmodes      = alpha_tensor_first_order.shape[2]
flavor_label = FLAVOR_DESC[flavor]

if flavor == 3:
    for imode in range(Nmodes):
        alpha_tensor_second_order[:, :, imode, imode, :] += alpha_tensor_double_res[:, :, imode, :]

# I(alpha,beta) ∝ |Σ_i  w_i · α¹[i]  +  Σ_ij w_i·w_j · α²[i,j]|²
# where w_i = sqrt((n_i+1) · ħ/(2·ω_i))

Nfreq_1st = excitation_energies_1st.shape[0]
Nfreq     = excitation_energies.shape[0]

safe_freqs_eV = np.maximum(freqs_eV, 1e-8)
bose_occ      = 1.0 / (np.exp(safe_freqs_eV / (k_B * T)) - 1)          # (Nmodes,)
phonon_weight = np.sqrt((bose_occ + 1) * hbar / (2 * safe_freqs_eV))    # (Nmodes,)

def is_valid_mode(imode):
    return not (freqs_rec_cm[imode] < 1e-2 and ignore_0_freq_modes)

def unpolarized_invariant(alpha_ab):
    """
    Compute the unpolarized Raman invariant 45|ᾱ|² + 7γ² + 5δ² for a single
    excitation-energy slice of the susceptibility tensor.

    alpha_ab : (3, 3, Nfreq) complex array  — the weighted tensor for one mode/pair
    Returns  : (Nfreq,) real array
    """
    a = alpha_ab  # shorthand
    alpha_bar = (a[0,0] + a[1,1] + a[2,2]) / 3.0
    gamma2 = (0.5 * (np.abs(a[0,0] - a[1,1])**2 +
                     np.abs(a[1,1] - a[2,2])**2 +
                     np.abs(a[2,2] - a[0,0])**2) +
              3/4 * (np.abs(a[0,1] + a[1,0])**2 +
                     np.abs(a[0,2] + a[2,0])**2 +
                     np.abs(a[1,2] + a[2,1])**2))
    delta2 = 3/4 * (np.abs(a[0,1] - a[1,0])**2 +
                    np.abs(a[0,2] - a[2,0])**2 +
                    np.abs(a[1,2] - a[2,1])**2)
    return 45 * np.abs(alpha_bar)**2 + 7 * gamma2 + 5 * delta2

# ---------------------------------------------------------------------------
# Build phonon frequency axis
# ---------------------------------------------------------------------------
min_vib_freq    = np.min(freqs_rec_cm)
max_vib_freq    = np.max(freqs_rec_cm)
phonon_broad_meV = 10.0                                  # meV
gamma_lor        = (phonon_broad_meV * 1e-3) / rec_cm_to_eV  # cm^-1

freq_axis_hi = (2 * max_vib_freq if flavor >= 2 else max_vib_freq) + 5 * gamma_lor
freq_axis    = np.linspace(max(0.0, min_vib_freq - 5 * gamma_lor), freq_axis_hi, Nfreq_ph)

# meshgrid: x = phonon freq, y = excitation energy  → both (Nfreq, Nfreq_ph)
freq_grid, excitation_grid = np.meshgrid(freq_axis, excitation_energies)

# ---------------------------------------------------------------------------
# Compute Raman intensity maps
# raman_maps[ialpha, ibeta, iE_exc, iE_ph]  — polarization-resolved
# raman_map_unpol[iE_exc, iE_ph]            — unpolarized (45|ᾱ|²+7γ²+5δ²)
# ---------------------------------------------------------------------------
raman_maps     = np.zeros((3, 3, Nfreq, Nfreq_ph))
raman_map_unpol = np.zeros((Nfreq, Nfreq_ph))

print('Computing Raman intensity maps...')
for ialpha in range(3):
    for ibeta in range(3):
        pol = f'{cart_dir[ialpha]}{cart_dir[ibeta]}'
        print(f'  {pol}')

        raman_map = np.zeros((Nfreq, Nfreq_ph))

        # First-order: |w_i · α¹[i, Eexc]|² placed at ω_i
        # α¹ covers only Nfreq_1st points; zero elsewhere on the larger grid
        for imode in range(Nmodes):
            if not is_valid_mode(imode):
                continue
            intensity = np.zeros(Nfreq)
            intensity[:Nfreq_1st] = np.abs(phonon_weight[imode] *
                                           alpha_tensor_first_order[ialpha, ibeta, imode, :])**2
            lorentz = gamma_lor**2 / ((freq_axis - freqs_rec_cm[imode])**2 + gamma_lor**2)
            raman_map += intensity[:, np.newaxis] * lorentz[np.newaxis, :]

        # Second-order: |w_i·w_j · α²[i,j, Eexc]|² placed at ω_i+ω_j
        if flavor >= 2:
            for imode in range(Nmodes):
                for jmode in range(Nmodes):
                    freq_cm_pair = freqs_rec_cm[imode] + freqs_rec_cm[jmode]
                    if freq_cm_pair < 1e-2 and ignore_0_freq_modes:
                        continue
                    w_ij      = phonon_weight[imode] * phonon_weight[jmode]
                    intensity = np.abs(w_ij *
                                       alpha_tensor_second_order[ialpha, ibeta, imode, jmode, :])**2
                    lorentz   = gamma_lor**2 / ((freq_axis - freq_cm_pair)**2 + gamma_lor**2)
                    raman_map += intensity[:, np.newaxis] * lorentz[np.newaxis, :]

        raman_maps[ialpha, ibeta] = raman_map

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.set_title(f'{pol} — flavor {flavor}: {flavor_label}', fontsize=12)
        pcm = ax.pcolormesh(freq_grid, excitation_grid, raman_map, shading='auto')
        fig.colorbar(pcm, ax=ax, label='Raman Intensity (a.u.)')
        ax.set_xlabel(r'$\omega_{\rm{ph}}$ (cm$^{-1}$)')
        ax.set_ylabel(r'$\Omega_{\rm{exc}}$ (eV)')
        plt.savefig(f'raman_map_{pol}_flavor_{flavor}.png', dpi=300)
        plt.close()

# ---------------------------------------------------------------------------
# Unpolarized map: accumulate unpolarized invariant per mode/pair
# ---------------------------------------------------------------------------
print('  unpolarized')

# First-order unpolarized
for imode in range(Nmodes):
    if not is_valid_mode(imode):
        continue
    # weighted tensor: (3, 3, Nfreq_1st) → embed in (3, 3, Nfreq)
    alpha_weighted = np.zeros((3, 3, Nfreq), dtype=complex)
    alpha_weighted[:, :, :Nfreq_1st] = (phonon_weight[imode] *
                                         alpha_tensor_first_order[:, :, imode, :])
    intensity = unpolarized_invariant(alpha_weighted)               # (Nfreq,)
    lorentz   = gamma_lor**2 / ((freq_axis - freqs_rec_cm[imode])**2 + gamma_lor**2)
    raman_map_unpol += intensity[:, np.newaxis] * lorentz[np.newaxis, :]

# Second-order unpolarized
if flavor >= 2:
    for imode in range(Nmodes):
        for jmode in range(Nmodes):
            freq_cm_pair = freqs_rec_cm[imode] + freqs_rec_cm[jmode]
            if freq_cm_pair < 1e-2 and ignore_0_freq_modes:
                continue
            w_ij          = phonon_weight[imode] * phonon_weight[jmode]
            alpha_weighted = w_ij * alpha_tensor_second_order[:, :, imode, jmode, :]  # (3,3,Nfreq)
            intensity      = unpolarized_invariant(alpha_weighted)                     # (Nfreq,)
            lorentz        = gamma_lor**2 / ((freq_axis - freq_cm_pair)**2 + gamma_lor**2)
            raman_map_unpol += intensity[:, np.newaxis] * lorentz[np.newaxis, :]

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.set_title(f'Unpolarized — flavor {flavor}: {flavor_label}', fontsize=12)
pcm = ax.pcolormesh(freq_grid, excitation_grid, raman_map_unpol, shading='auto')
fig.colorbar(pcm, ax=ax, label='Raman Intensity (a.u.)')
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
    hf.create_dataset('excitation_energies',  data=excitation_energies)  # (Nfreq,) eV
    hf.create_dataset('freq_axis_cm',         data=freq_axis)            # (Nfreq_ph,) cm^-1
    hf.create_dataset('phonon_frequencies_cm', data=freqs_rec_cm)        # (Nmodes,) cm^-1
    # raman_maps[ialpha, ibeta, iE_exc, iE_ph]
    hf.create_dataset('raman_maps',            data=raman_maps)            # (3,3,Nfreq,Nfreq_ph)
    hf.create_dataset('raman_map_unpolarized', data=raman_map_unpol)      # (Nfreq,Nfreq_ph)

print('Done.')
