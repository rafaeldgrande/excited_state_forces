
"""
Plot the raw susceptibility tensors α^(αβ) vs excitation energy.

Plot 1 — first-order:  one figure per phonon mode, 3×3 subplots for each (α,β) pair.
Plot 2 — second-order: one figure per (imode, jmode) pair, 3×3 subplots.
                       Title shows the sum of the two phonon frequencies.
"""

from pathlib import Path
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

from common import rec_cm_to_eV, FLAVOR_DESC, ignore_0_freq_modes

config_dir = Path(__file__).parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

parser = argparse.ArgumentParser(description='Plot susceptibility tensors α vs excitation energy')
parser.add_argument('--first-order-file',  type=str,
                    default='susceptibility_tensors_first_order.h5')
parser.add_argument('--second-order-file', type=str,
                    default='susceptibility_tensors_second_order.h5',
                    help='Required for flavors 2–5')
parser.add_argument('--freqs-file',        type=str, default='freqs.dat')
parser.add_argument('--flavor',            type=int, default=0,
                    choices=list(FLAVOR_DESC.keys()))
parser.add_argument('--temperature',       type=float, default=300,
                    help='Temperature in K — used in figure titles only (default: 300)')
args = parser.parse_args()

flavor       = args.flavor
flavor_label = FLAVOR_DESC[flavor]
T            = args.temperature
cart_dir     = ['x', 'y', 'z']

# Derived flags
has_first_order  = flavor in {0, 1, 4, 5}
use_d2           = flavor == 0
has_second_order = flavor in {2, 3, 4, 5}
has_double       = flavor in {3, 5}

freqs_rec_cm = np.loadtxt(args.freqs_file)
Nmodes       = len(freqs_rec_cm)

def is_valid_mode(imode):
    return not (freqs_rec_cm[imode] < 1e-2 and ignore_0_freq_modes)

# ---------------------------------------------------------------------------
# Load tensors
# ---------------------------------------------------------------------------
alpha_tensor_first_order = None
excitation_energies_1st  = None
if has_first_order:
    print(f'Reading first-order susceptibilities from {args.first_order_file}')
    with h5py.File(args.first_order_file, 'r') as f:
        excitation_energies_1st = f['excitation_energies'][:]
        alpha_tensor_d2         = f['alpha_tensor_d2'][:]
        alpha_tensor_d3         = f['alpha_tensor_d3'][:]
    alpha_tensor_first_order = alpha_tensor_d2 if use_d2 else alpha_tensor_d3

excitation_energies_2nd   = None
alpha_tensor_second_order = None
if has_second_order:
    print(f'Reading second-order susceptibilities from {args.second_order_file}')
    with h5py.File(args.second_order_file, 'r') as f:
        excitation_energies_2nd   = f['excitation_energies'][:]
        alpha_tensor_second_order = f['alpha_tensor_triple_resonance'][:]
        if has_double:
            alpha_tensor_double_res = f['alpha_tensor_double_resonance'][:]
    if has_double:
        for imode in range(Nmodes):
            alpha_tensor_second_order[:, :, imode, imode, :] += alpha_tensor_double_res[:, :, imode, :]

# ---------------------------------------------------------------------------
# Plot 1: |α¹[ialpha, ibeta, imode, :]| vs excitation energy, one fig per mode
# ---------------------------------------------------------------------------
if has_first_order:
    print('Plotting first-order susceptibility tensors...')
else:
    print('Skipping first-order plots (not included in this flavor).')
for imode in range(Nmodes) if has_first_order else []:
    if not is_valid_mode(imode):
        print(f'  Skipping mode {imode+1} (acoustic)')
        continue
    print(f'  mode {imode+1}/{Nmodes}')

    fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True,
                            layout='constrained')
    fig.suptitle(f'First-order susceptibility — mode {imode+1}, '
                 f'{freqs_rec_cm[imode]:.2f} cm$^{{-1}}$ — T={T} K\n'
                 f'Flavor {flavor}: {flavor_label}', fontsize=11)

    for ialpha in range(3):
        for ibeta in range(3):
            axs[ialpha, ibeta].set_title(f'{cart_dir[ialpha]}{cart_dir[ibeta]}')
            axs[ialpha, ibeta].plot(excitation_energies_1st,
                                    np.abs(alpha_tensor_first_order[ialpha, ibeta, imode, :]))

    fig.supxlabel('Excitation Energy (eV)')
    fig.supylabel(r'$|\alpha^{(1)}_{ij}|$ (a.u.)')
    plt.savefig(f'alpha1_mode_{imode+1}_flavor_{flavor}.png', dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Plot 2: |α²[ialpha, ibeta, imode, jmode, :]| vs excitation energy, one fig per pair
# ---------------------------------------------------------------------------
if has_second_order:
    print('Plotting second-order susceptibility tensors...')
    for imode in range(Nmodes):
        for jmode in range(Nmodes):
            if not is_valid_mode(imode) or not is_valid_mode(jmode):
                continue
            print(f'  pair ({imode+1},{jmode+1}) / ({Nmodes},{Nmodes})')

            freq_sum_cm = freqs_rec_cm[imode] + freqs_rec_cm[jmode]
            fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True,
                                    layout='constrained')
            fig.suptitle(
                f'Second-order susceptibility — modes ({imode+1},{jmode+1}), '
                f'{freqs_rec_cm[imode]:.2f}+{freqs_rec_cm[jmode]:.2f}='
                f'{freq_sum_cm:.2f} cm$^{{-1}}$\nFlavor {flavor}: {flavor_label}',
                fontsize=11)

            for ialpha in range(3):
                for ibeta in range(3):
                    axs[ialpha, ibeta].set_title(f'{cart_dir[ialpha]}{cart_dir[ibeta]}')
                    axs[ialpha, ibeta].plot(
                        excitation_energies_2nd,
                        np.abs(alpha_tensor_second_order[ialpha, ibeta, imode, jmode, :]))

            fig.supxlabel('Excitation Energy (eV)')
            fig.supylabel(r'$|\alpha^{(2)}_{ij}|$ (a.u.)')
            plt.savefig(f'alpha2_modes_{imode+1}_{jmode+1}_flavor_{flavor}.png', dpi=150)
            plt.close()

print('Done.')
