
from pathlib import Path
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

config_dir = Path(__file__).parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

parser = argparse.ArgumentParser(
    description='Plot exciton-phonon coupling: |<A|dH/dR_nu|A>| for each mode and exciton state.'
)
parser.add_argument('--exc-ph-file', type=str, default='exciton_phonon_couplings.h5',
                    help='HDF5 file from assemble_exciton_phonon_coeffs.py (default: exciton_phonon_couplings.h5)')
parser.add_argument('--freqs-file', type=str, default='freqs.dat',
                    help='Phonon frequencies in cm^-1, one per line (default: freqs.dat)')
parser.add_argument('--eigenvalues-file', type=str, default=None,
                    help='Exciton eigenvalues file: first column = excitation energy in eV. '
                         'If given, x-axis shows energy; otherwise exciton state index.')
parser.add_argument('--dataset', type=str, default='rpa_diag_plus_kernel',
                    choices=['rpa_diag', 'rpa_offdiag', 'rpa_diag_plus_kernel'],
                    help='Which coupling dataset to plot (default: rpa_diag_plus_kernel)')
parser.add_argument('--n-excitons', type=int, default=None,
                    help='Number of lowest-index exciton states to show (default: all)')
parser.add_argument('--min-freq', type=float, default=1.0,
                    help='Minimum phonon frequency in cm^-1; excludes near-zero acoustic modes (default: 1.0)')
parser.add_argument('--output', type=str, default='exc_ph_matrix.png',
                    help='Output figure filename (default: exc_ph_matrix.png)')
args = parser.parse_args()

# --- Load data ---
freqs = np.loadtxt(args.freqs_file)           # (Nmodes,) in cm^-1

with h5py.File(args.exc_ph_file, 'r') as f:
    coupling = f[args.dataset][:]              # (Nmodes, Nexc, Nexc)

Nmodes, Nexc, _ = coupling.shape

# Diagonal elements: |<A|dH/dR_nu|A>| for each mode nu and exciton A
n_idx   = np.arange(Nexc)
F_diag  = np.abs(coupling[:, n_idx, n_idx])   # (Nmodes, Nexc)

# --- Filter acoustic modes ---
valid   = freqs > args.min_freq
freqs_v = freqs[valid]
F_v     = F_diag[valid]                        # (Nvalid, Nexc)

# Sort modes by descending frequency so highest freq sits at the top
order       = np.argsort(freqs_v)[::-1]
freqs_plot  = freqs_v[order]                   # (Nvalid,)
F_plot      = F_v[order]                       # (Nvalid, Nexc)

# --- Select exciton states ---
n_exc  = min(args.n_excitons, Nexc) if args.n_excitons else Nexc
F_plot = F_plot[:, :n_exc]                     # (Nvalid, n_exc)

# --- Excitation energy labels ---
if args.eigenvalues_file is not None:
    evals        = np.loadtxt(args.eigenvalues_file)
    exc_energies = evals[:n_exc, 0]            # eV
    x_label      = r'$\Omega_{\rm exc}$ (eV)'
    x_ticklabels = [f'{e:.3f}' for e in exc_energies]
else:
    exc_energies = np.arange(1, n_exc + 1)
    x_label      = 'Exciton state index'
    x_ticklabels = [str(i) for i in exc_energies]

# --- Figure ---
Nvalid     = len(freqs_plot)
fig_height = max(6, Nvalid * 0.28)
fig_width  = max(5, n_exc * 0.55 + 3)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

im = ax.imshow(F_plot, aspect='auto', cmap='gnuplot',
               interpolation='none', origin='upper',
               extent=[-0.5, n_exc - 0.5, Nvalid - 0.5, -0.5])

cbar = fig.colorbar(im, ax=ax, label=r'$\langle A | dH/dR_{\nu} | A \rangle$ (eV/\AA)')

# y-axis: one tick per mode, labelled by frequency
ax.set_yticks(np.arange(Nvalid))
ax.set_yticklabels([f'{f:.1f}' for f in freqs_plot], fontsize=10)
ax.set_ylabel(r'$\omega_{\rm vibration}$ (cm$^{-1}$)')

# x-axis: one tick per exciton state
ax.set_xticks(np.arange(n_exc))
ax.set_xticklabels(x_ticklabels, rotation=45, ha='right', fontsize=11)
ax.set_xlabel(x_label)

plt.savefig(args.output, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved to {args.output}')
