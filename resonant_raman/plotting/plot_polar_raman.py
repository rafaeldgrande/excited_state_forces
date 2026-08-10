"""
Plot angle-resolved polarized Raman intensities from resonant_raman.py's
--polarized output (resonant_raman_polar_flavor{N}.h5).

Three plot types:
  --plot polar         Per-mode polar plots (I_parallel, I_perp vs theta) at
                        a fixed excitation energy. Needs only the default
                        (amplitude-only) output -- each phonon mode / mode
                        pair is an exact delta-function line, no broadening
                        needed to reconstruct its theta-dependence.
  --plot theta-omega    theta vs Raman-shift map at fixed excitation energy.
  --plot theta-Omega    theta vs excitation-energy map at fixed Raman shift.
The two map types need `raman_map_parallel`/`raman_map_perpendicular`,
written only when resonant_raman.py was run with --polar-store-maps.

Three more plot types read the helicity-resolved datasets, written
unconditionally whenever resonant_raman.py was run with --helicity (see
HELICITY_RAMAN_SPEC.md and resonant_raman/README.md's "Helicity-resolved
Raman" section for the sigma_+/sigma_- convention and its caveats):
  --plot helicity-spectra   I_{sigma+sigma+} (solid) and I_{sigma+sigma-}
                            (dashed) vs Raman shift, stacked by excitation
                            energy (--Eexc-list, falls back to --Eexc).
  --plot helicity-profile   I_{sigma+sigma+}/I_{sigma+sigma-} vs excitation
                            energy at each Raman-active mode's shift, one
                            panel per mode.
  --plot rho-circ           Degree of circular polarization rho_circ(Omega,
                            omega), diverging colormap centred at zero.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import k_B, rec_cm_to_eV, hbar, ignore_0_freq_modes

config_dir = Path(__file__).parent.parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

parser = argparse.ArgumentParser(
    description='Plot angle-resolved polarized Raman intensities')
parser.add_argument('--polar-file',   type=str, default='resonant_raman_polar_flavor0.h5',
                    help='Output of resonant_raman.py --polarized (default: resonant_raman_polar_flavor0.h5)')
parser.add_argument('--plot',         type=str, default='all',
                    choices=['polar', 'theta-omega', 'theta-Omega',
                             'helicity-spectra', 'helicity-profile', 'rho-circ', 'all'],
                    help='Which plot(s) to produce (default: all -- skips any '
                         'that need data not present in --polar-file)')
parser.add_argument('--Eexc',         type=float, default=None,
                    help='Excitation energy in eV (required for --plot polar / theta-omega)')
parser.add_argument('--Eexc-list',    type=float, default=None, nargs='+',
                    help='One or more excitation energies in eV to stack in '
                         '--plot helicity-spectra (default: falls back to --Eexc)')
parser.add_argument('--raman-shift',  type=float, default=None,
                    help='Raman shift in cm^-1 (required for --plot theta-Omega)')
parser.add_argument('--mode',         type=int, default=None,
                    help='First-order phonon mode index to plot (default: all valid modes)')
parser.add_argument('--mode-pair',    type=int, nargs=2, default=None,
                    help='Second-order (imode, jmode) pair to plot (default: none)')
parser.add_argument('--output-prefix', type=str, default='raman_polar',
                    help='Prefix for output PNG filenames (default: raman_polar)')
args = parser.parse_args()


def _close(theta, y):
    """Append the first point to close a polar curve (theta stored half-open [0,2pi))."""
    return np.append(theta, theta[0]), np.append(y, y[0])


print(f'Reading {args.polar_file}')
with h5py.File(args.polar_file, 'r') as hf:
    theta   = hf['theta'][:]
    n_hat   = hf['n_hat'][:]
    e1      = hf['e1'][:]
    e2      = hf['e2'][:]
    exc_en  = hf['excitation_energies'][:]
    freqs_rec_cm = hf['phonon_frequencies_cm'][:]
    freq_axis = hf['freq_axis_cm'][:]
    T       = float(hf.attrs['temperature'])
    flavor  = int(hf.attrs['flavor'])
    flavor_label = str(hf.attrs.get('flavor_label', f'flavor {flavor}'))

    has_1st = 'M_parallel_first_order' in hf
    if has_1st:
        exc_en_1st = hf['excitation_energies_1st'][:]
        M_par_1st  = hf['M_parallel_first_order'][:]
        M_perp_1st = hf['M_perp_first_order'][:]

    has_2nd = 'M_parallel_second_order' in hf
    if has_2nd:
        M_par_2nd  = hf['M_parallel_second_order'][:]
        M_perp_2nd = hf['M_perp_second_order'][:]

    has_maps = 'raman_map_parallel' in hf
    if has_maps:
        map_par  = hf['raman_map_parallel'][:]
        map_perp = hf['raman_map_perpendicular'][:]

    has_helicity = 'raman_map_sigma_pp' in hf
    if has_helicity:
        sigma_pp = hf['raman_map_sigma_pp'][:]
        sigma_pm = hf['raman_map_sigma_pm'][:]
        sigma_mp = hf['raman_map_sigma_mp'][:]
        sigma_mm = hf['raman_map_sigma_mm'][:]
        rho_circ = hf['degree_circular_polarization'][:]
        helicity_convention = str(hf.attrs.get('helicity_convention', 'jones'))

freqs_eV = freqs_rec_cm * rec_cm_to_eV
Nmodes   = len(freqs_rec_cm)
safe_freqs_eV = np.maximum(freqs_eV, 1e-8)
bose_occ      = 1.0 / (np.exp(safe_freqs_eV / (k_B * T)) - 1)
phonon_weight = np.sqrt((bose_occ + 1) * hbar / (2 * safe_freqs_eV))


def _is_valid_mode(im):
    return not (freqs_rec_cm[im] < 1e-2 and ignore_0_freq_modes)


def _frame_annotation():
    return (r'$\hat n$=' + f'[{n_hat[0]:.2f},{n_hat[1]:.2f},{n_hat[2]:.2f}]  '
            + r'$\hat e_1$=' + f'[{e1[0]:.2f},{e1[1]:.2f},{e1[2]:.2f}]')


def _convention_annotation():
    # See resonant_raman/README.md's "Helicity-resolved Raman" section --
    # under 'jones' (default), scattered sigma+ = e_+ (matches QERaman,
    # arXiv:2308.05900); under 'propagation', scattered sigma+ = e_-
    # (true helicity relative to the outgoing -n_hat). Labels swap between
    # the two -- always annotate the convention on helicity plots.
    return f'convention: {helicity_convention}  ' + _frame_annotation()


# ---------------------------------------------------------------------------
# Polar plots: I_parallel(theta), I_perp(theta) at fixed Eexc, per mode
# ---------------------------------------------------------------------------
if args.plot in ('polar', 'all'):
    if args.Eexc is None:
        if args.plot == 'polar':
            sys.exit('ERROR: --Eexc is required for --plot polar')
        print('Skipping polar plot: --Eexc not given')
    else:
        if has_1st:
            iE1 = int(np.argmin(np.abs(exc_en_1st - args.Eexc)))
            modes = [args.mode] if args.mode is not None else \
                    [m for m in range(Nmodes) if _is_valid_mode(m)]
            for m in modes:
                I_par  = np.abs(phonon_weight[m] * M_par_1st[:, m, iE1])**2
                I_perp = np.abs(phonon_weight[m] * M_perp_1st[:, m, iE1])**2
                th_c, Ip_c = _close(theta, I_par)
                _,    Ix_c = _close(theta, I_perp)
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                                       figsize=(6, 6), constrained_layout=True)
                ax.plot(th_c, Ip_c, label=r'$I_\parallel$')
                ax.plot(th_c, Ix_c, label=r'$I_\perp$')
                ax.set_title(f'1st order mode {m} ({freqs_rec_cm[m]:.1f} '
                             + r'cm$^{-1}$)'
                             + f' — {flavor_label}\n'
                             + r'$\Omega_{\rm exc}$=' + f'{exc_en_1st[iE1]:.3f} eV  '
                             + _frame_annotation(), fontsize=9)
                ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
                fname = f'{args.output_prefix}_1st_mode{m}_flavor{flavor}.png'
                fig.savefig(fname, dpi=200)
                plt.close(fig)
                print(f'Saved {fname}')

        if has_2nd and args.mode_pair is not None:
            i, j = args.mode_pair
            iE2 = int(np.argmin(np.abs(exc_en - args.Eexc)))
            w_ij = phonon_weight[i] * phonon_weight[j]
            I_par  = np.abs(w_ij * M_par_2nd[:, i, j, iE2])**2
            I_perp = np.abs(w_ij * M_perp_2nd[:, i, j, iE2])**2
            th_c, Ip_c = _close(theta, I_par)
            _,    Ix_c = _close(theta, I_perp)
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                                   figsize=(6, 6), constrained_layout=True)
            ax.plot(th_c, Ip_c, label=r'$I_\parallel$')
            ax.plot(th_c, Ix_c, label=r'$I_\perp$')
            shift = freqs_rec_cm[i] + freqs_rec_cm[j]
            ax.set_title(f'2nd order modes ({i},{j}), shift={shift:.1f} '
                         + r'cm$^{-1}$'
                         + f' — {flavor_label}\n'
                         + r'$\Omega_{\rm exc}$=' + f'{exc_en[iE2]:.3f} eV  '
                         + _frame_annotation(), fontsize=9)
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
            fname = f'{args.output_prefix}_2nd_mode{i}_{j}_flavor{flavor}.png'
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# theta-omega map at fixed Eexc  (needs --polar-store-maps output)
# ---------------------------------------------------------------------------
if args.plot in ('theta-omega', 'all'):
    if not has_maps:
        if args.plot == 'theta-omega':
            sys.exit('ERROR: raman_map_parallel/perpendicular not in --polar-file; '
                     'rerun resonant_raman.py --polarized --polar-store-maps')
        print('Skipping theta-omega map: file has no stored maps (rerun with --polar-store-maps)')
    elif args.Eexc is None:
        if args.plot == 'theta-omega':
            sys.exit('ERROR: --Eexc is required for --plot theta-omega')
        print('Skipping theta-omega map: --Eexc not given')
    else:
        iE = int(np.argmin(np.abs(exc_en - args.Eexc)))
        for label, data in (('parallel', map_par), ('perpendicular', map_perp)):
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            pcm = ax.pcolormesh(freq_axis, theta, data[:, iE, :], shading='auto')
            fig.colorbar(pcm, ax=ax, label='Raman Intensity (a.u.)')
            ax.set_xlabel(r'$\omega_{\rm{ph}}$ (cm$^{-1}$)')
            ax.set_ylabel(r'$\theta$ (rad)')
            ax.set_title(f'{label} — {flavor_label} — '
                         + r'$\Omega_{\rm exc}$=' + f'{exc_en[iE]:.3f} eV\n'
                         + _frame_annotation(), fontsize=9)
            fname = f'{args.output_prefix}_theta-omega_{label}_flavor{flavor}.png'
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# theta-Omega map at fixed Raman shift  (needs --polar-store-maps output)
# ---------------------------------------------------------------------------
if args.plot in ('theta-Omega', 'all'):
    if not has_maps:
        if args.plot == 'theta-Omega':
            sys.exit('ERROR: raman_map_parallel/perpendicular not in --polar-file; '
                     'rerun resonant_raman.py --polarized --polar-store-maps')
        print('Skipping theta-Omega map: file has no stored maps (rerun with --polar-store-maps)')
    elif args.raman_shift is None:
        if args.plot == 'theta-Omega':
            sys.exit('ERROR: --raman-shift is required for --plot theta-Omega')
        print('Skipping theta-Omega map: --raman-shift not given')
    else:
        iPh = int(np.argmin(np.abs(freq_axis - args.raman_shift)))
        for label, data in (('parallel', map_par), ('perpendicular', map_perp)):
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            pcm = ax.pcolormesh(exc_en, theta, data[:, :, iPh], shading='auto')
            fig.colorbar(pcm, ax=ax, label='Raman Intensity (a.u.)')
            ax.set_xlabel(r'$\Omega_{\rm{exc}}$ (eV)')
            ax.set_ylabel(r'$\theta$ (rad)')
            ax.set_title(f'{label} — {flavor_label} — '
                         + r'$\omega$=' + f'{freq_axis[iPh]:.1f}' + r' cm$^{-1}$'
                         + '\n' + _frame_annotation(), fontsize=9)
            fname = f'{args.output_prefix}_theta-Omega_{label}_flavor{flavor}.png'
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# Helicity-resolved spectra: I_{sigma+sigma+} (solid) / I_{sigma+sigma-}
# (dashed) vs Raman shift, stacked by excitation energy (HELICITY_RAMAN_SPEC
# sec. 6, reproducing QERaman Figs. 3/5). Needs --helicity output.
# ---------------------------------------------------------------------------
if args.plot in ('helicity-spectra', 'all'):
    if not has_helicity:
        if args.plot == 'helicity-spectra':
            sys.exit('ERROR: raman_map_sigma_pp not in --polar-file; '
                     'rerun resonant_raman.py --polarized --helicity')
        print('Skipping helicity-spectra plot: file has no helicity data (rerun with --helicity)')
    else:
        Eexc_list = args.Eexc_list if args.Eexc_list is not None else \
                    ([args.Eexc] if args.Eexc is not None else None)
        if Eexc_list is None:
            if args.plot == 'helicity-spectra':
                sys.exit('ERROR: --Eexc or --Eexc-list is required for --plot helicity-spectra')
            print('Skipping helicity-spectra plot: --Eexc/--Eexc-list not given')
        else:
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            # Stack traces with a vertical offset proportional to the tallest
            # peak seen across the requested excitation energies, so panels
            # stay legible regardless of the absolute intensity scale.
            iE_list = [int(np.argmin(np.abs(exc_en - E))) for E in Eexc_list]
            peak = max(np.max(sigma_pp[iE, :]) for iE in iE_list)
            offset_step = 1.2 * peak if peak > 0 else 1.0
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for k, (E, iE) in enumerate(zip(Eexc_list, iE_list)):
                off = k * offset_step
                c = colors[k % len(colors)]
                ax.plot(freq_axis, sigma_pp[iE, :] + off, '-', color=c,
                        label=r'$\Omega_{\rm exc}=$' + f'{exc_en[iE]:.3f} eV')
                ax.plot(freq_axis, sigma_pm[iE, :] + off, '--', color=c)
            ax.plot([], [], 'k-',  label=r'$I_{\sigma_+\sigma_+}$')
            ax.plot([], [], 'k--', label=r'$I_{\sigma_+\sigma_-}$')
            ax.set_xlabel(r'$\omega_{\rm{ph}}$ (cm$^{-1}$)')
            ax.set_ylabel('Raman Intensity (a.u., stacked)')
            ax.set_title(f'Helicity-resolved spectra — {flavor_label}\n' + _convention_annotation(),
                         fontsize=9)
            ax.legend(fontsize=8)
            fname = f'{args.output_prefix}_helicity-spectra_flavor{flavor}.png'
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# Helicity excitation profiles: I_{sigma+sigma+/-} vs Omega at each
# Raman-active mode's shift, one panel per mode.
# ---------------------------------------------------------------------------
if args.plot in ('helicity-profile', 'all'):
    if not has_helicity:
        if args.plot == 'helicity-profile':
            sys.exit('ERROR: raman_map_sigma_pp not in --polar-file; '
                     'rerun resonant_raman.py --polarized --helicity')
        print('Skipping helicity-profile plot: file has no helicity data (rerun with --helicity)')
    else:
        modes = [args.mode] if args.mode is not None else \
                [m for m in range(Nmodes) if _is_valid_mode(m)]
        for m in modes:
            iPh = int(np.argmin(np.abs(freq_axis - freqs_rec_cm[m])))
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            ax.plot(exc_en, sigma_pp[:, iPh], '-',  label=r'$I_{\sigma_+\sigma_+}$')
            ax.plot(exc_en, sigma_pm[:, iPh], '--', label=r'$I_{\sigma_+\sigma_-}$')
            ax.set_xlabel(r'$\Omega_{\rm{exc}}$ (eV)')
            ax.set_ylabel('Raman Intensity (a.u.)')
            ax.set_title(f'Mode {m} ({freqs_rec_cm[m]:.1f} ' + r'cm$^{-1}$)'
                         + f' — {flavor_label}\n' + _convention_annotation(), fontsize=9)
            ax.legend(fontsize=9)
            fname = f'{args.output_prefix}_helicity-profile_mode{m}_flavor{flavor}.png'
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f'Saved {fname}')

# ---------------------------------------------------------------------------
# Degree of circular polarization map, diverging colormap centred at zero.
# ---------------------------------------------------------------------------
if args.plot in ('rho-circ', 'all'):
    if not has_helicity:
        if args.plot == 'rho-circ':
            sys.exit('ERROR: degree_circular_polarization not in --polar-file; '
                     'rerun resonant_raman.py --polarized --helicity')
        print('Skipping rho-circ plot: file has no helicity data (rerun with --helicity)')
    else:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        pcm = ax.pcolormesh(freq_axis, exc_en, rho_circ, shading='auto',
                            cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        fig.colorbar(pcm, ax=ax, label=r'$\rho_{\rm circ}$')
        ax.set_xlabel(r'$\omega_{\rm{ph}}$ (cm$^{-1}$)')
        ax.set_ylabel(r'$\Omega_{\rm{exc}}$ (eV)')
        ax.set_title(f'Degree of circular polarization — {flavor_label}\n' + _convention_annotation(),
                     fontsize=9)
        fname = f'{args.output_prefix}_rho-circ_flavor{flavor}.png'
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print(f'Saved {fname}')

print('Done.')
