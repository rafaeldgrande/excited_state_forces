"""
Generate a self-contained interactive HTML viewer for resonant Raman maps.

Reads  resonant_raman_data_flavor{0..8}.h5  and embeds all data into a
single HTML file backed by Plotly.js (loaded from CDN).

Left panel   — 2-D Raman map (click anywhere to set both Omega_exc and Raman shift)
Middle panel — Raman spectrum at the selected Omega_exc
Right panel  — Excitation profile (intensity vs Omega_exc) at the selected Raman shift

Controls:
  Flavor dropdown  |  Polarization dropdown
  Omega_exc input  |  Raman shift input  |  Linear / Log toggles for each panel

Usage:
  python interactive_vis_resonant_map.py
  python interactive_vis_resonant_map.py --data-dir /path/to/run --output viewer.html
  python interactive_vis_resonant_map.py --max-eexc-points 100 --max-ph-points 200
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import FLAVOR_DESC, cartesian_from_bvec_basis
CART       = ['x', 'y', 'z']
POL_LABELS = ['unpolarized'] + [a + b for a in CART for b in CART]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Generate interactive HTML Raman viewer')
parser.add_argument('--data-dir', default='.',
                    help='Directory with resonant_raman_data_flavor*.h5  (default: .)')
parser.add_argument('--output', default='raman_interactive.html',
                    help='Output HTML file  (default: raman_interactive.html)')
parser.add_argument('--max-eexc-points', type=int, default=200,
                    help='Max Eexc-axis points after down-sampling  (default: 200)')
parser.add_argument('--max-ph-points', type=int, default=300,
                    help='Max phonon-freq-axis points after down-sampling  (default: 300)')
parser.add_argument('--gwbse-qdir', default='../GWBSE/5.2-absorption_Q_shift_comensurable',
                    help='Base dir with per-Q-index eigenvalues_b{1,2,3}.dat subdirs '
                         '(default: ../GWBSE/5.2-absorption_Q_shift_comensurable)')
parser.add_argument('--qpoints-file', default='../RESONANT_RAMAN_2nd_ORDER_all_q/qpoints_crystal.dat',
                    help='BZ q-points + weights file, 7 rows matching the finite-Q '
                         'susceptibility files (default: ../RESONANT_RAMAN_2nd_ORDER_all_q/qpoints_crystal.dat)')
parser.add_argument('--phonon-susc-dir', default='../RESONANT_RAMAN_2nd_ORDER_all_q',
                    help='Dir with susceptibility_tensors_second_order_q_{iq}.h5 files, '
                         'source of phonon_frequencies_cm per q (default: ../RESONANT_RAMAN_2nd_ORDER_all_q)')
parser.add_argument('--dos-gamma-lor', type=float, default=10.0,
                    help='Lorentzian broadening for phonon DOS/JDOS, cm^-1 (default: 10.0, '
                         'matches resonant_raman.py --gamma-lor default)')
parser.add_argument('--dos-gamma-ev', type=float, default=0.01,
                    help='Lorentzian broadening for exciton DOS/absorption, eV (default: 0.01, '
                         'matches susceptibility_tensors_*.py --gamma default)')
parser.add_argument('--dos-npts', type=int, default=400,
                    help='Number of points on each new DOS/absorption axis (default: 400)')
parser.add_argument('--dos-emin', type=float, default=1.0,
                    help='Lower bound of the exciton DOS/absorption energy axis, eV (default: 1.0)')
parser.add_argument('--dos-emax', type=float, default=4.0,
                    help='Upper bound of the exciton DOS/absorption energy axis, eV (default: 4.0)')
parser.add_argument('--plot-polar-plots', action='store_true',
                    help='Also embed angle-resolved polarized (I_parallel/I_perp) and, if present, '
                         'helicity-resolved (sigma_+/sigma_-) Raman data from '
                         'resonant_raman_polar_flavor{N}.h5, and add polar/helicity panels wired to '
                         'the same map click as the spectrum/excitation-profile panels (default: off; '
                         'supersedes the retired interactive_vis_polar_raman.py, see resonant_raman/README.md)')
parser.add_argument('--polar-max-eexc-points', type=int, default=40,
                    help='Max Eexc-axis points for polar/helicity data after down-sampling (default: 40 -- '
                         'kept small since JSON text-encodes each float and this axis is multiplied by '
                         'both --polar-max-ph-points and, for I_parallel/I_perp, --polar-max-theta-points)')
parser.add_argument('--polar-max-ph-points', type=int, default=40,
                    help='Max omega_ph-axis points for polar/helicity data after down-sampling (default: 40)')
parser.add_argument('--polar-max-theta-points', type=int, default=48,
                    help='Max theta-axis points for I_parallel/I_perp data after down-sampling (default: 48)')
parser.add_argument('--ipa-elph-file', default='../RESONANT_RAMAN_IPA/elph.h5',
                    help='Fine-grid el-ph h5 (as used by susceptibility_tensors_IPA.py) with '
                         'Eqp_cond/Eqp_val/Edft_cond/Edft_val datasets, for the IPA DOS curve '
                         'added to the Exciton DOS panel (default: ../RESONANT_RAMAN_IPA/elph.h5). '
                         'Skipped with a warning if not found or missing those datasets.')
parser.add_argument('--ipa-energy-levels', choices=['gw', 'dft'], default='gw',
                    help='Which band energies to use for the IPA DOS transition sum -- matches '
                         'susceptibility_tensors_IPA.py --flavor_energy_levels (1=gw, 2=dft; '
                         'default here: gw, matching that script\'s own default)')
args = parser.parse_args()

# Q-shift indices matching phonon q-points in elph.h5 (Gamma + 6 finite-Q,
# same order as --qpoints-file's 7 rows / the *_q_{iq}.h5 files' iq index)
QIDX_LIST = [0, 2, 4, 6, 12, 14, 18]

# ── load all available flavor files ──────────────────────────────────────────
all_data = {}
for flavor in range(9):
    path = Path(args.data_dir) / f'resonant_raman_data_flavor{flavor}.h5'
    if not path.exists():
        print(f'  Flavor {flavor}: not found, skipping')
        continue
    print(f'  Loading flavor {flavor} from {path} …', end=' ', flush=True)
    with h5py.File(path, 'r') as hf:
        exc_en      = hf['excitation_energies'][:]       # (Nfreq,)
        freq_ax     = hf['freq_axis_cm'][:]              # (Nfreq_ph,)
        raman_maps  = hf['raman_maps'][:]                # (3,3,Nfreq,Nfreq_ph)
        raman_unpol = hf['raman_map_unpolarized'][:]     # (Nfreq,Nfreq_ph)
        flavor_label = str(hf.attrs.get(
            'flavor_label', FLAVOR_DESC.get(flavor, f'flavor {flavor}')))
        T = float(hf.attrs.get('temperature_K', 300))

    # Down-sample to reduce HTML size
    ne, nph     = len(exc_en), len(freq_ax)
    ne_out      = min(ne,  args.max_eexc_points)
    nph_out     = min(nph, args.max_ph_points)
    ie          = np.round(np.linspace(0, ne  - 1, ne_out )).astype(int)
    iph         = np.round(np.linspace(0, nph - 1, nph_out)).astype(int)

    maps = {'unpolarized': raman_unpol[np.ix_(ie, iph)].tolist()}
    for ia, a in enumerate(CART):
        for ib, b in enumerate(CART):
            maps[f'{a}{b}'] = raman_maps[ia, ib][np.ix_(ie, iph)].tolist()

    all_data[str(flavor)] = {
        'flavor_label':       flavor_label,
        'temperature_K':      T,
        'excitation_energies': exc_en[ie].tolist(),
        'freq_axis_cm':        freq_ax[iph].tolist(),
        'maps':                maps,
    }
    print('done')

if not all_data:
    raise SystemExit('No resonant_raman_data_flavor*.h5 files found in '
                     f'"{args.data_dir}".')

# ── --plot-polar-plots: embed angle-resolved (I_par/I_perp) and, if present,
# helicity-resolved (sigma_+/sigma_-) data from resonant_raman_polar_flavor{N}.h5.
# No separate "guide map" is embedded (unlike the retired standalone
# interactive_vis_polar_raman.py) -- the map panel above already covers the
# same (Eexc, omega_ph) grid at the same default resolution, and the polar/
# helicity panels are wired to that same map click, so duplicating it here
# would be pure waste. ──────────────────────────────────────────────────────
any_helicity_map = False
if args.plot_polar_plots:
    print('\n--plot-polar-plots: loading angle-resolved / helicity-resolved data...')

    def _downsample_idx(n_full, n_target):
        n_target = min(n_full, n_target)
        return np.round(np.linspace(0, n_full - 1, n_target)).astype(int)

    for flavor_key in list(all_data.keys()):
        polar_path = Path(args.data_dir) / f'resonant_raman_polar_flavor{flavor_key}.h5'
        if not polar_path.exists():
            print(f'  Flavor {flavor_key}: no {polar_path.name}, skipping polar/helicity panels')
            continue
        with h5py.File(polar_path, 'r') as hf:
            if 'raman_map_parallel' not in hf:
                print(f'  Flavor {flavor_key}: {polar_path.name} has no stored theta maps '
                     '(rerun resonant_raman.py --polarized --polar-store-maps), skipping')
                continue
            theta    = hf['theta'][:]
            n_hat    = hf['n_hat'][:]
            e1       = hf['e1'][:]
            p_exc    = hf['excitation_energies'][:]
            p_ph     = hf['freq_axis_cm'][:]
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

        it  = _downsample_idx(len(theta), args.polar_max_theta_points)
        ie2 = _downsample_idx(len(p_exc), args.polar_max_eexc_points)
        iph2 = _downsample_idx(len(p_ph), args.polar_max_ph_points)

        polar_entry = {
            'theta':               theta[it].tolist(),
            'excitation_energies': p_exc[ie2].tolist(),
            'freq_axis_cm':        p_ph[iph2].tolist(),
            'n_hat': n_hat.tolist(),
            'e1':    e1.tolist(),
            'I_parallel': map_par[np.ix_(it, ie2, iph2)].transpose(1, 2, 0).tolist(),
            'I_perp':     map_perp[np.ix_(it, ie2, iph2)].transpose(1, 2, 0).tolist(),
        }
        if has_helicity:
            polar_entry['helicity'] = {
                'convention': helicity_convention,
                'sigma_pp': sigma_pp[np.ix_(ie2, iph2)].tolist(),
                'sigma_pm': sigma_pm[np.ix_(ie2, iph2)].tolist(),
                'sigma_mp': sigma_mp[np.ix_(ie2, iph2)].tolist(),
                'sigma_mm': sigma_mm[np.ix_(ie2, iph2)].tolist(),
                'rho_circ': rho_circ[np.ix_(ie2, iph2)].tolist(),
            }
            # Also merge the four sigma_+/sigma_- maps into this flavor's
            # `maps` dict, alongside xx/xy/.../unpolarized -- so they become
            # ordinary selectable "polarizations" in the main dropdown,
            # driving the Map/Spectrum/Excitation-profile panels exactly
            # like any Cartesian component (full curves/maps, not just a
            # single-point readout). The polar file's native excitation
            # axis is coarser than the main map's (computed with a reduced
            # --nfreq-exc), so nearest-neighbor-regrid onto the SAME
            # downsampled (excitation, phonon) axes already stored for this
            # flavor's `maps` entries, rather than adding a second axis
            # pair the JS panels would need to special-case.
            target_exc = np.array(all_data[flavor_key]['excitation_energies'])
            target_ph  = np.array(all_data[flavor_key]['freq_axis_cm'])
            ie_match  = np.abs(p_exc[np.newaxis, :] - target_exc[:, np.newaxis]).argmin(axis=1)
            iph_match = np.abs(p_ph[np.newaxis, :]  - target_ph[:, np.newaxis]).argmin(axis=1)
            for label, arr in (('sigma+sigma+', sigma_pp), ('sigma+sigma-', sigma_pm),
                                ('sigma-sigma+', sigma_mp), ('sigma-sigma-', sigma_mm)):
                all_data[flavor_key]['maps'][label] = arr[np.ix_(ie_match, iph_match)].tolist()
            any_helicity_map = True
        all_data[flavor_key]['polar'] = polar_entry
        print(f'  Flavor {flavor_key}: polar' + (' + helicity' if has_helicity else '') + ' data embedded')

if any_helicity_map:
    # Add once, globally -- the dropdown is shared across flavors; flavors
    # lacking helicity data for a given key simply won't have it in `maps`,
    # handled by the same missing-key placeholder as the polar/helicity panels.
    POL_LABELS += ['sigma+sigma+', 'sigma+sigma-', 'sigma-sigma+', 'sigma-sigma-']

# ── phonon DOS / two-phonon JDOS + exciton DOS / optical absorption ───────────
# All four are precomputed once here (peak-height Lorentzian broadening,
# same L(x) = gamma^2 / ((x-x0)^2 + gamma^2) form used throughout
# resonant_raman.py) and embedded as static curves -- no live recomputation
# in JS. BZ q-average uses the same 7 q-points / weights already established
# for the finite-Q resonant-Raman pipeline (Gamma + 6 matching Q-shift
# indices); optical absorption uses Q=0 only, since photons carry ~0
# momentum and only Gamma-point excitons couple to light.
def _lorentzian_sum(axis, centers, weights, gamma):
    """sum_i weights[i] * gamma^2 / ((axis - centers[i])^2 + gamma^2)"""
    diff = axis[np.newaxis, :] - centers[:, np.newaxis]
    return (weights[:, np.newaxis] * (gamma**2 / (diff**2 + gamma**2))).sum(axis=0)

dos_data = {'has_dos': False}
base_dir = Path(args.data_dir)
try:
    qpts = np.loadtxt(base_dir / args.qpoints_file)
    q_weights = qpts[:, -1]
    if len(q_weights) != len(QIDX_LIST):
        raise ValueError(f'{args.qpoints_file} has {len(q_weights)} rows, '
                         f'expected {len(QIDX_LIST)}')

    # -- phonon frequencies per q, from the already-computed per-q susceptibility files --
    ph_freqs = []
    for iq in range(len(QIDX_LIST)):
        with h5py.File(base_dir / args.phonon_susc_dir /
                       f'susceptibility_tensors_second_order_q_{iq}.h5', 'r') as hf:
            ph_freqs.append(hf['phonon_frequencies_cm'][:])
    ph_freqs = np.array(ph_freqs)                      # (Nq, Nmodes)
    Nq, Nmodes = ph_freqs.shape

    max_freq = ph_freqs.max()
    ph_axis  = np.linspace(0.0, 1.05 * max_freq, args.dos_npts)
    ph_centers  = ph_freqs.reshape(-1)                  # (Nq*Nmodes,)
    ph_weights  = np.repeat(q_weights, Nmodes)
    phonon_dos  = _lorentzian_sum(ph_axis, ph_centers, ph_weights, args.dos_gamma_lor)

    # two-phonon joint DOS: full double sum over mode pairs (i,j) per q
    pair_sums    = (ph_freqs[:, :, np.newaxis] + ph_freqs[:, np.newaxis, :]).reshape(Nq, -1)  # (Nq, Nmodes^2)
    jdos_axis    = np.linspace(0.0, 1.05 * 2 * max_freq, args.dos_npts)
    jdos_centers = pair_sums.reshape(-1)
    jdos_weights = np.repeat(q_weights, Nmodes * Nmodes)
    phonon_jdos  = _lorentzian_sum(jdos_axis, jdos_centers, jdos_weights, args.dos_gamma_lor)

    # -- exciton DOS: same BZ q-average, over exciton energies at each q --
    exc_axis = np.linspace(args.dos_emin, args.dos_emax, args.dos_npts)
    exc_dos  = np.zeros_like(exc_axis)
    for w, qidx in zip(q_weights, QIDX_LIST):
        edat = np.loadtxt(base_dir / args.gwbse_qdir / str(qidx) / 'eigenvalues_b1.dat')
        energies = edat[:, 0]
        exc_dos += w * _lorentzian_sum(exc_axis, energies,
                                       np.ones_like(energies), args.dos_gamma_ev)
    exc_dos /= exc_dos.max()   # peak-normalized to 1, for shape comparison against the IPA DOS below

    # -- optical absorption: Q=0 only. The b1/b2/b3 dipole moments are
    #    projections onto the (generally non-orthogonal) reciprocal lattice
    #    unit vectors -- BerkeleyGW's default polarization basis, per
    #    BSE/vmtxel.f90 -- not Cartesian x/y/z, so Sum_i|d_bi|^2 != |P|^2
    #    unless those axes happen to be orthonormal. Convert to Cartesian
    #    first, then take the genuinely isotropic |dx|^2+|dy|^2+|dz|^2.
    dip_b = []
    for b in (1, 2, 3):
        edat = np.loadtxt(base_dir / args.gwbse_qdir / '0' / f'eigenvalues_b{b}.dat')
        dip_b.append(edat[:, 2] + 1j * edat[:, 3])       # complex dipole moment
    energies_q0 = edat[:, 0]
    with h5py.File(base_dir / args.gwbse_qdir / '0' / 'eigenvectors.h5', 'r') as hf:
        bvec = hf['mf_header/crystal/bvec'][:]
    dip_x, dip_y, dip_z = cartesian_from_bvec_basis(*dip_b, bvec)
    dip2_iso   = np.abs(dip_x)**2 + np.abs(dip_y)**2 + np.abs(dip_z)**2
    absorption = _lorentzian_sum(exc_axis, energies_q0, dip2_iso, args.dos_gamma_ev)

    dos_data = {
        'has_dos':      True,
        'ph_axis':      ph_axis.tolist(),
        'phonon_dos':   phonon_dos.tolist(),
        'jdos_axis':    jdos_axis.tolist(),
        'phonon_jdos':  phonon_jdos.tolist(),
        'exc_axis':     exc_axis.tolist(),
        'exciton_dos':  exc_dos.tolist(),
        'absorption':   absorption.tolist(),
        'has_ipa_dos':  False,
    }
    print('  DOS/JDOS/exciton-DOS/absorption panels: computed')

    # -- IPA (non-interacting electron-hole) DOS, same exc_axis/broadening,
    # added as a second curve on the Exciton DOS panel for direct comparison
    # with the excitonic (BSE) DOS above. Uses the SAME elph.h5 (and the
    # same Eqp_cond/Eqp_val/Edft_cond/Edft_val datasets) that
    # susceptibility_tensors_IPA.py itself reads for flavors 0-2 -- so this
    # reflects the actual energies driving this project's own IPA Raman
    # results, not a separately-chosen convention. Q=0 (Gamma) only, same
    # caveat as the optical-absorption curve above -- NOT BZ-q-averaged like
    # the excitonic DOS/JDOS curves (those average over 7 phonon-Q-shift
    # points; this is inherently single-q since it's built directly from the
    # electron/hole k-grid, not the phonon-Q grid).
    try:
        with h5py.File(base_dir / args.ipa_elph_file, 'r') as hf:
            if 'Eqp_cond' not in hf:
                raise KeyError(f'Eqp_cond not in {args.ipa_elph_file} -- run '
                                'interpolate_elph_bgw.py --eqp to add QP energies first')
            if args.ipa_energy_levels == 'gw':
                E_cond = hf['Eqp_cond'][:]    # (Nk, Nc)
                E_val  = hf['Eqp_val'][:]     # (Nk, Nv)
            else:
                if 'Edft_cond' not in hf:
                    raise KeyError(f'Edft_cond not in {args.ipa_elph_file} '
                                    '(--ipa-energy-levels dft requested)')
                E_cond = hf['Edft_cond'][:]
                E_val  = hf['Edft_val'][:]
        # (Nk, Nc, Nv) transition energies, all k/c/v pairs equally weighted
        # -- matches susceptibility_tensors_IPA.py's plain sum over the fine
        # k-grid (already 1/Nk-normalized elsewhere; a DOS just needs the
        # energies, not that normalization).
        delta_E = (E_cond[:, :, np.newaxis] - E_val[:, np.newaxis, :]).reshape(-1)
        ipa_dos = _lorentzian_sum(exc_axis, delta_E, np.ones_like(delta_E), args.dos_gamma_ev)
        ipa_dos /= ipa_dos.max()   # peak-normalized to 1, matching the excitonic DOS above
        dos_data['has_ipa_dos']       = True
        dos_data['ipa_dos']           = ipa_dos.tolist()
        dos_data['ipa_energy_levels'] = args.ipa_energy_levels
        print(f'  IPA DOS ({args.ipa_energy_levels.upper()}, Q=0): computed '
              f'({len(delta_E)} transitions from {args.ipa_elph_file})')
    except (FileNotFoundError, OSError, KeyError) as e:
        print(f'  IPA DOS: skipped ({e})')
except (FileNotFoundError, OSError, ValueError) as e:
    print(f'  DOS/JDOS/exciton-DOS/absorption panels: skipped ({e})')

# ── serialise data ────────────────────────────────────────────────────────────
data_json = json.dumps(all_data,    separators=(',', ':'))
pol_json  = json.dumps(POL_LABELS,  separators=(',', ':'))
dos_json  = json.dumps(dos_data,    separators=(',', ':'))

# ── HTML template (raw string — backslashes kept for JS unicode escapes) ──────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Resonant Raman Viewer</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js" charset="utf-8"></script>
<style>
  body  { font-family: Arial, sans-serif; margin: 12px; background: #f7f7f7; }
  h2    { margin: 4px 0 10px; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-end;
    background: #ebebeb; padding: 10px 14px; border-radius: 6px;
    margin-bottom: 12px;
  }
  .ctrl-group > label { display: block; font-weight: bold; font-size: 13px; margin-bottom: 3px; }
  select, input[type=number] {
    font-size: 13px; padding: 4px 6px;
    border: 1px solid #bbb; border-radius: 4px; background: #fff;
  }
  .radio-row label { font-weight: normal; margin-right: 10px; cursor: pointer; }
  .plots { display: flex; gap: 8px; }
  .plots-polar { display: flex; gap: 8px; margin-top: 14px; }
  .plots-dos { display: flex; gap: 8px; margin-top: 14px; }
  .plot  { flex: 1; min-width: 0; }
  .overlay-panel {
    background: #ebebeb; padding: 10px 14px; border-radius: 6px;
    margin-bottom: 12px;
  }
  .overlay-panel > .title { font-weight: bold; font-size: 13px; margin-bottom: 6px; }
  .overlay-row {
    display: flex; align-items: center; gap: 10px; font-size: 13px;
    padding: 2px 0;
  }
  .overlay-row label.flavor-label { min-width: 260px; font-weight: normal; cursor: pointer; }
  .overlay-row select, .overlay-row input[type=number] {
    font-size: 12px; padding: 2px 4px;
    border: 1px solid #bbb; border-radius: 4px; background: #fff;
  }
  .overlay-row input[type=number] { width: 55px; }
  .dos-placeholder { color: #999; font-size: 13px; padding: 30px; text-align: center; }
</style>
</head>
<body>

<h2>Resonant Raman Interactive Viewer</h2>

<div class="controls">
  <div class="ctrl-group">
    <label>Flavor</label>
    <select id="flavor-sel"></select>
  </div>
  <div class="ctrl-group">
    <label>Polarization &nbsp;<small style="font-weight:normal">(ctrl/cmd-click for multiple)</small></label>
    <select id="pol-sel" multiple size="4"></select>
  </div>
  <div class="ctrl-group">
    <label>Stacked rows &nbsp;<small style="font-weight:normal">(&gt;1 polarization)</small></label>
    <label style="font-weight:normal; cursor:pointer;">
      <input type="checkbox" id="share-y-checkbox"> Share y-axis across rows
    </label>
  </div>
  <div class="ctrl-group">
    <label>&Omega;<sub>exc</sub> (eV) &nbsp;<small style="font-weight:normal">(or click map)</small></label>
    <input id="eexc-input" type="number" step="0.001" style="width:90px">
  </div>
  <div class="ctrl-group">
    <label>Raman shift (cm&#8315;&#185;) &nbsp;<small style="font-weight:normal">(or click map)</small></label>
    <input id="rshift-input" type="number" step="1" style="width:100px">
  </div>
  <div class="ctrl-group">
    <label>Map scale</label>
    <div class="radio-row">
      <label><input type="radio" name="scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Spectrum scale</label>
    <div class="radio-row">
      <label><input type="radio" name="spec-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="spec-scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Excitation profile scale</label>
    <div class="radio-row">
      <label><input type="radio" name="exc-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="exc-scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>DOS / absorption scale</label>
    <div class="radio-row">
      <label><input type="radio" name="dos-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="dos-scale" value="log"> Log</label>
    </div>
  </div>
  <div class="ctrl-group">
    <label>&omega;<sub>ph</sub> range (cm&#8315;&#185;) &nbsp;<small style="font-weight:normal">(default: 0 to max; blank = auto)</small></label>
    <span style="display:flex; gap:4px;">
      <input id="ph-min-input" type="number" step="1" placeholder="min" style="width:70px">
      <input id="ph-max-input" type="number" step="1" placeholder="max" style="width:70px">
    </span>
  </div>
  <div class="ctrl-group">
    <label>&Omega;<sub>exc</sub> range (eV) &nbsp;<small style="font-weight:normal">(default: 0 to max; blank = auto)</small></label>
    <span style="display:flex; gap:4px;">
      <input id="eexc-min-input" type="number" step="0.01" placeholder="min" style="width:70px">
      <input id="eexc-max-input" type="number" step="0.01" placeholder="max" style="width:70px">
    </span>
  </div>
__POLAR_CONTROLS__
</div>

<div class="overlay-panel">
  <div class="title">Overlay additional flavors (Spectrum &amp; Excitation Profile panels only)</div>
  <div id="overlay-rows"></div>
</div>

<div class="plots">
  <div class="plot" id="map-div"></div>
  <div class="plot" id="spec-div"></div>
  <div class="plot" id="exc-div"></div>
</div>

__POLAR_PLOTS_ROW__

<div class="plots-dos">
  <div class="plot" id="phdos-div"></div>
  <div class="plot" id="phjdos-div"></div>
  <div class="plot" id="excdos-div"></div>
  <div class="plot" id="abs-div"></div>
</div>

<script>
const DATA       = __DATA__;
const POL_LABELS = __POL_LABELS__;
const DOS_DATA   = __DOS_DATA__;

// ── populate dropdowns ────────────────────────────────────────────────────────
const flavorSel = document.getElementById('flavor-sel');
for (const [k, v] of Object.entries(DATA)) {
  const o = document.createElement('option');
  o.value = k;
  o.text  = 'Flavor ' + k + ': ' + v.flavor_label;
  flavorSel.appendChild(o);
}

const polSel = document.getElementById('pol-sel');
for (const p of POL_LABELS) {
  const o = document.createElement('option');
  o.value = p; o.text = p;
  polSel.appendChild(o);
}
polSel.options[0].selected = true;   // 'unpolarized' selected by default

// ── overlay panel: one row per flavor (color/linestyle/linewidth pickers) ─────
const OVERLAY_COLORS = ['red', 'blue', 'green', 'orange', 'purple',
                         'brown', 'magenta', 'cyan', 'gray'];
const LINESTYLES = [['solid', 'Solid'], ['dash', 'Dashed'], ['dot', 'Dotted']];
const overlayRowsDiv = document.getElementById('overlay-rows');
const overlayState = {};   // flavor key -> {checkbox, colorSel, styleSel, widthInput}

Object.keys(DATA).forEach(function(k, i) {
  const row = document.createElement('div');
  row.className = 'overlay-row';

  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.id = 'overlay-cb-' + k;

  const label = document.createElement('label');
  label.className = 'flavor-label';
  label.htmlFor = cb.id;
  label.textContent = 'Flavor ' + k + ': ' + DATA[k].flavor_label;

  const colorSel = document.createElement('select');
  OVERLAY_COLORS.forEach(function(c) {
    const o = document.createElement('option'); o.value = c; o.text = c;
    colorSel.appendChild(o);
  });
  colorSel.value = OVERLAY_COLORS[i % OVERLAY_COLORS.length];

  const styleSel = document.createElement('select');
  LINESTYLES.forEach(function(s) {
    const o = document.createElement('option'); o.value = s[0]; o.text = s[1];
    styleSel.appendChild(o);
  });

  const widthInput = document.createElement('input');
  widthInput.type = 'number'; widthInput.step = '0.5'; widthInput.min = '0.5';
  widthInput.value = '1.5';

  row.appendChild(cb);
  row.appendChild(label);
  row.appendChild(colorSel);
  row.appendChild(styleSel);
  row.appendChild(widthInput);
  overlayRowsDiv.appendChild(row);

  overlayState[k] = { checkbox: cb, colorSel: colorSel, styleSel: styleSel, widthInput: widthInput };

  [cb, colorSel, styleSel, widthInput].forEach(function(el) {
    el.addEventListener('change', function() { updateSpectrum(); updateExcProfile(); });
  });
});

function activeOverlays() {
  // list of {key, color, dash, width} for checked flavors, excluding the
  // primary (map) flavor to avoid a redundant duplicate legend entry
  const out = [];
  for (const [k, st] of Object.entries(overlayState)) {
    if (st.checkbox.checked && k !== flavorSel.value) {
      out.push({
        key:   k,
        color: st.colorSel.value,
        dash:  st.styleSel.value,
        width: parseFloat(st.widthInput.value) || 1.5,
      });
    }
  }
  return out;
}

// ── global log scale bounds (computed once over ALL maps / flavors / pols) ────
var GLOBAL_LOG_FLOOR = Infinity;   // smallest positive value seen
var GLOBAL_LOG_MAX   = -Infinity;  // largest value seen
(function() {
  for (const fdata of Object.values(DATA)) {
    for (const rows of Object.values(fdata.maps)) {
      for (const row of rows) {
        for (const v of row) {
          if (v > 0 && v < GLOBAL_LOG_FLOOR) GLOBAL_LOG_FLOOR = v;
          if (v > GLOBAL_LOG_MAX)            GLOBAL_LOG_MAX   = v;
        }
      }
    }
  }
  // Fallback guards
  if (!isFinite(GLOBAL_LOG_FLOOR)) GLOBAL_LOG_FLOOR = 1e-30;
  if (!isFinite(GLOBAL_LOG_MAX))   GLOBAL_LOG_MAX   = 1.0;
})();

// ── helpers ───────────────────────────────────────────────────────────────────
function curData() { return DATA[flavorSel.value]; }
// pol-sel is a multi-select (item 3): selectedPols() is the checked list
// (falling back to the first POL_LABELS entry if somehow none are selected,
// e.g. a user ctrl/cmd-clicking the only checked option off). curPol() --
// used by the Map panel, which stays single-polarization by design -- is
// always just the first selected entry.
function selectedPols() {
  const vals = Array.from(polSel.selectedOptions).map(function(o) { return o.value; });
  return vals.length ? vals : [POL_LABELS[0]];
}
function curPol()  { return selectedPols()[0]; }
function curPolAvailable() {
  // sigma+sigma+/sigma+sigma-/sigma-sigma+/sigma-sigma- (from --plot-polar-plots)
  // are only present for flavors that have --helicity data -- every other
  // polarization (unpolarized, xx, xy, ...) is always present for every flavor.
  return curData().maps[curPol()] !== undefined;
}
function isLog()   {
  return document.querySelector('input[name="scale"]:checked').value === 'log';
}
function isLogSpec() {
  return document.querySelector('input[name="spec-scale"]:checked').value === 'log';
}
function isLogExc() {
  return document.querySelector('input[name="exc-scale"]:checked').value === 'log';
}
function isLogDos() {
  return document.querySelector('input[name="dos-scale"]:checked').value === 'log';
}

function applyLog(z) {
  return z.map(function(row) {
    return row.map(function(v) { return Math.log10(Math.max(v, GLOBAL_LOG_FLOOR)); });
  });
}

function nearestIdx(arr, val) {
  let best = 0, d0 = Math.abs(arr[0] - val);
  for (let i = 1; i < arr.length; i++) {
    const d = Math.abs(arr[i] - val);
    if (d < d0) { d0 = d; best = i; }
  }
  return best;
}

function eexcVal() {
  return parseFloat(document.getElementById('eexc-input').value);
}
function setEexcInput(val) {
  document.getElementById('eexc-input').value = val.toFixed(4);
}

function rshiftVal() {
  return parseFloat(document.getElementById('rshift-input').value);
}
function setRshiftInput(val) {
  document.getElementById('rshift-input').value = val.toFixed(1);
}

// omega_ph / Eexc axis-range controls (item 4) -- undefined (not [NaN,NaN])
// when blank, so callers can `if (range)` rather than checking each bound.
function phRangeVal() {
  const lo = parseFloat(document.getElementById('ph-min-input').value);
  const hi = parseFloat(document.getElementById('ph-max-input').value);
  if (isNaN(lo) && isNaN(hi)) return undefined;
  const d = curData();
  return [isNaN(lo) ? d.freq_axis_cm[0] : lo,
          isNaN(hi) ? d.freq_axis_cm[d.freq_axis_cm.length - 1] : hi];
}
function eexcRangeVal() {
  const lo = parseFloat(document.getElementById('eexc-min-input').value);
  const hi = parseFloat(document.getElementById('eexc-max-input').value);
  if (isNaN(lo) && isNaN(hi)) return undefined;
  const d = curData();
  return [isNaN(lo) ? d.excitation_energies[0] : lo,
          isNaN(hi) ? d.excitation_energies[d.excitation_energies.length - 1] : hi];
}

// ── figure builders ───────────────────────────────────────────────────────────
function buildMapTrace() {
  const d        = curData();
  const raw      = d.maps[curPol()];
  const logScale = isLog();
  const z        = logScale ? applyLog(raw) : raw;

  const trace = {
    type:          'heatmap',
    x:             d.freq_axis_cm,
    y:             d.excitation_energies,
    z:             z,
    colorscale:    'Viridis',
    colorbar: {
      title: { text: logScale ? 'log\u2081\u2080(I)' : 'I (a.u.)', side: 'right' },
      thickness: 14,
    },
    hovertemplate: '\u03c9: %{x:.1f} cm\u207b\u00b9<br>'
                 + '\u03a9<sub>exc</sub>: %{y:.4f} eV<br>'
                 + 'I: %{z:.3e}<extra></extra>',
  };

  if (logScale) {
    // Pin colorbar to the true global min/max across ALL maps so the scale
    // is consistent when switching flavor or polarization.
    trace.zmin = Math.log10(GLOBAL_LOG_FLOOR);
    trace.zmax = Math.log10(GLOBAL_LOG_MAX);
  }

  return [trace];
}

function buildMapLayout(eexc, rshift) {
  const d  = curData();
  const x0 = d.freq_axis_cm[0];
  const x1 = d.freq_axis_cm[d.freq_axis_cm.length - 1];
  const y0 = d.excitation_energies[0];
  const y1 = d.excitation_energies[d.excitation_energies.length - 1];
  const phRange   = phRangeVal();
  const eexcRange = eexcRangeVal();
  return {
    title: {
      text: 'Raman Map \u2014 ' + curPol()
          + ' \u2014 Flavor ' + flavorSel.value
          + ' \u2014 T = ' + d.temperature_K + ' K',
      font: { size: 13 },
    },
    xaxis: { title: '\u03c9<sub>ph</sub> (cm\u207b\u00b9)', range: phRange },
    yaxis: { title: '\u03a9<sub>exc</sub> (eV)', range: eexcRange },
    shapes: [
      {
        // horizontal line — fixed excitation energy (red)
        type: 'line',
        x0: x0, x1: x1, y0: eexc, y1: eexc,
        line: { color: 'red', width: 1.5, dash: 'dash' },
      },
      {
        // vertical line — fixed Raman shift (orange)
        type: 'line',
        x0: rshift, x1: rshift, y0: y0, y1: y1,
        line: { color: '#ff7f0e', width: 1.5, dash: 'dash' },
      },
    ],
    margin: { l: 60, r: 10, t: 55, b: 55 },
  };
}

function buildSpectrumTraces(eexc) {
  const d  = curData();
  const iE = nearestIdx(d.excitation_energies, eexc);
  const traces = [{
    type: 'scatter',
    name: 'Flavor ' + flavorSel.value + ' (map)',
    x:    d.freq_axis_cm,
    y:    d.maps[curPol()][iE],
    mode: 'lines',
    line: { color: 'black', width: 2.5 },
    hovertemplate: '\u03c9: %{x:.1f} cm\u207b\u00b9<br>I: %{y:.3e}<extra>Flavor ' + flavorSel.value + '</extra>',
  }];
  for (const ov of activeOverlays()) {
    const od  = DATA[ov.key];
    if (od.maps[curPol()] === undefined) continue;   // overlay flavor lacks this polarization
    const oiE = nearestIdx(od.excitation_energies, eexc);
    traces.push({
      type: 'scatter',
      name: 'Flavor ' + ov.key + ': ' + od.flavor_label,
      x:    od.freq_axis_cm,
      y:    od.maps[curPol()][oiE],
      mode: 'lines',
      line: { color: ov.color, width: ov.width, dash: ov.dash },
      hovertemplate: '\u03c9: %{x:.1f} cm\u207b\u00b9<br>I: %{y:.3e}<extra>Flavor ' + ov.key + '</extra>',
    });
  }
  return traces;
}

function buildSpectrumLayout(eexc_actual, rshift) {
  const logSpec = isLogSpec();
  return {
    title: {
      text: 'Raman Spectrum \u2014 ' + curPol()
          + ' \u2014 \u03a9<sub>exc</sub> \u2248 ' + eexc_actual.toFixed(4) + ' eV',
      font: { size: 13 },
    },
    xaxis: { title: 'Raman shift (cm\u207b\u00b9)', range: phRangeVal() },
    yaxis: {
      title: logSpec ? 'log\u2081\u2080(I)' : 'Raman Intensity (a.u.)',
      type:  logSpec ? 'log' : 'linear',
    },
    showlegend: true,
    // outside the plotting area (below), not overlapping the curves
    legend: { font: { size: 9 }, orientation: 'h', x: 0.5, xanchor: 'center', y: -0.22, yanchor: 'top' },
    // vertical marker at the pinned Raman shift (orange, matches map crosshair)
    shapes: [{
      type: 'line',
      x0: rshift, x1: rshift, y0: 0, y1: 1, yref: 'paper',
      line: { color: '#ff7f0e', width: 1.5, dash: 'dash' },
    }],
    margin: { l: 65, r: 10, t: 55, b: 90 },
  };
}

function buildExcProfileTraces(rshift) {
  const d    = curData();
  const iPh  = nearestIdx(d.freq_axis_cm, rshift);
  const traces = [{
    type: 'scatter',
    name: 'Flavor ' + flavorSel.value + ' (map)',
    x:    d.excitation_energies,
    y:    d.maps[curPol()].map(function(row) { return row[iPh]; }),
    mode: 'lines',
    line: { color: 'black', width: 2.5 },
    hovertemplate: '\u03a9<sub>exc</sub>: %{x:.4f} eV<br>I: %{y:.3e}<extra>Flavor ' + flavorSel.value + '</extra>',
  }];
  for (const ov of activeOverlays()) {
    const od   = DATA[ov.key];
    if (od.maps[curPol()] === undefined) continue;   // overlay flavor lacks this polarization
    const oiPh = nearestIdx(od.freq_axis_cm, rshift);
    traces.push({
      type: 'scatter',
      name: 'Flavor ' + ov.key + ': ' + od.flavor_label,
      x:    od.excitation_energies,
      y:    od.maps[curPol()].map(function(row) { return row[oiPh]; }),
      mode: 'lines',
      line: { color: ov.color, width: ov.width, dash: ov.dash },
      hovertemplate: '\u03a9<sub>exc</sub>: %{x:.4f} eV<br>I: %{y:.3e}<extra>Flavor ' + ov.key + '</extra>',
    });
  }
  return traces;
}

function buildExcProfileLayout(ph_actual, eexc) {
  const logExc = isLogExc();
  return {
    title: {
      text: 'Excitation Profile \u2014 ' + curPol()
          + ' \u2014 \u03c9 \u2248 ' + ph_actual.toFixed(1) + ' cm\u207b\u00b9',
      font: { size: 13 },
    },
    xaxis: { title: '\u03a9<sub>exc</sub> (eV)', range: eexcRangeVal() },
    yaxis: {
      title: logExc ? 'log\u2081\u2080(I)' : 'Raman Intensity (a.u.)',
      type:  logExc ? 'log' : 'linear',
    },
    showlegend: true,
    // outside the plotting area (below), not overlapping the curves
    legend: { font: { size: 9 }, orientation: 'h', x: 0.5, xanchor: 'center', y: -0.22, yanchor: 'top' },
    // vertical marker at the current Eexc (red, matches map crosshair)
    shapes: [{
      type: 'line',
      x0: eexc, x1: eexc, y0: 0, y1: 1, yref: 'paper',
      line: { color: 'red', width: 1.5, dash: 'dash' },
    }],
    margin: { l: 65, r: 10, t: 55, b: 90 },
  };
}

// ── stacked multi-polarization figures (item 3) ────────────────────────────────
// When >1 polarization is checked, the Spectrum and Excitation-profile panels
// become an N-row grid of subplots (one row per polarization), sharing both
// x and y axes (row 2+ `matches` row 1's axis) so absolute intensities stay
// directly comparable. Each row shows the same primary-flavor + overlay
// curves as the single-polarization panel, just restricted to that row's
// polarization and skipped (not erroring) wherever a flavor lacks it --
// same missing-key handling as the single-pol path above.
function axisSuffix(i) { return i === 0 ? '' : String(i + 1); }

// kind: 'spectrum' (row x-axis = freq_axis_cm, indexed by nearest excitation
// energy to `targetValue`) or 'excprofile' (row x-axis = excitation_energies,
// indexed by nearest Raman shift to `targetValue`). Each flavor (primary and
// every active overlay) looks up its OWN nearest index against its OWN axis
// -- flavors can have slightly different downsampled grids, so a single
// shared index would silently misalign one flavor's curve against another's.
function buildStackedFigure(pols, kind, targetValue) {
  const N = pols.length;
  const d = curData();
  const logScale = (kind === 'spectrum' ? isLogSpec : isLogExc)();
  const shareY = document.getElementById('share-y-checkbox').checked;   // default: unchecked (independent y per row)
  const traces = [];

  function seriesFor(flavorData, pol) {
    const indexAxis = kind === 'spectrum' ? flavorData.excitation_energies : flavorData.freq_axis_cm;
    const curveAxis = kind === 'spectrum' ? flavorData.freq_axis_cm : flavorData.excitation_energies;
    const i = nearestIdx(indexAxis, targetValue);
    const series = kind === 'spectrum'
      ? flavorData.maps[pol][i]
      : flavorData.maps[pol].map(function(row) { return row[i]; });
    return { x: curveAxis, y: series, actual: indexAxis[i] };
  }

  let actualLabel = '';
  const layout = {
    grid: { rows: N, columns: 1, pattern: 'independent', ygap: 0.12 },
    showlegend: true,
    legend: { font: { size: 9 }, orientation: 'h', x: 0.5, xanchor: 'center', y: -0.09, yanchor: 'top' },
    margin: { l: 65, r: 10, t: 40, b: 80 },
  };

  pols.forEach(function(pol, i) {
    const suf = axisSuffix(i);
    const xkey = 'x' + suf, ykey = 'y' + suf;
    if (d.maps[pol] !== undefined) {
      const s = seriesFor(d, pol);
      if (i === 0) actualLabel = s.actual;
      traces.push({
        type: 'scatter', mode: 'lines', name: pol + ' (Flavor ' + flavorSel.value + ')',
        x: s.x, y: s.y, xaxis: xkey, yaxis: ykey, showlegend: i === 0,
        legendgroup: 'primary',
        line: { color: 'black', width: 2 },
      });
    }
    for (const ov of activeOverlays()) {
      const od = DATA[ov.key];
      if (od.maps[pol] === undefined) continue;
      const s = seriesFor(od, pol);
      traces.push({
        type: 'scatter', mode: 'lines', name: 'Flavor ' + ov.key + ': ' + od.flavor_label,
        x: s.x, y: s.y, xaxis: xkey, yaxis: ykey, showlegend: i === 0,
        legendgroup: 'ov' + ov.key,
        line: { color: ov.color, width: ov.width, dash: ov.dash },
      });
    }
    layout['yaxis' + suf] = {
      title: pol, type: logScale ? 'log' : 'linear',
      matches: (shareY && i !== 0) ? 'y' : undefined,
    };
    layout['xaxis' + suf] = {
      range: kind === 'spectrum' ? phRangeVal() : eexcRangeVal(),
      title: (i === N - 1) ? (kind === 'spectrum' ? 'Raman shift (cm⁻¹)' : 'Ω<sub>exc</sub> (eV)') : '',
      matches: i === 0 ? undefined : 'x',
      showticklabels: (i === N - 1),
    };
  });

  const titlePrefix = kind === 'spectrum' ? 'Raman Spectrum' : 'Excitation Profile';
  const titleSuffix = kind === 'spectrum'
    ? 'Ω<sub>exc</sub> ≈ ' + Number(actualLabel).toFixed(4) + ' eV'
    : 'ω ≈ ' + Number(actualLabel).toFixed(1) + ' cm⁻¹';
  layout.title = { text: titlePrefix + ' (stacked) — Flavor ' + flavorSel.value + ' — ' + titleSuffix,
                    font: { size: 13 } };
  return { traces: traces, layout: layout };
}

function updateSpectrumMulti(pols, eexc) {
  const fig = buildStackedFigure(pols, 'spectrum', eexc);
  return Plotly.react('spec-div', fig.traces, fig.layout);
}

function updateExcProfileMulti(pols, rshift) {
  const fig = buildStackedFigure(pols, 'excprofile', rshift);
  return Plotly.react('exc-div', fig.traces, fig.layout);
}

// ── update functions ──────────────────────────────────────────────────────────
function polUnavailableMsg() {
  return '"' + curPol() + '" not available for Flavor ' + flavorSel.value
       + ' (rerun resonant_raman.py --polarized --helicity for this flavor)';
}

// map-div keeps a plotly_click handler bound at init time -- always route it
// through Plotly.react (an empty figure with an annotation, here) rather than
// overwriting its innerHTML, so that handler stays attached to the div.
function updateMapUnavailable() {
  return Plotly.react('map-div', [], {
    title: { text: polUnavailableMsg(), font: { size: 12 } },
    xaxis: { visible: false }, yaxis: { visible: false },
    margin: { l: 20, r: 10, t: 55, b: 20 },
  });
}

function renderPolUnavailablePlaceholder(divId) {
  document.getElementById(divId).innerHTML =
    '<div class="dos-placeholder">' + polUnavailableMsg() + '</div>';
}

function updateMap() {
  if (!curPolAvailable()) { return updateMapUnavailable(); }
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.freq_axis_cm[0]        : rshiftVal();
  return Plotly.react('map-div', buildMapTrace(), buildMapLayout(eexc, rshift));
}

function updateSpectrum() {
  const d      = curData();
  const eexc   = isNaN(eexcVal())   ? d.excitation_energies[0] : eexcVal();
  const pols   = selectedPols();
  if (pols.length > 1) { return updateSpectrumMulti(pols, eexc); }
  if (!curPolAvailable()) { renderPolUnavailablePlaceholder('spec-div'); return; }
  const rshift = isNaN(rshiftVal()) ? d.freq_axis_cm[0]        : rshiftVal();
  const iE     = nearestIdx(d.excitation_energies, eexc);
  return Plotly.react('spec-div',
                      buildSpectrumTraces(eexc),
                      buildSpectrumLayout(d.excitation_energies[iE], rshift));
}

function updateExcProfile() {
  const d      = curData();
  const rshift = isNaN(rshiftVal()) ? d.freq_axis_cm[0]        : rshiftVal();
  const pols   = selectedPols();
  if (pols.length > 1) { return updateExcProfileMulti(pols, rshift); }
  if (!curPolAvailable()) { renderPolUnavailablePlaceholder('exc-div'); return; }
  const eexc   = isNaN(eexcVal())   ? d.excitation_energies[0] : eexcVal();
  const iPh    = nearestIdx(d.freq_axis_cm, rshift);
  return Plotly.react('exc-div',
                      buildExcProfileTraces(rshift),
                      buildExcProfileLayout(d.freq_axis_cm[iPh], eexc));
}

function updateAll() {
  updateMap(); updateSpectrum(); updateExcProfile();
  if (typeof updatePolarPanels === 'function') updatePolarPanels();
}

// ── static DOS / JDOS / exciton-DOS / absorption panels ───────────────────────
function dosLineLayout(title, xtitle, ytitle) {
  const logDos = isLogDos();
  return {
    title: { text: title, font: { size: 13 } },
    xaxis: { title: xtitle },
    yaxis: {
      title: logDos ? 'log\u2081\u2080(' + ytitle + ')' : ytitle,
      type:  logDos ? 'log' : 'linear',
    },
    margin: { l: 55, r: 10, t: 45, b: 50 },
  };
}

function renderDosPlaceholder(divId, label) {
  document.getElementById(divId).innerHTML =
    '<div class="dos-placeholder">' + label + ' data not available<br>'
    + '(run with --gwbse-qdir / --qpoints-file / --phonon-susc-dir pointing '
    + 'at the resonant-Raman pipeline directories)</div>';
}

function updateDosPanels() {
  if (!DOS_DATA.has_dos) {
    renderDosPlaceholder('phdos-div',  'Phonon DOS');
    renderDosPlaceholder('phjdos-div', 'Phonon joint DOS');
    renderDosPlaceholder('excdos-div', 'Exciton DOS');
    renderDosPlaceholder('abs-div',    'Optical absorption');
    return;
  }
  const phDosLayout = dosLineLayout('Phonon DOS', '\u03c9 (cm\u207b\u00b9)', 'DOS (a.u.)');
  phDosLayout.xaxis.range = phRangeVal();
  Plotly.react('phdos-div', [{
    type: 'scatter', mode: 'lines', x: DOS_DATA.ph_axis, y: DOS_DATA.phonon_dos,
    line: { color: '#1f77b4', width: 1.5 },
  }], phDosLayout);

  const phJdosLayout = dosLineLayout('Two-Phonon Joint DOS', '\u03c9<sub>1</sub>+\u03c9<sub>2</sub> (cm\u207b\u00b9)', 'JDOS (a.u.)');
  phJdosLayout.xaxis.range = phRangeVal();
  Plotly.react('phjdos-div', [{
    type: 'scatter', mode: 'lines', x: DOS_DATA.jdos_axis, y: DOS_DATA.phonon_jdos,
    line: { color: '#9467bd', width: 1.5 },
  }], phJdosLayout);

  const excDosTraces = [{
    type: 'scatter', mode: 'lines', name: 'Excitonic (BZ-averaged)',
    x: DOS_DATA.exc_axis, y: DOS_DATA.exciton_dos,
    line: { color: '#2ca02c', width: 1.5 },
  }];
  if (DOS_DATA.has_ipa_dos) {
    excDosTraces.push({
      type: 'scatter', mode: 'lines',
      name: 'IPA (' + DOS_DATA.ipa_energy_levels.toUpperCase() + ', Q=0 only)',
      x: DOS_DATA.exc_axis, y: DOS_DATA.ipa_dos,
      line: { color: '#8c564b', width: 1.5 },
    });
  }
  const excDosLayout = dosLineLayout(
    DOS_DATA.has_ipa_dos ? 'Exciton DOS \u2014 excitonic vs. IPA' : 'Exciton DOS (BZ-averaged)',
    '\u03a9 (eV)', 'DOS (peak-normalized)');
  excDosLayout.xaxis.range = eexcRangeVal();
  if (DOS_DATA.has_ipa_dos) {
    excDosLayout.showlegend = true;
    excDosLayout.legend = { font: { size: 9 }, orientation: 'h', x: 0.5, xanchor: 'center', y: -0.3, yanchor: 'top' };
    excDosLayout.margin.b = 75;
  }
  Plotly.react('excdos-div', excDosTraces, excDosLayout);

  const absLayout = dosLineLayout('Optical Absorption (Q=0)', '\u03a9 (eV)', 'Absorption (a.u.)');
  absLayout.xaxis.range = eexcRangeVal();
  Plotly.react('abs-div', [{
    type: 'scatter', mode: 'lines', x: DOS_DATA.exc_axis, y: DOS_DATA.absorption,
    line: { color: '#d62728', width: 1.5 },
  }], absLayout);
}

// ── event wiring ──────────────────────────────────────────────────────────────
flavorSel.addEventListener('change', function() {
  // Keep the current Omega_exc / Raman shift cursor values across a flavor
  // switch (nearestIdx() already clamps gracefully if they fall outside the
  // new flavor's grid) so comparing the same point across flavors is easy.
  updateAll();
});
polSel.addEventListener('change', updateAll);
document.getElementById('share-y-checkbox').addEventListener('change', function() {
  updateSpectrum(); updateExcProfile();
});

document.getElementById('eexc-input').addEventListener('change', function() {
  updateMap(); updateSpectrum(); updateExcProfile();
});
document.getElementById('rshift-input').addEventListener('change', function() {
  updateMap(); updateSpectrum(); updateExcProfile();
});

// omega_ph / Eexc range controls (item 4) -- affect the Map/Spectrum/
// Excitation-profile axis ranges plus, for Eexc, the DOS panels' x-axis.
['ph-min-input', 'ph-max-input'].forEach(function(id) {
  document.getElementById(id).addEventListener('change', function() {
    updateMap(); updateSpectrum(); updateDosPanels();
  });
});
['eexc-min-input', 'eexc-max-input'].forEach(function(id) {
  document.getElementById(id).addEventListener('change', function() {
    updateMap(); updateExcProfile(); updateDosPanels();
  });
});

document.querySelectorAll('input[name="scale"]')
        .forEach(function(r) { r.addEventListener('change', updateMap); });
document.querySelectorAll('input[name="spec-scale"]')
        .forEach(function(r) { r.addEventListener('change', updateSpectrum); });
document.querySelectorAll('input[name="exc-scale"]')
        .forEach(function(r) { r.addEventListener('change', updateExcProfile); });
document.querySelectorAll('input[name="dos-scale"]')
        .forEach(function(r) { r.addEventListener('change', updateDosPanels); });

// ── initialise ────────────────────────────────────────────────────────────────
(async function() {
  const d0 = curData();
  setEexcInput(d0.excitation_energies[Math.floor(d0.excitation_energies.length / 2)]);
  setRshiftInput(d0.freq_axis_cm[Math.floor(d0.freq_axis_cm.length / 2)]);

  // Default omega_ph/Eexc ranges to [0, max] (of the initial flavor) rather
  // than blank/auto -- set once at init, same as the Eexc/Raman-shift inputs
  // above; editable afterward and not auto-updated on flavor switch.
  document.getElementById('ph-min-input').value   = 0;
  document.getElementById('ph-max-input').value   = d0.freq_axis_cm[d0.freq_axis_cm.length - 1];
  document.getElementById('eexc-min-input').value = 0;
  document.getElementById('eexc-max-input').value = d0.excitation_energies[d0.excitation_energies.length - 1];

  // First render — must await so Plotly attaches .on() to the element
  await updateMap();
  await updateSpectrum();
  await updateExcProfile();
  updateDosPanels();
  if (typeof updatePolarPanels === 'function') updatePolarPanels();

  // Map click → set BOTH Eexc (y) and Raman shift (x) → refresh all panels
  document.getElementById('map-div').on('plotly_click', function(evtData) {
    setEexcInput(evtData.points[0].y);
    setRshiftInput(evtData.points[0].x);
    updateAll();
  });
})();

__POLAR_JS__
</script>
</body>
</html>
"""

# ── --plot-polar-plots: HTML/JS fragments (kept out of the fixed template
# above so the file is byte-for-byte close to unchanged, size-wise, when the
# flag is off) ────────────────────────────────────────────────────────────
if args.plot_polar_plots:
    polar_controls = r"""  <div class="ctrl-group">
    <label>Polar radial scale</label>
    <div class="radio-row">
      <label><input type="radio" name="polar-scale" value="linear" checked> Linear</label>
      <label><input type="radio" name="polar-scale" value="log"> Log</label>
    </div>
  </div>"""
    polar_plots_row = r"""<div class="plots-polar">
  <div class="plot" id="polar-div"></div>
  <div class="plot" id="helicity-div"></div>
</div>"""
    polar_js = r"""
// ── polar (I_parallel/I_perp) + helicity (sigma_+/sigma_-) panels ─────────────
// (--plot-polar-plots) -- reuses eexcVal/rshiftVal/nearestIdx/curData from
// above; hooked into updateAll()/init via the `typeof updatePolarPanels ===
// 'function'` guards already present there. Not every flavor necessarily has
// a `polar` (or `polar.helicity`) sub-object -- render a placeholder for
// whichever piece is missing rather than erroring, mirroring the DOS-panel
// pattern (renderDosPlaceholder) above.
function isLogPolar() {
  return document.querySelector('input[name="polar-scale"]:checked').value === 'log';
}

function renderPolarPlaceholder() {
  document.getElementById('polar-div').innerHTML =
    '<div class="dos-placeholder">I<sub>&parallel;</sub>/I<sub>&perp;</sub> data not available for this flavor<br>'
    + '(rerun resonant_raman.py --polarized --polar-store-maps)</div>';
}
function renderHelicityPlaceholder() {
  document.getElementById('helicity-div').innerHTML =
    '<div class="dos-placeholder">Helicity (&sigma;<sub>+</sub>/&sigma;<sub>-</sub>) data not available for this flavor<br>'
    + '(rerun resonant_raman.py --polarized --helicity)</div>';
}

function buildPolarTraces(eexc, rshift) {
  const p   = curData().polar;
  const iE  = nearestIdx(p.excitation_energies, eexc);
  const iPh = nearestIdx(p.freq_axis_cm, rshift);
  var Ipar  = p.I_parallel[iE][iPh];
  var Iperp = p.I_perp[iE][iPh];
  if (isLogPolar()) {
    const floor = Math.max(1e-30, Math.min(...Ipar.concat(Iperp).filter(function(v){return v>0;})) || 1e-30);
    Ipar  = Ipar.map(function(v)  { return Math.log10(Math.max(v, floor)); });
    Iperp = Iperp.map(function(v) { return Math.log10(Math.max(v, floor)); });
  }
  const theta_deg = p.theta.map(function(t) { return t * 180 / Math.PI; });
  const theta_c = theta_deg.concat([theta_deg[0]]);
  return [
    { type: 'scatterpolar', mode: 'lines', name: 'I_parallel',
      r: Ipar.concat([Ipar[0]]),  theta: theta_c, line: { color: 'black', width: 2.5 } },
    { type: 'scatterpolar', mode: 'lines', name: 'I_perp',
      r: Iperp.concat([Iperp[0]]), theta: theta_c, line: { color: '#d62728', width: 2.5 } },
  ];
}

function buildPolarLayout(eexc_actual, ph_actual) {
  const p = curData().polar;
  return {
    title: {
      text: 'I<sub>&parallel;</sub> / I<sub>&perp;</sub> — Flavor ' + flavorSel.value
          + ' — Ω<sub>exc</sub> ≈ ' + eexc_actual.toFixed(4) + ' eV'
          + ', ω ≈ ' + ph_actual.toFixed(1) + ' cm⁻¹'
          + '<br><sub>n̂=[' + p.n_hat.map(function(v){return v.toFixed(2);}).join(',')
          + ']  ê₁=[' + p.e1.map(function(v){return v.toFixed(2);}).join(',') + ']</sub>',
      font: { size: 12 },
    },
    polar: { angularaxis: { direction: 'counterclockwise', rotation: 0 } },
    showlegend: true,
    // outside the plotting area (below), not overlapping the curves
    legend: { font: { size: 10 }, orientation: 'h', x: 0.5, xanchor: 'center', y: -0.12, yanchor: 'top' },
    margin: { l: 40, r: 40, t: 80, b: 55 },
  };
}

function buildHelicityTraces(eexc, rshift) {
  const d   = curData();
  const h   = d.polar.helicity;
  const iE  = nearestIdx(d.polar.excitation_energies, eexc);
  const iPh = nearestIdx(d.polar.freq_axis_cm, rshift);
  return [{
    type: 'bar',
    x: ['σ+σ+', 'σ+σ-', 'σ-σ+', 'σ-σ-'],
    y: [h.sigma_pp[iE][iPh], h.sigma_pm[iE][iPh], h.sigma_mp[iE][iPh], h.sigma_mm[iE][iPh]],
    marker: { color: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] },
  }];
}

function buildHelicityLayout(eexc_actual, ph_actual) {
  const d   = curData();
  const h   = d.polar.helicity;
  const iE  = nearestIdx(d.polar.excitation_energies, eexc_actual);
  const iPh = nearestIdx(d.polar.freq_axis_cm, ph_actual);
  const rho = h.rho_circ[iE][iPh];
  return {
    title: {
      text: 'Helicity (' + h.convention + ') — ρ<sub>circ</sub> ≈ ' + rho.toFixed(3)
          + '<br><sub>Ω<sub>exc</sub> ≈ ' + eexc_actual.toFixed(4) + ' eV, '
          + 'ω ≈ ' + ph_actual.toFixed(1) + ' cm⁻¹</sub>',
      font: { size: 12 },
    },
    yaxis: { title: 'Raman Intensity (a.u.)' },
    margin: { l: 55, r: 10, t: 60, b: 40 },
  };
}

function updatePolarPanels() {
  const d = curData();
  if (!d.polar) { renderPolarPlaceholder(); renderHelicityPlaceholder(); return; }
  const eexc   = isNaN(eexcVal())   ? d.polar.excitation_energies[0] : eexcVal();
  const rshift = isNaN(rshiftVal()) ? d.polar.freq_axis_cm[0]        : rshiftVal();
  const iE  = nearestIdx(d.polar.excitation_energies, eexc);
  const iPh = nearestIdx(d.polar.freq_axis_cm, rshift);
  Plotly.react('polar-div', buildPolarTraces(eexc, rshift),
               buildPolarLayout(d.polar.excitation_energies[iE], d.polar.freq_axis_cm[iPh]));
  if (d.polar.helicity) {
    Plotly.react('helicity-div', buildHelicityTraces(eexc, rshift),
                 buildHelicityLayout(d.polar.excitation_energies[iE], d.polar.freq_axis_cm[iPh]));
  } else {
    renderHelicityPlaceholder();
  }
}

document.querySelectorAll('input[name="polar-scale"]')
        .forEach(function(r) { r.addEventListener('change', updatePolarPanels); });
"""
else:
    polar_controls = ''
    polar_plots_row = ''
    polar_js = ''

html_out = (HTML_TEMPLATE
            .replace('__DATA__',       data_json)
            .replace('__POL_LABELS__', pol_json)
            .replace('__DOS_DATA__',   dos_json)
            .replace('__POLAR_CONTROLS__',  polar_controls)
            .replace('__POLAR_PLOTS_ROW__', polar_plots_row)
            .replace('__POLAR_JS__',        polar_js))

out_path = Path(args.output)
out_path.write_text(html_out, encoding='utf-8')
size_mb = out_path.stat().st_size / 1e6
print(f'\nSaved: {out_path}  ({size_mb:.1f} MB)')
print('Open in any browser — no server required.')
