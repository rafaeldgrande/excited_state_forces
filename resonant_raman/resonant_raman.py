
import sys
from pathlib import Path
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (k_B, rec_cm_to_eV, hbar, FLAVOR_DESC,
                    ignore_0_freq_modes, _downsample_idx, unpolarized_invariant)
sys.path.insert(0, str(Path(__file__).parent))
from polarization import (build_frame, polarization_vectors,
                          contract_first_order, contract_second_order,
                          alpha_plane_first_order, alpha_plane_second_order,
                          contract_plane, JONES_PLUS, JONES_MINUS)

config_dir = Path(__file__).parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

parser = argparse.ArgumentParser(
    description=(
        'Compute resonant Raman intensity maps.\n\n'
        'Required files per flavor:\n'
        '  Flavor 0  IPA first order\n'
        '            --ipa-first-order-file\n'
        '  Flavor 1  IPA second order\n'
        '            --ipa-second-order-file\n'
        '  Flavor 2  IPA first + second order\n'
        '            --ipa-first-order-file  +  --ipa-second-order-file\n'
        '  Flavor 3  First order, diagonal exciton-phonon only\n'
        '            --first-order-file\n'
        '  Flavor 4  First order, diagonal + off-diagonal exciton-phonon\n'
        '            --first-order-file\n'
        '  Flavor 5  Second order, triple resonance only\n'
        '            --second-order-file  (or --q-points-file for finite-q BZ average)\n'
        '  Flavor 6  Second order, double resonance only\n'
        '            --second-order-file  (Gamma-only -- double-resonance needs the\n'
        '            real 2nd-derivative el-ph, not available on the finite-q grid\n'
        '            in this pipeline; --q-points-file gives an all-zero result)\n'
        '  Flavor 7  Second order, double + triple resonance\n'
        '            --second-order-file  (Gamma-only, same caveat as flavor 6)\n'
        '  Flavor 8  First order (diag+offdiag) + second order (double+triple)\n'
        '            --first-order-file  +  --second-order-file  (Gamma-only)\n\n'
        'Note: --q-points-file enables BZ-averaged second-order Raman (finite-q phonons),\n'
        'meaningful for flavor 5 only (see above for why 6-8 are Gamma-only).\n'
        '  Row iq in the file maps to susceptibility_tensors_second_order_q_{iq}.h5.'
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--temperature',       type=float, default=300,
                    help='Temperature in Kelvin (default: 300)')
parser.add_argument('--first-order-file',  type=str,
                    default='susceptibility_tensors_first_order.h5',
                    help='HDF5 file from susceptibility_tensors_first_order.py '
                         '(default: susceptibility_tensors_first_order.h5)')
parser.add_argument('--second-order-file', type=str,
                    default='susceptibility_tensors_second_order.h5',
                    help='HDF5 file from susceptibility_tensors_second_order.py '
                         '(required for flavors 2 and 3; default: susceptibility_tensors_second_order.h5)')
parser.add_argument('--ipa-first-order-file', type=str,
                    default='susceptibility_tensors_first_order_IPA.h5',
                    help='HDF5 file from susceptibility_tensors_IPA.py '
                         '(required for flavors 6 and 8; default: susceptibility_tensors_first_order_IPA.h5)')
parser.add_argument('--ipa-second-order-file', type=str,
                    default='susceptibility_tensors_second_order_IPA.h5',
                    help='HDF5 file from susceptibility_tensors_IPA.py '
                         '(required for flavors 7 and 8; default: susceptibility_tensors_second_order_IPA.h5)')
parser.add_argument('--freqs-file',        type=str, default=None,
                    help='File with phonon frequencies in cm^-1 (optional; read from susceptibility h5 if not given)')
parser.add_argument('--flavor',            type=int, default=0,
                    choices=list(FLAVOR_DESC.keys()),
                    help='Which susceptibility to use: ' +
                         ', '.join(f'{k}={v}' for k, v in FLAVOR_DESC.items()) +
                         ' (default: 0)')
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
parser.add_argument('--q-points-file',     type=str, default=None,
                    help='File with q-point weights: rows of "qx qy qz weight". '
                         'Row iq=0 loads susceptibility_tensors_second_order_q_0.h5, etc. '
                         'Overrides --second-order-file for the second-order contribution.')
parser.add_argument('--polarized',         action='store_true',
                    help='Also compute angle-resolved polarized Raman intensities '
                         'I_parallel(theta), I_perp(theta) (see resonant_raman/README.md).')
parser.add_argument('--dtheta',            type=float, default=2.0 * np.pi / 100,
                    help='Theta step in radians (default: 2*pi/100)')
parser.add_argument('--dtheta-deg',        type=float, default=None,
                    help='Theta step in degrees; overrides --dtheta if given')
parser.add_argument('--n-hat',             type=float, nargs=3, default=[0.0, 0.0, 1.0],
                    help='Scattering-plane normal, Cartesian (default: 0 0 1)')
parser.add_argument('--theta-ref',         type=float, nargs=3, default=None,
                    help='Reference vector fixing theta=0 (default: auto -- x-hat, '
                         'or y-hat if within ~25 deg of --n-hat)')
parser.add_argument('--polar-output',      type=str, default=None,
                    help='Output file for polarized amplitudes '
                         '(default: resonant_raman_polar_flavor{flavor}.h5)')
parser.add_argument('--polar-store-maps',  action='store_true',
                    help='Also store full (theta, Omega, omega) intensity maps '
                         '(float32; can be large -- off by default, see README)')
parser.add_argument('--helicity',          action='store_true',
                    help='Also compute the four helicity-resolved intensities '
                         'I_sigma+sigma+, I_sigma+sigma-, I_sigma-sigma+, I_sigma-sigma- '
                         '(shares --n-hat/--theta-ref with --polarized; see README)')
parser.add_argument('--helicity-convention', type=str, default='jones',
                    choices=['jones', 'propagation'],
                    help='jones (default): scattered sigma+ = e_+ (fixed lab frame, '
                         'matches QERaman/most 2D-materials literature). propagation: '
                         'scattered sigma+ = e_- (true helicity vs. the outgoing -n_hat '
                         'direction in backscattering). See README for which is which.')
parser.add_argument('--jones-incident',    type=float, nargs=3, default=None,
                    help='Explicit incident Jones vector "Px Py phi" (phi in radians), '
                         'overriding the sigma+/- presets (default: None)')
parser.add_argument('--jones-scattered',   type=float, nargs=3, default=None,
                    help='Explicit scattered Jones vector "Px Py phi", same convention '
                         'as --jones-incident (default: None)')
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
q_points_file        = args.q_points_file
polarized            = args.polarized
dtheta               = np.deg2rad(args.dtheta_deg) if args.dtheta_deg is not None else args.dtheta
n_hat_arg            = np.array(args.n_hat)
theta_ref_arg        = np.array(args.theta_ref) if args.theta_ref is not None else None
polar_store_maps     = args.polar_store_maps
helicity             = args.helicity
helicity_convention  = args.helicity_convention


def _jones_from_args(triplet):
    """Px Py phi -> complex 2-vector (Px, Py*exp(i*phi)) in the {e1,e2} basis."""
    Px, Py, phi = triplet
    return np.array([Px, Py * np.exp(1j * phi)])


jones_incident_arg  = _jones_from_args(args.jones_incident)  if args.jones_incident  is not None else None
jones_scattered_arg = _jones_from_args(args.jones_scattered) if args.jones_scattered is not None else None

flavor_label = FLAVOR_DESC[flavor]
polar_output = args.polar_output or f'resonant_raman_polar_flavor{flavor}.h5'
print(f'Flavor {flavor}: {flavor_label}')

# Derived flags (flavor numbering per FLAVOR_DESC in common/utils.py --
# renumbered 2026-08-05: IPA flavors 0-2, first-order-only 3-4, second-order
# 5-7 (triple/double/both), "everything" 8. has_triple/has_double are only
# meaningful for the non-IPA second-order path (BSE flavors 5-8); IPA's
# second order is a single combined tensor, no triple/double split.
is_ipa           = flavor in {0, 1, 2}
has_first_order  = flavor in {0, 2, 3, 4, 8}
use_diag_only    = flavor == 3
has_second_order = flavor in {1, 2, 5, 6, 7, 8}
has_triple       = flavor in {5, 7, 8}
has_double       = flavor in {6, 7, 8}

cart_dir = ['x', 'y', 'z']

freqs_rec_cm = None
# Try to read phonon frequencies from whichever susceptibility h5 file is available.
# In --q-points-file (BZ-averaged) mode, second_order_file itself is never written;
# per-q files named ..._q_{iq}.h5 are used instead (see the q_points_file branch
# below), so try the iq=0 file too.
_freq_candidates = [first_order_file, second_order_file, ipa_first_order_file, ipa_second_order_file]
if q_points_file is not None:
    _freq_candidates.insert(1, second_order_file.replace('.h5', '_q_0.h5'))
    _freq_candidates.insert(2, ipa_second_order_file.replace('.h5', '_q_0.h5'))
for _h5 in _freq_candidates:
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
        alpha_tensor_diag       = f['alpha_tensor_d2'][:]        # h5 dataset name kept for compatibility
        alpha_tensor_full       = f['alpha_tensor_d3'][:]        # (misleadingly-named, see README) diag+offdiag
    alpha_tensor_first_order = alpha_tensor_diag if use_diag_only else alpha_tensor_full

alpha_tensor_second_order = None
excitation_energies_2nd   = None
q_contributions = []  # populated when --q-points-file is given

if has_second_order and not is_ipa:
    if q_points_file is not None:
        # q-averaged second order: load one h5 per q-point
        _q_data = np.loadtxt(q_points_file)
        if _q_data.ndim == 1:
            _q_data = _q_data[np.newaxis, :]
        _q_weights = _q_data[:, 3]
        _q_norm    = _q_weights.sum()
        print(f'Loading q-averaged second-order susceptibilities from {len(_q_weights)} q-points (q_points.dat)')
        for iq, w_q in enumerate(_q_weights):
            fname = f'susceptibility_tensors_second_order_q_{iq}.h5'
            print(f'  iq={iq}: {fname}  (weight={w_q})')
            with h5py.File(fname, 'r') as f:
                exc_en_q   = f['excitation_energies'][:]
                freqs_q_cm = f['phonon_frequencies_cm'][:]
                # Same 3-way triple/double/both choice as the single-file path
                # below -- kept consistent so a flavor's meaning doesn't
                # silently change between --q-points-file and single-file runs.
                if has_triple:
                    alpha_q = f['alpha_tensor_triple_resonance'][:]
                else:
                    _tr_ds = f['alpha_tensor_triple_resonance']
                    alpha_q = np.zeros(_tr_ds.shape, dtype=_tr_ds.dtype)
                if has_double:
                    alpha_db_q = f['alpha_tensor_double_resonance'][:]
                    for imode in range(alpha_q.shape[2]):
                        if has_triple:
                            alpha_q[:, :, imode, imode, :] += alpha_db_q[:, :, imode, :]
                        else:
                            alpha_q[:, :, imode, imode, :] = alpha_db_q[:, :, imode, :]
            q_contributions.append({'weight': w_q, 'alpha': alpha_q,
                                     'freqs_cm': freqs_q_cm, 'exc_en': exc_en_q})
        excitation_energies_2nd = q_contributions[0]['exc_en']
    else:
        print(f'Reading second-order susceptibilities from {second_order_file}')
        with h5py.File(second_order_file, 'r') as f:
            excitation_energies_2nd = f['excitation_energies'][:]
            # 3-way choice: triple only (flavor 5), double only (flavor 6,
            # new -- start from zero, SET the diagonal rather than fold into
            # a loaded triple term), or both folded together (flavor 7, 8).
            if has_triple:
                alpha_tensor_second_order = f['alpha_tensor_triple_resonance'][:]
            else:
                _tr_ds = f['alpha_tensor_triple_resonance']
                alpha_tensor_second_order = np.zeros(_tr_ds.shape, dtype=_tr_ds.dtype)
            if has_double:
                alpha_tensor_double_res = f['alpha_tensor_double_resonance'][:]
        if has_double:
            for imode in range(Nmodes):
                if has_triple:
                    alpha_tensor_second_order[:, :, imode, imode, :] += alpha_tensor_double_res[:, :, imode, :]
                else:
                    alpha_tensor_second_order[:, :, imode, imode, :] = alpha_tensor_double_res[:, :, imode, :]

if flavor in {0, 2}:
    print(f'Reading IPA first-order susceptibilities from {ipa_first_order_file}')
    with h5py.File(ipa_first_order_file, 'r') as f:
        excitation_energies_1st  = f['excitation_energies'][:]
        alpha_tensor_first_order = f['susceptibility_tensor_first_order'][:]

if flavor in {1, 2}:
    if q_points_file is not None:
        _q_data = np.loadtxt(q_points_file)
        if _q_data.ndim == 1:
            _q_data = _q_data[np.newaxis, :]
        _q_weights = _q_data[:, 3]
        _q_norm    = _q_weights.sum()
        print(f'Loading q-averaged IPA second-order susceptibilities from {len(_q_weights)} q-points (q_points.dat)')
        for iq, w_q in enumerate(_q_weights):
            fname = f'susceptibility_tensors_second_order_IPA_q_{iq}.h5'
            print(f'  iq={iq}: {fname}  (weight={w_q})')
            with h5py.File(fname, 'r') as f:
                exc_en_q   = f['excitation_energies'][:]
                alpha_q    = f['susceptibility_tensor_second_order'][:]
                freqs_q_cm = f['phonon_frequencies_cm'][:]
            q_contributions.append({'weight': w_q, 'alpha': alpha_q,
                                     'freqs_cm': freqs_q_cm, 'exc_en': exc_en_q})
        excitation_energies_2nd = q_contributions[0]['exc_en']
    else:
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
            excitation_energies_2nd = excitation_energies_2nd[ie2]
            if q_contributions:
                for _d in q_contributions:
                    _d['alpha']  = _d['alpha'][..., ie2]
                    _d['exc_en'] = _d['exc_en'][ie2]
            else:
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

# I(alpha,beta) = Σ_i |w_i · α¹[i]|²  +  Σ_ij |w_i·w_j · α²[i,j]|²
# (first- and second-order contributions are summed INCOHERENTLY -- each
# mode/mode-pair's amplitude is squared before summing, not after -- since
# they are distinct final states; see resonant_raman/README.md's "Polarized
# Raman" subsection for the derivation of the polarized-intensity analogue)
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

# For q-averaged second order, the axis must cover up to 2 * max(all q-point freqs)
if q_contributions:
    _max_2nd = max(d['freqs_cm'].max() for d in q_contributions)
else:
    _max_2nd = max_vib_freq

freq_axis_lo = max(0.0, min_vib_freq - 5 * gamma_lor)
freq_axis_hi = (2 * _max_2nd if has_second_order else max_vib_freq) + 5 * gamma_lor
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
if has_second_order and not q_contributions:
    # Single-file path: use gamma-point frequencies
    freq_pairs = (freqs_rec_cm[:, np.newaxis] +
                  freqs_rec_cm[np.newaxis, :]).ravel()                 # (Nmodes²,)
    w_pairs    = (phonon_weight[:, np.newaxis] *
                  phonon_weight[np.newaxis, :]).ravel()                # (Nmodes²,)
    if ignore_0_freq_modes:
        valid_pairs = (np.outer(valid_modes, valid_modes).ravel() &
                       (freq_pairs >= 1e-2))
    else:
        valid_pairs = np.ones(Nmodes**2, dtype=bool)

    freq_pairs_v = freq_pairs[valid_pairs]
    w_pairs_v    = w_pairs[valid_pairs]                                # (Npairs_v,)
    lor_2nd = (gamma_lor**2 /
               ((freq_axis[np.newaxis, :] - freq_pairs_v[:, np.newaxis])**2
                + gamma_lor**2))                                       # (Npairs_v, Nfreq_ph)

elif q_contributions:
    # q-averaged path: precompute per-q phonon weights, pair masks, and Lorentzians
    for _d in q_contributions:
        _freqs_q_eV = _d['freqs_cm'] * rec_cm_to_eV
        _safe_q_eV  = np.maximum(_freqs_q_eV, 1e-8)
        _bose_q     = 1.0 / (np.exp(_safe_q_eV / (k_B * T)) - 1)
        _ph_wt_q    = np.sqrt((_bose_q + 1) * hbar / (2 * _safe_q_eV))
        _Nm_q       = len(_d['freqs_cm'])
        _fp_q       = (_d['freqs_cm'][:, None] + _d['freqs_cm'][None, :]).ravel()
        _wp_q       = (_ph_wt_q[:, None] * _ph_wt_q[None, :]).ravel()
        if ignore_0_freq_modes:
            _vm_q    = _d['freqs_cm'] > 1e-2
            _valid_q = np.outer(_vm_q, _vm_q).ravel() & (_fp_q >= 1e-2)
        else:
            _valid_q = np.ones(_Nm_q**2, dtype=bool)
        _lor_q = (gamma_lor**2 /
                  ((freq_axis[np.newaxis, :] - _fp_q[_valid_q, np.newaxis])**2
                   + gamma_lor**2))                                    # (Npairs_v_q, Nfreq_ph)
        _d['valid_pairs'] = _valid_q
        _d['w_pairs_v']   = _wp_q[_valid_q]
        _d['lor_2nd']     = _lor_q

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
        if q_contributions:
            for _d in q_contributions:
                _Nm_q      = _d['alpha'].shape[2]
                _ap_q      = (_d['alpha'][ialpha, ibeta]
                              .reshape(_Nm_q**2, -1)[_d['valid_pairs']])   # (Npairs_v_q, Nfreq)
                _int_2nd   = np.abs(_d['w_pairs_v'][:, np.newaxis] * _ap_q)**2
                raman_map += (_d['weight'] / _q_norm) * (_int_2nd.T @ _d['lor_2nd'])
        elif has_second_order:
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

if q_contributions:
    for _d in q_contributions:
        _Nm_q  = _d['alpha'].shape[2]
        _Nf_q  = _d['alpha'].shape[4]
        _aw_q  = (_d['w_pairs_v'][np.newaxis, np.newaxis, :, np.newaxis]
                  * _d['alpha'].reshape(3, 3, _Nm_q**2, _Nf_q)[:, :, _d['valid_pairs'], :])
        _int_u = unpolarized_invariant(_aw_q)            # (Npairs_v_q, Nfreq)
        raman_map_unpol += (_d['weight'] / _q_norm) * (_int_u.T @ _d['lor_2nd'])
elif has_second_order:
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
# Polarized (angle-resolved) Raman intensities -- optional, purely additive.
# Contract Cartesian indices with the polarization vectors BEFORE squaring
# (unlike raman_maps above, which squares each (ialpha,ibeta) independently
# and therefore cannot be used to reconstruct I_parallel(theta)/I_perp(theta)
# -- see resonant_raman/README.md's "Polarized Raman" subsection).
# ---------------------------------------------------------------------------
if polarized:
    print('Computing polarized (angle-resolved) Raman intensities...')
    e1, e2, n_hat = build_frame(n_hat_arg, theta_ref_arg)
    Ntheta = int(round(2.0 * np.pi / dtheta))
    theta  = np.arange(Ntheta) * dtheta          # half-open [0, 2pi)
    e_i, e_par, e_perp = polarization_vectors(theta, e1, e2)

    map_parallel = np.zeros((Ntheta, Nfreq, Nfreq_ph))
    map_perp     = np.zeros((Ntheta, Nfreq, Nfreq_ph))
    M_par_1st = M_perp_1st = None                  # (Ntheta, Nvalid, Nfreq_1st)
    M_par_2nd = M_perp_2nd = None                  # (Ntheta, Nmodes, Nmodes, Nfreq) -- single-file path only
    M_par_2nd_q = M_perp_2nd_q = None               # (Nq, Ntheta, Nmodes, Nmodes, Nfreq) -- q-averaged path

    # --- First order: same zero-pad-to-Nfreq / square / broaden pattern as
    # the Cartesian loop above (lines ~366-371), just with the contracted
    # scalar amplitude M in place of alpha_v, no (ialpha,ibeta) loop. ---
    if has_first_order:
        M_par_1st_full  = contract_first_order(alpha_tensor_first_order, e_par,  e_i)  # (Ntheta, Nmodes, Nfreq_1st)
        M_perp_1st_full = contract_first_order(alpha_tensor_first_order, e_perp, e_i)
        M_par_1st  = M_par_1st_full[:, valid_modes, :]
        M_perp_1st = M_perp_1st_full[:, valid_modes, :]

        for M_dir, map_dir in ((M_par_1st, map_parallel), (M_perp_1st, map_perp)):
            int_1st = np.zeros((Ntheta, valid_modes.sum(), Nfreq))
            int_1st[:, :, :Nfreq_1st] = np.abs(
                phonon_weight[np.newaxis, valid_modes, np.newaxis] * M_dir)**2
            map_dir += np.einsum('tmf,mw->tfw', int_1st, lor_1st, optimize=True)

    # --- Second order, q-averaged path: contract each q-point's tensor
    # BEFORE the reshape/mask/square, then accumulate with the SAME
    # weight-applied-after-squaring convention as the Cartesian q loop above
    # (lines ~374-380). ---
    if q_contributions:
        M_par_2nd_q, M_perp_2nd_q = [], []
        for _d in q_contributions:
            _Nm_q = _d['alpha'].shape[2]
            _Mp_q = contract_second_order(_d['alpha'], e_par,  e_i)    # (Ntheta, Nmodes_q, Nmodes_q, Nfreq)
            _Mx_q = contract_second_order(_d['alpha'], e_perp, e_i)
            M_par_2nd_q.append(_Mp_q)
            M_perp_2nd_q.append(_Mx_q)
            for M_q, map_dir in ((_Mp_q, map_parallel), (_Mx_q, map_perp)):
                M_flat  = M_q.reshape(Ntheta, _Nm_q**2, -1)[:, _d['valid_pairs'], :]
                int_2nd = np.abs(_d['w_pairs_v'][np.newaxis, :, np.newaxis] * M_flat)**2
                map_dir += (_d['weight'] / _q_norm) * np.einsum(
                    'tpf,pw->tfw', int_2nd, _d['lor_2nd'], optimize=True)

    # --- Second order, single-file (Gamma-only) path ---
    elif has_second_order:
        M_par_2nd  = contract_second_order(alpha_tensor_second_order, e_par,  e_i)  # (Ntheta, Nmodes, Nmodes, Nfreq)
        M_perp_2nd = contract_second_order(alpha_tensor_second_order, e_perp, e_i)
        for M_full, map_dir in ((M_par_2nd, map_parallel), (M_perp_2nd, map_perp)):
            M_flat  = M_full.reshape(Ntheta, Nmodes**2, Nfreq)[:, valid_pairs, :]
            int_2nd = np.abs(w_pairs_v[np.newaxis, :, np.newaxis] * M_flat)**2
            map_dir += np.einsum('tpf,pw->tfw', int_2nd, lor_2nd, optimize=True)

    # --- Helicity-resolved intensities (HELICITY_RAMAN_SPEC.md) -- fixed
    # polarization pairs (no theta axis), so these maps are small and written
    # unconditionally once --helicity is set. Uses the small in-plane block
    # alpha_plane_* (2,2,...) rather than re-contracting the full (3,3,...)
    # tensor -- see polarization.py's helicity section for why this is both
    # cheap and mathematically identical to going through the full tensor.
    sigma_pp = sigma_pm = sigma_mp = sigma_mm = rho_circ = None
    if helicity:
        print(f'Computing helicity-resolved intensities (convention: {helicity_convention})...')
        # Under 'jones', scattered sigma+ = e_+ (matches QERaman); under
        # 'propagation', scattered sigma+ = e_- (true helicity vs. the
        # outgoing -n_hat direction in backscattering). Incident is always
        # unambiguous (propagates along +n_hat). See README for the full
        # discussion of why this choice matters and swaps every label.
        if helicity_convention == 'jones':
            c_s_plus, c_s_minus = JONES_PLUS, JONES_MINUS
        else:
            c_s_plus, c_s_minus = JONES_MINUS, JONES_PLUS
        c_i_plus, c_i_minus = JONES_PLUS, JONES_MINUS
        if jones_incident_arg is not None:
            c_i_plus = c_i_minus = jones_incident_arg
        if jones_scattered_arg is not None:
            c_s_plus = c_s_minus = jones_scattered_arg

        sigma_pp = np.zeros((Nfreq, Nfreq_ph))
        sigma_pm = np.zeros((Nfreq, Nfreq_ph))
        sigma_mp = np.zeros((Nfreq, Nfreq_ph))
        sigma_mm = np.zeros((Nfreq, Nfreq_ph))
        _sigma_pairs = ((c_s_plus, c_i_plus, sigma_pp), (c_s_plus, c_i_minus, sigma_pm),
                        (c_s_minus, c_i_plus, sigma_mp), (c_s_minus, c_i_minus, sigma_mm))

        if has_first_order:
            alpha_plane_1st = alpha_plane_first_order(alpha_tensor_first_order, e1, e2)  # (2,2,Nmodes,Nfreq_1st)
            for c_s, c_i, sigma_map in _sigma_pairs:
                M = contract_plane(alpha_plane_1st, c_s, c_i)[valid_modes, :]  # (Nvalid, Nfreq_1st)
                int_1st = np.zeros((valid_modes.sum(), Nfreq))
                int_1st[:, :Nfreq_1st] = np.abs(phonon_weight[valid_modes, np.newaxis] * M)**2
                sigma_map += int_1st.T @ lor_1st

        if q_contributions:
            for _d in q_contributions:
                _Nm_q = _d['alpha'].shape[2]
                _ap_q = alpha_plane_second_order(_d['alpha'], e1, e2)  # (2,2,Nmodes_q,Nmodes_q,Nfreq)
                for c_s, c_i, sigma_map in _sigma_pairs:
                    M2 = contract_plane(_ap_q, c_s, c_i).reshape(_Nm_q**2, -1)[_d['valid_pairs']]
                    int_2nd = np.abs(_d['w_pairs_v'][:, np.newaxis] * M2)**2
                    sigma_map += (_d['weight'] / _q_norm) * (int_2nd.T @ _d['lor_2nd'])
        elif has_second_order:
            alpha_plane_2nd = alpha_plane_second_order(alpha_tensor_second_order, e1, e2)  # (2,2,Nmodes,Nmodes,Nfreq)
            for c_s, c_i, sigma_map in _sigma_pairs:
                M2 = contract_plane(alpha_plane_2nd, c_s, c_i).reshape(Nmodes**2, Nfreq)[valid_pairs]
                int_2nd = np.abs(w_pairs_v[:, np.newaxis] * M2)**2
                sigma_map += int_2nd.T @ lor_2nd

        # Degree of circular polarization for incident sigma+ (denominator guarded).
        _denom = sigma_pp + sigma_pm
        rho_circ = np.where(_denom > 1e-30, (sigma_pp - sigma_pm) / np.where(_denom > 1e-30, _denom, 1.0), 0.0)

    # --- Save amplitudes (small; regenerate any map/spectrum/polar slice
    # from these) plus, optionally, the full maps. Amplitudes are stored at
    # their NATIVE frequency-axis length (Nfreq_1st for first order, which
    # can differ from the main Nfreq used for raman_maps/the maps below --
    # NOT zero-padded, to avoid amplitude data being misread as genuine
    # zero-intensity points; the maps ARE zero-padded, matching raman_maps).
    # For the q-averaged 2nd-order term, amplitudes are stored per q-point
    # (q_M_parallel_second_order etc., shape (Nq, Ntheta, Nmodes, Nmodes,
    # Nfreq)) plus q_weights, mirroring this codebase's existing per-q-file
    # convention -- skipped with a warning if q-points have differing
    # Nmodes (not expected in practice, but not assumed away either). ---
    print(f'Saving polarized Raman amplitudes to {polar_output}')
    with h5py.File(polar_output, 'w') as hf:
        hf.attrs['flavor']       = flavor
        hf.attrs['flavor_label'] = flavor_label
        hf.attrs['temperature']  = T
        hf.attrs['gamma_lor']    = gamma_lor
        hf.attrs['dtheta']       = dtheta
        hf.attrs['theta_ref']    = 'auto' if theta_ref_arg is None else str(theta_ref_arg.tolist())
        if helicity:
            hf.attrs['helicity_convention'] = helicity_convention
        hf.create_dataset('theta', data=theta)
        hf.create_dataset('n_hat', data=n_hat)
        hf.create_dataset('e1',    data=e1)
        hf.create_dataset('e2',    data=e2)
        hf.create_dataset('excitation_energies',   data=excitation_energies)
        hf.create_dataset('phonon_frequencies_cm', data=freqs_rec_cm)
        hf.create_dataset('freq_axis_cm',          data=freq_axis)  # (Nfreq_ph,) -- needed to plot raman_map_parallel/perpendicular
        _kw = dict(compression='gzip', compression_opts=4)

        if has_first_order:
            hf.create_dataset('excitation_energies_1st', data=excitation_energies_1st)
            hf.create_dataset('M_parallel_first_order', data=M_par_1st,  **_kw)
            hf.create_dataset('M_perp_first_order',     data=M_perp_1st, **_kw)

        if q_contributions:
            hf.create_dataset('q_weights', data=np.array([d['weight'] for d in q_contributions]))
            if len({d['alpha'].shape[2] for d in q_contributions}) == 1:
                hf.create_dataset('q_M_parallel_second_order',
                                   data=np.stack(M_par_2nd_q), **_kw)
                hf.create_dataset('q_M_perp_second_order',
                                   data=np.stack(M_perp_2nd_q), **_kw)
            else:
                print('  WARNING: q-points have differing Nmodes -- skipping amplitude '
                      'storage for the q-averaged 2nd-order term (use --polar-store-maps instead)')
        elif has_second_order:
            hf.create_dataset('M_parallel_second_order', data=M_par_2nd,  **_kw)
            hf.create_dataset('M_perp_second_order',     data=M_perp_2nd, **_kw)

        if helicity:
            _kw32 = dict(dtype=np.float32, compression='gzip', compression_opts=4)
            hf.create_dataset('raman_map_sigma_pp', data=sigma_pp, **_kw32)
            hf.create_dataset('raman_map_sigma_pm', data=sigma_pm, **_kw32)
            hf.create_dataset('raman_map_sigma_mp', data=sigma_mp, **_kw32)
            hf.create_dataset('raman_map_sigma_mm', data=sigma_mm, **_kw32)
            hf.create_dataset('degree_circular_polarization', data=rho_circ, **_kw32)

        if polar_store_maps:
            hf.create_dataset('raman_map_parallel',
                               data=map_parallel.astype(np.float32), **_kw)
            hf.create_dataset('raman_map_perpendicular',
                               data=map_perp.astype(np.float32), **_kw)
    print('Polarized Raman done.')

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
