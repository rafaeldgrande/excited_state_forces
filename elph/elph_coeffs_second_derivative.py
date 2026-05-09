"""
elph_coeffs_second_derivative.py
=================================
Compute second-order electron-phonon coupling coefficients from the
fine-grid el-ph file produced by interpolate_elph_bgw.py.

Theory
------
Second-order e-ph matrix elements via second-order perturbation theory:

    g2[alpha, k, n, m] = - sum_l g[alpha, k, n, l] * g[alpha, k, l, m] / (E_n - E_l)
                         - sum_l g[alpha, k, n, l] * g[alpha, k, l, m] / (E_m - E_l)

where:
  alpha : Cartesian atomic DOF index (3*Nat total)
  k     : k-point
  n, m  : band indices (conduction-conduction or valence-valence block)
  E_n   : quasiparticle energy of band n at k  (units must match g: Ry)

The input Cartesian el-ph from interpolate_elph_bgw.py (elph_fine_cond_cart /
elph_fine_val_cart, units Ry/bohr) is used directly — no displacement-pattern
rotation is needed.  The result g2 has units Ry/bohr^2.

The output is saved in the same HDF5 format as elph_fine.h5, so it can be
used as elph_fine_h5_file in forces.inp with use_second_derivatives_elph_coeffs = True.

Usage
-----
python elph_coeffs_second_derivative.py \\
    --elph_fine elph_fine.h5 \\
    --eqp eqp1.dat \\
    --Nval 13 \\
    --out 2nd_order_elph_fine.h5
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import TOL_ZERO, eV2ry


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _inv_dE(E):
    """
    E : (Nk, Nb) quasiparticle energies (Ry).
    Returns (Nk, Nb, Nb) where [ik, n, m] = 1/(E_n - E_m),
    set to 0.0 where |E_n - E_m| <= TOL_ZERO.
    """
    dE = E[:, :, None] - E[:, None, :]      # (Nk, Nb, Nb)
    mask = np.abs(dE) > TOL_ZERO
    return np.where(mask, 1.0 / np.where(mask, dE, 1.0), 0.0)


def compute_g2_cart(g_cart_q, inv_dE):
    """
    Vectorized second-order el-ph in the Cartesian basis for one q-point.

    Parameters
    ----------
    g_cart_q : (Npert, Nk, Nb, Nb)  first-order Cartesian el-ph, Ry/bohr
    inv_dE   : (Nk, Nb, Nb)         1/(E_n - E_m) in Ry^{-1}

    Returns
    -------
    g2 : (Npert, Nk, Nb, Nb)  second-order, Ry/bohr^2

    Derivation
    ----------
    g2[a,k,n,m] = -sum_l g[a,k,n,l]*g[a,k,l,m] * (1/(E_n-E_l) + 1/(E_m-E_l))

    term1: -sum_l g[n,l] * inv_dE[n,l] * g[l,m]  =  -(g * inv_dE) @ g
    term2: -sum_l g[n,l] * g[l,m] / (E_m-E_l)
           = +sum_l g[n,l] * g[l,m] * inv_dE[l,m]  since 1/(E_m-E_l) = -inv_dE[l,m]
           = g @ (g * inv_dE)
    """
    idE = inv_dE[None]                         # (1, Nk, Nb, Nb) — broadcast over Npert
    term1 = -(g_cart_q * idE) @ g_cart_q
    term2 =   g_cart_q @ (g_cart_q * idE)
    return term1 + term2


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_eqp(eqp_file, Nk, Nc, Nv, Nval):
    """
    Read fine-grid QP and DFT energies from eqp1.dat (eV).

    Band ordering matches bgw_interface_m.read_eqp_data and the el-ph convention:
      Conduction: ic=0 → LUMO  (ibnd = Nval+1 in the eqp file)
      Valence:    iv=0 → HOMO  (ibnd = Nval   in the eqp file)

    Returns
    -------
    Eqp_cond, Eqp_val, Edft_cond, Edft_val : (Nk, Nc/Nv) arrays in eV
    """
    Eqp_cond  = np.zeros((Nk, Nc));  Edft_cond = np.zeros((Nk, Nc))
    Eqp_val   = np.zeros((Nk, Nv));  Edft_val  = np.zeros((Nk, Nv))
    ik = -1
    with open(eqp_file) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] != '1':
                ik += 1
            else:
                ibnd = int(p[1])
                edft, eqp = float(p[2]), float(p[3])
                if ibnd > Nval:
                    ic = ibnd - Nval - 1    # 0-based: ic=0 → LUMO
                    if 0 <= ic < Nc:
                        Edft_cond[ik, ic] = edft
                        Eqp_cond[ik, ic]  = eqp
                else:
                    iv = Nval - ibnd        # 0-based: iv=0 → HOMO
                    if 0 <= iv < Nv:
                        Edft_val[ik, iv] = edft
                        Eqp_val[ik, iv]  = eqp
    return Eqp_cond, Eqp_val, Edft_cond, Edft_val


def _build_q_map(qpts_elph_cart, ph_qpts_cart, tol=1e-5):
    """
    Map each elph q-point (Cartesian) to its index in phonon_modes q-points.
    Returns array of ints, -1 where not matched.
    """
    n = len(qpts_elph_cart)
    idx = np.full(n, -1, dtype=int)
    for iq, q in enumerate(qpts_elph_cart):
        for iq_md, qm in enumerate(ph_qpts_cart):
            if np.linalg.norm(qm - q) < tol:
                idx[iq] = iq_md
                break
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cli = argparse.ArgumentParser(
        description='Compute second-order el-ph from elph_fine.h5 (Cartesian basis).')
    cli.add_argument('--elph_fine', default='elph_fine.h5',
                     help='Input from interpolate_elph_bgw.py (default: elph_fine.h5)')
    cli.add_argument('--eqp',       default='eqp1.dat',
                     help='Fine-grid QP energy file from absorption step (default: eqp1.dat)')
    cli.add_argument('--Nval',      required=True, type=int,
                     help='Number of valence bands in DFPT (QE nbnd convention)')
    cli.add_argument('--out',       default='2nd_order_elph_fine.h5',
                     help='Output HDF5 filename (default: 2nd_order_elph_fine.h5)')
    args = cli.parse_args()

    # ── 1. Load fine-grid el-ph ───────────────────────────────────────────────
    print(f'\nLoading {args.elph_fine} ...')
    _required = (
        'elph_fine_cond_cart', 'elph_fine_val_cart',
        'elph_fine_cond_mode', 'elph_fine_val_mode',
        'Kpoints_in_elph_file',
        'phonon_modes/eigenvectors', 'phonon_modes/frequencies', 'phonon_modes/qpoints',
    )
    with h5py.File(args.elph_fine, 'r') as fh:
        for key in _required:
            if key not in fh:
                raise KeyError(
                    f"'{key}' not found in {args.elph_fine}. "
                    f"Re-run interpolate_elph_bgw.py to regenerate.")

        g_cond_cart = fh['elph_fine_cond_cart'][:]   # (Nq, Npert, Nk, Nc, Nc)
        g_val_cart  = fh['elph_fine_val_cart'][:]    # (Nq, Npert, Nk, Nv, Nv)
        Kpoints     = fh['Kpoints_in_elph_file'][:]  # (Nk, 3) crystal coords
        evecs       = fh['phonon_modes/eigenvectors'][:]  # (Nq_md, Nmodes, Nat, 3)
        freqs       = fh['phonon_modes/frequencies'][:]   # (Nq_md, Nmodes) cm^-1
        ph_qpts     = fh['phonon_modes/qpoints'][:]       # (Nq_md, 3) Cartesian 2pi/a

        has_qcart = 'qpoints_cart' in fh
        has_qcrys = 'qpoints_crystal' in fh
        qpts_cart_elph = fh['qpoints_cart'][:]    if has_qcart else None
        qpts_crys_elph = fh['qpoints_crystal'][:] if has_qcrys else None

    Nq, Npert, Nk, Nc, _ = g_cond_cart.shape
    Nv     = g_val_cart.shape[3]
    Nq_md  = evecs.shape[0]
    Nmodes = evecs.shape[1]
    Nat    = evecs.shape[2]

    print(f'  g_cond_cart : {g_cond_cart.shape}  (Nq, Npert, Nk, Nc, Nc)')
    print(f'  g_val_cart  : {g_val_cart.shape}   (Nq, Npert, Nk, Nv, Nv)')
    print(f'  Nq={Nq}, Npert={Npert}, Nk={Nk}, Nc={Nc}, Nv={Nv}')
    print(f'  Nmodes={Nmodes}, Nat={Nat}, Nq_md={Nq_md}')

    # ── 2. Build elph-q → phonon_modes-q index map ───────────────────────────
    if has_qcart:
        iq_to_md = _build_q_map(qpts_cart_elph, ph_qpts)
        n_matched = np.sum(iq_to_md >= 0)
        print(f'  q-matching (Cartesian): {n_matched}/{Nq} matched')
    elif Nq_md == Nq:
        iq_to_md = np.arange(Nq, dtype=int)
        print('  WARNING: qpoints_cart not in file — using index-based q-matching '
              '(valid only if elph and phonon_modes share the same q-ordering)')
    else:
        iq_to_md = np.zeros(Nq, dtype=int)
        print(f'  WARNING: Nq_md={Nq_md} != Nq={Nq} and qpoints_cart absent — '
              f'mapping all q-points to phonon_modes index 0')

    # ── 3. Read QP energies ───────────────────────────────────────────────────
    print(f'\nReading QP energies from {args.eqp} ...')
    Eqp_cond, Eqp_val, Edft_cond, Edft_val = read_eqp(
        args.eqp, Nk, Nc, Nv, args.Nval)
    print(f'  Eqp_cond : shape {Eqp_cond.shape},  '
          f'range [{Eqp_cond.min():.3f}, {Eqp_cond.max():.3f}] eV')
    print(f'  Eqp_val  : shape {Eqp_val.shape},   '
          f'range [{Eqp_val.min():.3f}, {Eqp_val.max():.3f}] eV')

    # Convert eV → Ry for energy denominators (g is in Ry/bohr)
    E_cond_ry = Eqp_cond * eV2ry   # (Nk, Nc)
    E_val_ry  = Eqp_val  * eV2ry   # (Nk, Nv)

    inv_dE_cond = _inv_dE(E_cond_ry)   # (Nk, Nc, Nc)
    inv_dE_val  = _inv_dE(E_val_ry)    # (Nk, Nv, Nv)

    # Flattened eigenvectors for mode projection: (Nq_md, Nmodes, Npert)
    evec_flat = evecs.reshape(Nq_md, Nmodes, Npert)

    # ── 4. Compute second-order el-ph ─────────────────────────────────────────
    g2_cond_cart = np.zeros_like(g_cond_cart)
    g2_val_cart  = np.zeros_like(g_val_cart)
    g2_cond_mode = np.zeros((Nq, Nmodes, Nk, Nc, Nc), dtype=np.complex128)
    g2_val_mode  = np.zeros((Nq, Nmodes, Nk, Nv, Nv), dtype=np.complex128)

    print(f'\nComputing second-order el-ph for {Nq} q-point(s) ...')
    for iq in range(Nq):
        g2_cond_cart[iq] = compute_g2_cart(g_cond_cart[iq], inv_dE_cond)
        g2_val_cart[iq]  = compute_g2_cart(g_val_cart[iq],  inv_dE_val)

        iq_md = int(iq_to_md[iq])
        if iq_md >= 0:
            e = evec_flat[iq_md]   # (Nmodes, Npert)
            g2_cond_mode[iq] = np.einsum('va,aknm->vknm', e, g2_cond_cart[iq])
            g2_val_mode[iq]  = np.einsum('va,aknm->vknm', e, g2_val_cart[iq])
        else:
            print(f'  WARNING: iq={iq} — no matching phonon_modes q-point; '
                  f'g2_mode will be zero for this q.')

        print(f'  iq={iq+1}/{Nq}:  '
              f'max|g2_cond_cart|={np.max(np.abs(g2_cond_cart[iq])):.3e}  '
              f'max|g2_val_cart|={np.max(np.abs(g2_val_cart[iq])):.3e}  Ry/bohr^2')

    # ── 5. Save ───────────────────────────────────────────────────────────────
    print(f'\nSaving to {args.out} ...')
    kw = dict(compression='gzip', compression_opts=4)
    with h5py.File(args.out, 'w') as out:
        ds = out.create_dataset('elph_fine_cond_mode', data=g2_cond_mode, **kw)
        ds.attrs['axes']  = 'elph_fine_cond_mode[iq, nu, ik_fi, ic_fi, ic_fi_prime]'
        ds.attrs['units'] = 'Ry/bohr^2'
        ds.attrs['note']  = 'Second-order e-ph, phonon-mode basis, ic=0 → LUMO'

        ds = out.create_dataset('elph_fine_val_mode', data=g2_val_mode, **kw)
        ds.attrs['axes']  = 'elph_fine_val_mode[iq, nu, ik_fi, iv_fi, iv_fi_prime]'
        ds.attrs['units'] = 'Ry/bohr^2'
        ds.attrs['note']  = 'Second-order e-ph, phonon-mode basis, iv=0 → HOMO'

        ds = out.create_dataset('elph_fine_cond_cart', data=g2_cond_cart, **kw)
        ds.attrs['axes']  = 'elph_fine_cond_cart[iq, alpha, ik_fi, ic_fi, ic_fi_prime]'
        ds.attrs['units'] = 'Ry/bohr^2'
        ds.attrs['note']  = 'Second-order e-ph, Cartesian basis, alpha=3*iatom+idir'

        ds = out.create_dataset('elph_fine_val_cart', data=g2_val_cart, **kw)
        ds.attrs['axes']  = 'elph_fine_val_cart[iq, alpha, ik_fi, iv_fi, iv_fi_prime]'
        ds.attrs['units'] = 'Ry/bohr^2'
        ds.attrs['note']  = 'Second-order e-ph, Cartesian basis, alpha=3*iatom+idir'

        out.create_dataset('Kpoints_in_elph_file', data=Kpoints, **kw)
        out['Kpoints_in_elph_file'].attrs['units'] = 'crystal (fractional) coordinates'

        # Copy phonon_modes group and q-point datasets from input
        with h5py.File(args.elph_fine, 'r') as src:
            src.copy('phonon_modes', out, name='phonon_modes')
            for name in ('qpoints_crystal', 'qpoints_cart'):
                if name in src:
                    src.copy(name, out, name=name)

        out.attrs['Nq']              = Nq
        out.attrs['Nmodes']          = Nmodes
        out.attrs['Npert']           = Npert
        out.attrs['Nk_fi']           = Nk
        out.attrs['Nc_fi']           = Nc
        out.attrs['Nv_fi']           = Nv
        out.attrs['Nval']            = args.Nval
        out.attrs['note']            = ('Second-order el-ph coefficients (Ry/bohr^2), '
                                        'same format as elph_fine.h5')
        out.attrs['source_elph_fine'] = args.elph_fine
        out.attrs['source_eqp']       = args.eqp

    print(f'Done.  Output: {args.out}')
    print(f'  g2_cond_cart shape : {g2_cond_cart.shape}')
    print(f'  g2_val_cart  shape : {g2_val_cart.shape}')
    print(f'  g2_cond_mode shape : {g2_cond_mode.shape}')
    print(f'  g2_val_mode  shape : {g2_val_mode.shape}')
