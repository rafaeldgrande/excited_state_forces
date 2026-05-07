"""

Usage:
python interpolate_elph_bgw.py --elph_coarse elph.h5 --dtmat dtmat --Nval 13

interpolate_elph_bgw.py
=======================
Interpolate electron-phonon (el-ph) matrix elements from a coarse to a fine
k-grid using the BerkeleyGW coarse-to-fine overlap matrices stored in dtmat.

Interpolation formula (one q at a time, fixed q)
-------------------------------------------------
  <n, k_fi+q | dV(q) | m, k_fi>
      = sum_{a,b} <n, k_fi+q | a, k_co+q>
                * <a, k_co+q | dV(q) | b, k_co>
                * <b, k_co | m, k_fi>

where:
  n, m  — fine-grid band indices (conduction or valence)
  a, b  — coarse-grid band indices (same sector)
  k_co  — nearest coarse k-point to k_fi       (from fi2co_wfn)
  k_co+q— nearest coarse k-point to k_fi+q

The overlap matrices <n,k_fi|a,k_co> are dcn (conduction) and dvn (valence)
from the dtmat file (read via bgw_binary_io.read_dtmat).

BGW valence-band ordering convention
-------------------------------------
In BerkeleyGW, valence bands are indexed from the Fermi level *downward*:
  BGW v=0  →  QE band Nval-1  (HOMO, 0-indexed)
  BGW v=1  →  QE band Nval-2  (HOMO-1)
  ...
This means the valence block of g_mode must be reversed along both band axes
before applying dvn.  The output elph_fine_val uses the same BGW ordering
(v=0 = HOMO), which is what the rest of the excited_forces code expects.

Inputs
------
elph_h5_file : str
    Path to elph_coarse.h5 (produced by assemble_elph_h5.py).
    Expected datasets:
      g_mode  (Nq, Nmodes, Nk_co, Nbnds, Nbnds)  — phonon-mode basis
      g       (Nq, Npert,  Nk_co, Nbnds, Nbnds)  — Cartesian basis
      kpoints_dft_crystal  (Nk_co, 3)
      qpoints_crystal      (Nq,    3)

dtmat_file : str
    Path to the BerkeleyGW dtmat binary.

Nval : int
    Number of occupied (valence) bands in the DFPT calculation (QE nbnd
    convention, counting from band 1).  Determines which rows/cols of
    g_mode are conduction vs. valence.
    Constraint: n1b_co <= Nval  and  Nval + n2b_co <= Nbnds_elph.

kpts_fi_crys : (Nk_fi, 3) array, optional
    Fine k-point coordinates in crystal (fractional) coordinates.
    Required only for finite-q interpolation (q != 0).
    Can be read from WFN_fi.h5 → mf_header/kpoints/rk.

dataset : str, optional  {'g_mode', 'g'}
    Which el-ph dataset from elph_h5_file to interpolate.
    Default: 'g_mode'  (phonon-mode basis, shape Nq×Nmodes×Nk×Nb×Nb).
    Use 'g' for the Cartesian-displacement basis.

tol_k : float, optional
    Tolerance for k-point matching in crystal coordinates (default 1e-5).

Returns
-------
elph_fine_cond : (Nq, Nmodes, Nk_fi, Nc_fi, Nc_fi)  complex128
    Conduction-conduction block on the fine grid.
    Band ordering: ic=0 → LUMO, ic=1 → LUMO+1, ...

elph_fine_val  : (Nq, Nmodes, Nk_fi, Nv_fi, Nv_fi)  complex128
    Valence-valence block on the fine grid.
    Band ordering: iv=0 → HOMO, iv=1 → HOMO-1, ...  (BGW convention)

Notes
-----
* For npts_intp_kernel > 1 (multi-vertex interpolation), the contributions
  from each vertex are weighted by intp_coefs and summed.
* When band counts in elph_h5_file and dtmat disagree, a warning is printed
  and the available (smaller) set of bands is used.
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import h5py

# ── locate bgw_binary_io relative to this file ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bgw_binary_io import read_dtmat


# ─────────────────────────────────────────────────────────────────────────────
# k-point utilities
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_bz(k: np.ndarray) -> np.ndarray:
    """Wrap crystal coordinates to [0, 1)."""
    return k - np.floor(k + 1e-10)


def _build_kpt_map(kpts_query: np.ndarray,
                   kpts_ref: np.ndarray,
                   tol: float = 1e-5) -> np.ndarray:
    """
    For each row in kpts_query find the matching row index in kpts_ref
    (modulo reciprocal lattice vectors).  Returns array of ints; -1 = not found.
    """
    nq = len(kpts_query)
    idx_map = np.full(nq, -1, dtype=int)
    kref_w = _wrap_bz(kpts_ref)               # (Nref, 3)
    for iq, kq in enumerate(kpts_query):
        kq_w = _wrap_bz(kq)
        diff = kref_w - kq_w                  # (Nref, 3)
        diff -= np.round(diff)                 # fold to [-0.5, 0.5)
        dists = np.linalg.norm(diff, axis=1)
        best = int(np.argmin(dists))
        if dists[best] < tol:
            idx_map[iq] = best
    return idx_map


# ─────────────────────────────────────────────────────────────────────────────
# Main interpolation function
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_elph(
    elph_h5_file: str,
    dtmat_file:   str,
    Nval:         int,
    kpts_fi_crys: np.ndarray | None = None,
    dataset:      str  = 'g_mode',
    tol_k:        float = 1e-5,
    complex_flavor: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate el-ph matrix elements from coarse to fine k-grid.

    See module docstring for full description.
    """

    # ── 1. Load coarse el-ph ─────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"Loading coarse el-ph from: {os.path.basename(elph_h5_file)}")
    print(f"{'='*64}")

    with h5py.File(elph_h5_file, 'r') as fh:
        if dataset not in fh:
            raise KeyError(
                f"Dataset '{dataset}' not found in {elph_h5_file}. "
                f"Available: {list(fh.keys())}")
        g_co_raw = fh[dataset][:]              # (Nq, Nmodes, Nk_co, Nbnds, Nbnds)
        kpts_co  = fh['kpoints_dft_crystal'][:] # (Nk_co, 3)
        qpts_co  = fh['qpoints_crystal'][:]     # (Nq, 3)

    Nq, Nmodes, Nk_co, Nbnds, _ = g_co_raw.shape
    print(f"  g_co shape   : {g_co_raw.shape}  ({Nq} q, {Nmodes} modes, "
          f"{Nk_co} k_co, {Nbnds} bands)")
    print(f"  dtype        : {g_co_raw.dtype}")
    print(f"  Nval         : {Nval}")

    # ── 2. Load dtmat ────────────────────────────────────────────────────────
    print(f"\nLoading dtmat from: {os.path.basename(dtmat_file)}")
    d = read_dtmat(dtmat_file, complex_flavor=complex_flavor)

    nk_co_dt  = d['nkpt_co']
    nk_fi     = d['nkpt_fi']
    nc_fi     = d['ncb_fi']
    nv_fi     = d['nvb_fi']
    nc_co     = d['n2b_co']
    nv_co     = d['n1b_co']
    nspin     = d['nspin']
    npts      = d['npts_intp_kernel']
    dcn       = d['dcn']          # (nc_fi, nc_co, nspin, nk_fi, npts)
    dvn       = d['dvn']          # (nv_fi, nv_co, nspin, nk_fi, npts)
    fi2co     = d['fi2co_wfn']    # (npts, nk_fi) — 1-indexed coarse k
    intp_coefs= d['intp_coefs']   # (npts, nk_fi)
    kco_dt    = d['kco']          # (nk_co_dt, 3) — coarse k in crystal

    print(f"  nkpt_co      : {nk_co_dt}    nkpt_fi  : {nk_fi}")
    print(f"  nc_co        : {nc_co}       nc_fi    : {nc_fi}")
    print(f"  nv_co        : {nv_co}       nv_fi    : {nv_fi}")
    print(f"  nspin        : {nspin}       npts_intp: {npts}")

    # ── 3. Consistency checks / warnings ────────────────────────────────────
    # 3a. Number of coarse k-points
    if nk_co_dt != Nk_co:
        warnings.warn(
            f"Coarse k-point count mismatch: elph has {Nk_co}, dtmat has {nk_co_dt}. "
            f"They may come from different runs.  Will match k-points by coordinates.")
        _build_kco_map = True
    else:
        _build_kco_map = False

    # 3b. Band ranges
    nc_co_avail = Nbnds - Nval          # cond bands available in g_mode
    nv_co_avail = Nval                  # val bands available in g_mode

    if nc_co_avail < nc_co:
        warnings.warn(
            f"dtmat requests {nc_co} coarse conduction bands, but g_mode only has "
            f"{nc_co_avail} (Nbnds={Nbnds}, Nval={Nval}).  "
            f"Using {nc_co_avail} cond bands.")
    if nv_co_avail < nv_co:
        warnings.warn(
            f"dtmat requests {nv_co} coarse valence bands, but g_mode only has "
            f"{nv_co_avail} (Nval={Nval}).  "
            f"Using {nv_co_avail} val bands.")

    nc_co_use = min(nc_co, nc_co_avail)
    nv_co_use = min(nv_co, nv_co_avail)

    # 3c. nspin warning (code handles spin index 0 only for now)
    if nspin > 1:
        warnings.warn(
            f"nspin={nspin} in dtmat — current implementation uses spin=0 only.")

    # ── 4. Build coarse k-point map (elph → dtmat) ──────────────────────────
    #       The elph g_mode uses indices [0, Nk_co) in the order from scf.in.
    #       dtmat stores kco in its own order. We need to map between them.
    elph_to_dt_co = _build_kpt_map(kpts_co, kco_dt, tol=tol_k)
    n_matched = np.sum(elph_to_dt_co >= 0)
    if n_matched < Nk_co:
        warnings.warn(
            f"Only {n_matched}/{Nk_co} coarse k-points matched between "
            f"elph_h5 and dtmat.  Unmatched k-points will be skipped (g_fine=0).")
    print(f"\n  Coarse k-point matching: {n_matched}/{Nk_co} matched")

    # ── 5. Extract and reorder coarse-grid cond/val blocks ──────────────────
    #
    #  Conduction (same ordering as QE, counting from LUMO):
    #    g_co_cond[iq, nu, ik_co, a, b] = g_co_raw[iq, nu, ik_co, Nval+a, Nval+b]
    #    a = 0 → LUMO
    #
    #  Valence (BGW convention: counting from HOMO downward):
    #    g_co_val[iq, nu, ik_co, a, b] = g_co_raw[iq, nu, ik_co, Nval-1-a, Nval-1-b]
    #    a = 0 → HOMO,  a = 1 → HOMO-1, ...
    #    Implemented by taking the natural QE slice and flipping both axes.

    # conduction
    c0 = Nval
    c1 = Nval + nc_co_use
    g_co_cond = g_co_raw[:, :, :, c0:c1, c0:c1]          # (Nq,Nm,Nk_co,nc_co_use,nc_co_use)
    if nc_co_use < nc_co:
        # Pad with zeros so matrix dims match dcn
        pad = np.zeros((*g_co_cond.shape[:3], nc_co, nc_co), dtype=g_co_cond.dtype)
        pad[..., :nc_co_use, :nc_co_use] = g_co_cond
        g_co_cond = pad

    # valence — extract and flip
    v0 = Nval - nv_co_use
    v1 = Nval
    _g_val_qe = g_co_raw[:, :, :, v0:v1, v0:v1]           # QE order (lowest val first)
    # Flip to BGW order: [::-1, ::-1] over the last two axes
    g_co_val = _g_val_qe[..., ::-1, ::-1]                 # (Nq,Nm,Nk_co,nv_co_use,nv_co_use)
    if nv_co_use < nv_co:
        pad = np.zeros((*g_co_val.shape[:3], nv_co, nv_co), dtype=g_co_val.dtype)
        pad[..., :nv_co_use, :nv_co_use] = g_co_val
        g_co_val = pad

    print(f"\n  Coarse block shapes after extraction:")
    print(f"    g_co_cond : {g_co_cond.shape}  (Nq, Nmodes, Nk_co, nc_co, nc_co)")
    print(f"    g_co_val  : {g_co_val.shape}  (Nq, Nmodes, Nk_co, nv_co, nv_co)")

    # ── 6. Handle finite-q: build fine k-point map k_fi → k_fi+q ───────────
    #
    # For q=0 (Gamma): the same fine k-point index is used for left and right
    #   factors, so ik_fi_q = ik_fi for all ik_fi.
    #
    # For finite q: for each fine k-point k_fi we need the fine k-point index
    #   of k_fi + q (mod lattice).  This requires kpts_fi_crys.
    #   Note: we use the SAME g_co[ik_co] for both sides (where ik_co comes from
    #   fi2co_wfn for the right/ket side), consistent with the BGW convention
    #   that k_co is the coarse representative for k_fi.

    ik_fi_q_map = np.arange(nk_fi, dtype=int)   # default: q=0, no shift

    _any_finite_q = not np.all(np.abs(qpts_co) < tol_k)

    if _any_finite_q:
        if kpts_fi_crys is None:
            raise ValueError(
                "kpts_fi_crys (fine k-point coordinates) is required for "
                "finite-q interpolation.  Read it from WFN_fi.h5 → "
                "mf_header/kpoints/rk, then pass it as kpts_fi_crys.")
        # We build a q-dependent shift map for EACH q-point.
        # For now we store it per iq; the loop below handles it.
        print(f"\n  Finite-q detected — will build k_fi → k_fi+q maps per q.")
    else:
        print(f"\n  q = Gamma (all q-points are 0,0,0) — no k-shift needed.")

    # ── 7. Precompute dtmat-coarse → elph-coarse index map ───────────────────
    #       fi2co[ivert, ik_fi] is a 1-indexed index into dtmat's kco ordering.
    #       We need to map each dtmat coarse index to the elph g_mode coarse index.
    #       Build this lookup once for all unique coarse k-points in dtmat.
    dt_to_elph_co = np.full(nk_co_dt, -1, dtype=int)
    for ik_co_dt in range(nk_co_dt):
        match = _build_kpt_map(kco_dt[ik_co_dt][None, :], kpts_co, tol=tol_k)
        dt_to_elph_co[ik_co_dt] = match[0]
    n_co_matched = np.sum(dt_to_elph_co >= 0)
    if n_co_matched < nk_co_dt:
        warnings.warn(
            f"{nk_co_dt - n_co_matched}/{nk_co_dt} dtmat coarse k-points have no "
            f"match in elph_h5 coarse grid. Their contributions will be skipped.")
    print(f"  dtmat→elph coarse k-map: {n_co_matched}/{nk_co_dt} matched")

    # Precompute per fine k-point: for each ivert the elph coarse k index
    # Shape: (npts, nk_fi), value = elph coarse index (-1 if not matched)
    fi2co_elph = np.full((npts, nk_fi), -1, dtype=int)
    for ivert in range(npts):
        for ik_fi in range(nk_fi):
            ik_co_dt = int(fi2co[ivert, ik_fi]) - 1   # 0-indexed in dtmat
            fi2co_elph[ivert, ik_fi] = dt_to_elph_co[ik_co_dt]

    # ── 8. Allocate output arrays ────────────────────────────────────────────
    elph_fine_cond = np.zeros((Nq, Nmodes, nk_fi, nc_fi, nc_fi), dtype=np.complex128)
    elph_fine_val  = np.zeros((Nq, Nmodes, nk_fi, nv_fi, nv_fi), dtype=np.complex128)

    print(f"\n  Output shapes:")
    print(f"    elph_fine_cond : {elph_fine_cond.shape}  (Nq, Nmodes, Nk_fi, nc_fi, nc_fi)")
    print(f"    elph_fine_val  : {elph_fine_val.shape}  (Nq, Nmodes, Nk_fi, nv_fi, nv_fi)")

    # ── 9. Main interpolation loop ───────────────────────────────────────────
    print(f"\nInterpolating ... ({nk_fi} fine k-points, {Nq} q-points, "
          f"{Nmodes} modes)")

    for iq in range(Nq):

        # Build k_fi → k_fi+q map for this q
        if _any_finite_q:
            q_crys = qpts_co[iq]                                     # (3,)
            k_fi_q = _wrap_bz(kpts_fi_crys + q_crys[None, :])       # (Nk_fi, 3)
            ik_fi_q_map = _build_kpt_map(k_fi_q, kpts_fi_crys, tol=tol_k)
            n_q_matched = np.sum(ik_fi_q_map >= 0)
            if n_q_matched < nk_fi:
                warnings.warn(
                    f"q-point iq={iq}: only {n_q_matched}/{nk_fi} fine k+q "
                    f"points matched in fine grid. Missing ones skipped.")
        else:
            ik_fi_q_map = np.arange(nk_fi, dtype=int)

        for ik_fi in range(nk_fi):

            ik_fi_q = ik_fi_q_map[ik_fi]
            if ik_fi_q < 0:
                continue   # k_fi+q not found in fine grid; leave g_fine=0

            # Accumulate weighted coarse-grid blocks over interpolation vertices
            # g_co_X_block[nu, a, b] = sum_ivert w[ivert] * g_co_X[iq, nu, ik_co, a, b]
            g_co_cond_block = np.zeros((Nmodes, nc_co, nc_co), dtype=np.complex128)
            g_co_val_block  = np.zeros((Nmodes, nv_co, nv_co), dtype=np.complex128)

            for ivert in range(npts):
                w          = intp_coefs[ivert, ik_fi]     # interpolation weight
                ik_co_elph = fi2co_elph[ivert, ik_fi]     # elph coarse k index
                if ik_co_elph < 0:
                    continue   # coarse k-point not matched; skip vertex
                g_co_cond_block += w * g_co_cond[iq, :, ik_co_elph, :, :]
                g_co_val_block  += w * g_co_val[ iq, :, ik_co_elph, :, :]

            # Wavefunction overlap factors (spin index 0; npts_intp_kernel axis 0)
            # D_right[m, b] = <m, k_fi   | b, k_co>   from dcn/dvn at ik_fi
            # D_left [n, a] = <n, k_fi+q | a, k_co+q> from dcn/dvn at ik_fi_q
            # For npts_intp_kernel > 1 we use ivert=0 for the wavefunction factors
            # (BGW convention: intp_coefs weight only the coarse matrix elements,
            #  not the wavefunction overlaps themselves).
            D_right_cond = dcn[:, :, 0, ik_fi,   0]   # (nc_fi, nc_co)
            D_left_cond  = dcn[:, :, 0, ik_fi_q, 0]   # (nc_fi, nc_co) at k_fi+q
            D_right_val  = dvn[:, :, 0, ik_fi,   0]   # (nv_fi, nv_co)
            D_left_val   = dvn[:, :, 0, ik_fi_q, 0]   # (nv_fi, nv_co) at k_fi+q

            # Interpolation:
            #   G_fine[n, m] = sum_{a,b} D_left[n,a]^* * G_co[a,b] * D_right[m,b]
            # Vectorised over the modes axis (v) via einsum 'na, vab, mb -> vnm'
            elph_fine_cond[iq, :, ik_fi, :, :] = np.einsum(
                'na, vab, mb -> vnm',
                np.conj(D_left_cond),
                g_co_cond_block,
                D_right_cond,
                optimize=True,
            )
            elph_fine_val[iq, :, ik_fi, :, :] = np.einsum(
                'na, vab, mb -> vnm',
                np.conj(D_left_val),
                g_co_val_block,
                D_right_val,
                optimize=True,
            )

    # ── 9. Summary ───────────────────────────────────────────────────────────
    print(f"\nDone.")
    print(f"  elph_fine_cond : {elph_fine_cond.shape}")
    print(f"    max|Re| = {np.max(np.abs(np.real(elph_fine_cond))):.4e}")
    print(f"    max|Im| = {np.max(np.abs(np.imag(elph_fine_cond))):.4e}")
    print(f"  elph_fine_val  : {elph_fine_val.shape}")
    print(f"    max|Re| = {np.max(np.abs(np.real(elph_fine_val))):.4e}")
    print(f"    max|Im| = {np.max(np.abs(np.imag(elph_fine_val))):.4e}")

    return elph_fine_cond, elph_fine_val


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: save to HDF5
# ─────────────────────────────────────────────────────────────────────────────

def save_elph_fine(
    elph_fine_cond: np.ndarray,
    elph_fine_val:  np.ndarray,
    out_path:       str,
    kpts_fi_crys:   np.ndarray | None = None,
    compress:       bool = True,
    extra_attrs:    dict | None = None,
) -> None:
    """
    Save interpolated fine-grid el-ph arrays to an HDF5 file.

    Parameters
    ----------
    elph_fine_cond : (Nq, Nmodes, Nk_fi, Nc_fi, Nc_fi)
    elph_fine_val  : (Nq, Nmodes, Nk_fi, Nv_fi, Nv_fi)
    out_path       : output .h5 filename
    kpts_fi_crys   : (Nk_fi, 3) fine k-points in crystal (fractional) coords
    compress       : whether to apply gzip compression
    extra_attrs    : optional dict of additional root-level attributes
    """
    kw = dict(compression='gzip', compression_opts=4) if compress else {}

    with h5py.File(out_path, 'w') as fh:
        ds = fh.create_dataset('elph_fine_cond', data=elph_fine_cond, **kw)
        ds.attrs['axes']  = 'elph_fine_cond[iq, nu, ik_fi, ic_fi, ic_fi_prime]'
        ds.attrs['note']  = ('<c_fi, k_fi+q | dV(q) | c_fi_prime, k_fi>, '
                             'ic=0 -> LUMO (BGW convention)')

        ds = fh.create_dataset('elph_fine_val', data=elph_fine_val, **kw)
        ds.attrs['axes']  = 'elph_fine_val[iq, nu, ik_fi, iv_fi, iv_fi_prime]'
        ds.attrs['note']  = ('<v_fi, k_fi+q | dV(q) | v_fi_prime, k_fi>, '
                             'iv=0 -> HOMO (BGW convention, counting from Fermi level down)')

        if kpts_fi_crys is not None:
            ds = fh.create_dataset('Kpoints_in_elph_file', data=kpts_fi_crys, **kw)
            ds.attrs['axes'] = 'Kpoints_in_elph_file[ik_fi, xyz]'
            ds.attrs['units'] = 'crystal (fractional) coordinates'

        fh.attrs['Nq']     = elph_fine_cond.shape[0]
        fh.attrs['Nmodes'] = elph_fine_cond.shape[1]
        fh.attrs['Nk_fi']  = elph_fine_cond.shape[2]
        fh.attrs['Nc_fi']  = elph_fine_cond.shape[3]
        fh.attrs['Nv_fi']  = elph_fine_val.shape[3]
        fh.attrs['interpolation'] = 'BerkeleyGW dtmat coarse-to-fine'
        if extra_attrs:
            for k, v in extra_attrs.items():
                fh.attrs[k] = v

    print(f"\nSaved fine-grid el-ph to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI / example usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    cli = argparse.ArgumentParser(
        description='Interpolate el-ph from coarse to fine k-grid using BGW dtmat.')
    cli.add_argument('--elph_coarse', required=True, help='Path to elph_coarse.h5')
    cli.add_argument('--dtmat',       required=True, help='Path to dtmat binary')
    cli.add_argument('--Nval',        required=True, type=int,
                     help='Number of valence bands in DFPT (QE nbnd convention)')
    cli.add_argument('--out',    default='elph_fine.h5',
                     help='Output HDF5 filename (default: elph_fine.h5)')
    cli.add_argument('--dataset', default='g_mode',
                     choices=['g_mode', 'g'],
                     help='Which el-ph dataset to interpolate (default: g_mode)')
    cli.add_argument('--wfn-fi', default=None,
                     help='Path to WFN_fi.h5 (needed for finite-q interpolation)')
    cli.add_argument('--real',   action='store_true',
                     help='Use real-flavor dtmat (default: complex)')
    args = cli.parse_args()

    kpts_fi = None
    wfn_fi_path = args.wfn_fi
    if wfn_fi_path is None:
        # auto-discover WFN_fi.h5 next to the dtmat file
        _dtmat_dir = os.path.dirname(os.path.abspath(args.dtmat))
        _candidate = os.path.join(_dtmat_dir, 'WFN_fi.h5')
        if os.path.isfile(_candidate):
            wfn_fi_path = _candidate
            print(f"Auto-discovered WFN_fi.h5: {wfn_fi_path}")
        else:
            print(f"WARNING: --wfn-fi not given and WFN_fi.h5 not found next to dtmat. "
                  f"'Kpoints_in_elph_file' will NOT be saved in the output.")
    if wfn_fi_path:
        print(f"Reading fine k-points from {wfn_fi_path} ...")
        with h5py.File(wfn_fi_path, 'r') as fh:
            rk = fh['mf_header/kpoints/rk'][:]  # h5py reads Fortran rk(3,nrk) as (nrk, 3)
            kpts_fi = rk if rk.shape[-1] == 3 else rk.T   # ensure (nrk, 3)
        print(f"  {len(kpts_fi)} fine k-points loaded, shape {kpts_fi.shape}")

    elph_cond, elph_val = interpolate_elph(
        elph_h5_file  = args.elph_coarse,
        dtmat_file    = args.dtmat,
        Nval          = args.Nval,
        kpts_fi_crys  = kpts_fi,
        dataset       = args.dataset,
        complex_flavor= not args.real,
    )

    save_elph_fine(
        elph_cond, elph_val,
        out_path      = args.out,
        kpts_fi_crys  = kpts_fi,
        extra_attrs   = {'Nval': args.Nval, 'source_elph': args.elph_coarse,
                         'source_dtmat': args.dtmat},
    )
