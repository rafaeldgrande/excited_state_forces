"""
elph_xml_to_h5_elephany.py
===========================
Assembles DFT-level, finite-difference electron-phonon (el-ph) matrix
elements computed by ElePhAny.jl into the same elph.h5 schema produced by
elph_xml_to_h5_QE.py (the DFPT-based assembler), so nothing downstream
(excited_forces.py, elph_coeffs_second_derivative.py,
susceptibility_tensors_*.py) needs to know or care which method produced
the file.

Companion to elph_xml_to_h5_QE.py -- see that file's module docstring for
the full elph.h5 schema description. This script only implements the
"assembly" stage (no coarse-to-fine dtmat interpolation): ElePhAny.jl is
run directly on the fine k-grid the BSE calculation uses (sc_size=k_mesh
matches the desired dense grid, e.g. [6,6,1] for a 36x36x1 fine grid), and
all sc_size-commensurate q-points, so no interpolation is needed or
performed -- this mirrors elph_xml_to_h5_QE.py's DEFAULT (non
--interpolate_elph_coeffs) mode.

Input: a raw JLD2 dump produced by a dump_elph_elephany*.jl driver script
(e.g. dump_elph_elephany.jl / dump_elph_elephany_sc6.jl), containing:
  elph_cart   : (Nq, 3*Nat, Nk, nbands, nbands) complex128, Ry/Angstrom,
                Cartesian basis, alpha = 3*(iat-1)+icart (1-indexed Julia)
  omega_arr   : (Nq, Nmodes) float64, cm-1 (ElePhAny/phonopy's own
                pwscf_to_cm1 --factor already applied -- confirmed via
                source read, no further unit conversion needed)
  eps_arr     : (Nq, Nmodes, 3*Nat) complex128, phonopy's MASS-WEIGHTED,
                Euclidean-unit-normalized eigenvector convention (confirmed
                via source read of ElePhAny.jl's renorm_mass_eps -- the
                DFPT/matdyn.x comparison path converts ITS raw real-space
                eigenvectors *into* this same phonopy convention via
                renorm_mass_eps, and the default (non-DFPT-comparison)
                path uses eps_arr_list.jld2 unmodified -- so eps_arr IS
                already in the phonopy mass-weighted convention, and must
                be converted BACK to the real-space, non-mass-weighted,
                unit-normalized convention this script's target schema
                uses (matching parse_matdyn_modes' matdyn.x-sourced
                eigenvectors) by dividing by sqrt(mass) per atom and
                renormalizing.
  m_arr       : (Nat,) float64, atomic masses (amu)
  kpoints_mat : (Nk, 3) float64, crystal (fractional) coordinates
  qpoints_mat : (Nq, 3) float64, crystal (fractional) coordinates
  Nk, Nq, Nat, Nmodes, nbands : int

Units
-----
ElePhAny.jl's abs_disp is documented (phonons.jl) and used (via
phonopy.generate_displacements(distance=abs_disp)) in ANGSTROM. Combined
with electron_phonon()'s own `scale = ev_to_ry / abs_disp` factor (applied
before returning braket_list / braket_list_rotated), the raw elph_cart
values are in Ry/Angstrom, NOT Ry/bohr -- confirmed via direct source read
of src/electron_phonons.jl, NOT assumed. elph_xml_to_h5_QE.py's own
elph_cond_cart/elph_val_cart datasets are in Ry/bohr (confirmed via
read_elph_xml's docstring: g_mat is "[Ry/bohr]", parsed straight from QE's
elph*.xml). This script therefore multiplies elph_cart by BOHR_TO_ANGSTROM
(the bohr radius in Angstrom, imported from elph_xml_to_h5_QE) to convert
Ry/Angstrom -> Ry/bohr before writing.

Usage
-----
python elph_xml_to_h5_elephany.py --elph_raw elph_elephany_raw.jld2 \\
    --wfn_ref WFN_fi.h5 --eqp eqp.dat --elph_out elph.h5

# No eqp.dat available yet -- write without band-matching/QP rescaling
python elph_xml_to_h5_elephany.py --elph_raw elph_elephany_raw.jld2 \\
    --wfn_ref WFN_fi.h5 --elph_out elph.h5
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # elph_xml_to_h5_QE (same dir)
from elph_xml_to_h5_QE import (
    BOHR_TO_ANGSTROM,
    RY_TO_EV,
    apply_acoustic_sum_rule,
    split_cond_val,
    save_elph_h5,
    read_wfn_h5_header,
    _read_eqp_and_build_rescaling,
    _get_band_window_from_eqp,
    _truncate_or_pad_bands,
)


def load_elephany_raw(path: str) -> dict:
    """
    Load a dump_elph_elephany*.jl JLD2 output via h5py. JLD2 is HDF5-based,
    but Julia is column-major while h5py/numpy are row-major, so array axes
    come out REVERSED relative to the logical Julia shape (confirmed
    empirically against a real dump: Julia (9,16,4,35,35) -> h5py
    (35,35,4,16,9)). Complex numbers are stored as a compound dtype with
    're'/'im' fields, also confirmed empirically, not assumed.
    """
    with h5py.File(path, 'r') as f:
        raw = f['elph_cart'][:]                       # h5py shape (nb, nb, Nq, Nk, 3Nat)
        elph_cart = raw['re'] + 1j * raw['im']
        elph_cart = np.transpose(elph_cart, (2, 4, 3, 0, 1))  # -> (Nq, 3Nat, Nk, nb, nb)

        raw_eps = f['eps_arr'][:]                      # h5py shape (3Nat, Nmodes, Nq)
        eps_arr = raw_eps['re'] + 1j * raw_eps['im']
        eps_arr = np.transpose(eps_arr, (2, 1, 0))      # -> (Nq, Nmodes, 3Nat)

        omega_arr = f['omega_arr'][:].T                 # h5py (Nmodes, Nq) -> (Nq, Nmodes)
        m_arr = f['m_arr'][:]
        kpoints_mat = f['kpoints_mat'][:].T              # h5py (3, Nk) -> (Nk, 3)
        qpoints_mat = f['qpoints_mat'][:].T              # h5py (3, Nq) -> (Nq, 3)
        Nk = int(f['Nk'][()])
        Nq = int(f['Nq'][()])
        Nat = int(f['Nat'][()])
        Nmodes = int(f['Nmodes'][()])
        nbands = int(f['nbands'][()])

    assert elph_cart.shape == (Nq, 3 * Nat, Nk, nbands, nbands), (
        f"elph_cart shape {elph_cart.shape} != expected "
        f"({Nq}, {3*Nat}, {Nk}, {nbands}, {nbands})")
    assert eps_arr.shape == (Nq, Nmodes, 3 * Nat)
    assert omega_arr.shape == (Nq, Nmodes)
    assert m_arr.shape == (Nat,)

    return dict(elph_cart=elph_cart, eps_arr=eps_arr, omega_arr=omega_arr,
                m_arr=m_arr, kpoints_mat=kpoints_mat, qpoints_mat=qpoints_mat,
                Nk=Nk, Nq=Nq, Nat=Nat, Nmodes=Nmodes, nbands=nbands)


def massweighted_to_realspace_eigenvectors(eps_massweighted: np.ndarray,
                                            m_arr: np.ndarray) -> np.ndarray:
    """
    Convert ElePhAny/phonopy's mass-weighted, Euclidean-unit-normalized
    eigenvectors into the real-space (Cartesian displacement pattern),
    unit-normalized convention that matdyn.x prints and this project's
    elph.h5 schema documents (see save_elph_h5's 'eigenvectors' dataset
    note). This is the exact inverse of ElePhAny.jl's own renorm_mass_eps
    (phonons.jl): that function multiplies QE's real-space eigenvectors by
    sqrt(mass) per atom then renormalizes to get to the phonopy convention;
    here we divide by sqrt(mass) then renormalize to undo it.

    eps_massweighted : (Nq, Nmodes, 3*Nat) complex128, packed
                        alpha = icart + 3*(iat-1) (0-indexed here)
    m_arr             : (Nat,) float64, amu

    Returns (Nq, Nmodes, Nat, 3) complex128, unit-normalized real-space
    displacement patterns.
    """
    Nq, Nmodes, npert = eps_massweighted.shape
    Nat = len(m_arr)
    assert npert == 3 * Nat

    eps = eps_massweighted.reshape(Nq, Nmodes, Nat, 3)
    w = np.sqrt(m_arr)[None, None, :, None]              # (1,1,Nat,1)
    eps_real = eps / w

    norms = np.sqrt(np.sum(np.abs(eps_real) ** 2, axis=(-2, -1), keepdims=True))
    norms = np.where(norms < 1e-30, 1.0, norms)
    eps_real /= norms

    return eps_real


def main():
    parser = argparse.ArgumentParser(
        description="Assemble ElePhAny.jl (finite-difference DFT) electron-phonon "
                     "matrix elements into the same elph.h5 schema used by "
                     "elph_xml_to_h5_QE.py (DFPT-based assembler).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--elph_raw', required=True,
                         help='JLD2 file from a dump_elph_elephany*.jl driver script.')
    parser.add_argument('--wfn_ref', required=True,
                         help='BerkeleyGW WFN.h5 built (via pw2bgw.x) from the SAME '
                              'scf_0 SCF calculation used by ElePhAny.jl -- required for '
                              'gauge consistency between elph_cart and the exciton-side '
                              'wavefunctions (see NOTES.md 2026-08-14 entry). Supplies '
                              'crystal structure, reciprocal lattice, and Nval (via ifmax).')
    parser.add_argument('--eqp', default=None,
                         help='Fine-grid eqp.dat (BerkeleyGW inteqp output) for band-matching '
                              'and QP rescaling. If omitted, el-ph is written without QP '
                              'rescaling and without band-window truncation.')
    parser.add_argument('--Nval', type=int, default=None,
                         help='Override the number of occupied bands (default: read from '
                              '--wfn_ref via ifmax).')
    parser.add_argument('--elph_out', default='elph.h5',
                         help='Output HDF5 path (default: elph.h5).')
    parser.add_argument('--no-ASR', dest='asr', action='store_false', default=True,
                         help='Skip the acoustic sum rule correction (default: apply it).')
    parser.add_argument('--save_not_filtered', default=None,
                         help='Also write the full (unfiltered, all available cond/val bands) '
                              'el-ph to this path, before any --eqp band-window truncation.')
    args = parser.parse_args()

    t0 = datetime.now()

    print(f"Loading raw ElePhAny.jl dump from {args.elph_raw} ...")
    raw = load_elephany_raw(args.elph_raw)
    Nq, Nk, Nat, Nmodes, nbands = (raw['Nq'], raw['Nk'], raw['Nat'],
                                    raw['Nmodes'], raw['nbands'])
    print(f"  Nq={Nq}  Nk={Nk}  Nat={Nat}  Nmodes={Nmodes}  nbands={nbands}")

    print(f"\nReading structure/Nval from {args.wfn_ref} ...")
    wfn_hdr = read_wfn_h5_header(args.wfn_ref, flag='--wfn_ref')
    Nval = args.Nval if args.Nval is not None else wfn_hdr['Nval']
    print(f"  Using Nval = {Nval}")

    structure = dict(
        atomic_numbers=wfn_hdr['atomic_numbers'],
        atomic_positions_ang=wfn_hdr['atomic_positions_ang'],
        lattice_vectors_ang=wfn_hdr['lattice_vectors_ang'],
        nat=wfn_hdr['nat'],
        source=os.path.basename(args.wfn_ref),
    )
    assert wfn_hdr['nat'] == Nat, (
        f"--wfn_ref has {wfn_hdr['nat']} atoms but --elph_raw has Nat={Nat}")

    # ── Units: Ry/Angstrom (ElePhAny) -> Ry/bohr (elph.h5 convention) ──
    print(f"\nConverting elph_cart: Ry/Angstrom -> Ry/bohr "
          f"(x{BOHR_TO_ANGSTROM:.10f}) ...")
    g_all = raw['elph_cart'] * BOHR_TO_ANGSTROM   # (Nq, 3Nat, Nk, nb, nb)

    # ── Acoustic sum rule ──
    if args.asr:
        print("\nApplying acoustic sum rule (default; pass --no-ASR to disable).")
        before, after = apply_acoustic_sum_rule(g_all, Nat)
        print(f"  max |sum_atoms g_d|  before ASR = {before:.4e} Ry/bohr")
        print(f"  max |sum_atoms g_d|  after  ASR = {after:.4e} Ry/bohr "
              f"({'OK' if after < 1e-10 else 'WARNING: residual is non-negligible'})")
    else:
        print("\nSkipping acoustic sum rule (--no-ASR set). g is left as-is.")

    # ── Phonon frequencies (already cm-1, confirmed via ElePhAny's own
    #    --factor pwscf_to_cm1 passed to phonopy) + eigenvectors (converted
    #    from phonopy's mass-weighted convention to matdyn.x's real-space,
    #    unit-normalized convention) ──
    print("\nConverting phonon eigenvectors: mass-weighted (phonopy) -> "
          "real-space unit-normalized (matdyn.x convention) ...")
    eigenvectors_realspace = massweighted_to_realspace_eigenvectors(
        raw['eps_arr'], raw['m_arr'])   # (Nq, Nmodes, Nat, 3)

    phonon_modes_dict = dict(
        qpoints=raw['qpoints_mat'],   # NOTE: crystal coords here, not Cartesian 2pi/a
        frequencies=raw['omega_arr'],
        eigenvectors=eigenvectors_realspace,
    )
    print(f"  Frequency range: {raw['omega_arr'].min():.2f} - "
          f"{raw['omega_arr'].max():.2f} cm-1")

    # ── g_mode: project Cartesian g onto phonon eigenvectors ──
    evec_flat = eigenvectors_realspace.reshape(Nq, Nmodes, 3 * Nat)
    g_mode = np.einsum('qva,qaknm->qvknm', evec_flat, g_all)
    print(f"  g_mode shape: {g_mode.shape}  (Nq, Nmodes, Nk, nb, nb)")
    print(f"  max|Re(g_mode)| = {np.max(np.abs(np.real(g_mode))):.4f} Ry/bohr")
    print(f"  max|Im(g_mode)| = {np.max(np.abs(np.imag(g_mode))):.4f} Ry/bohr")

    # ── q-points: crystal -> Cartesian 2pi/alat, via WFN.h5's reciprocal vectors ──
    qpoints_cart = raw['qpoints_mat'] @ wfn_hdr['rec_vecs']

    # ── Split into cond/val (BGW convention) ──
    g_cond_cart, g_val_cart = split_cond_val(g_all, Nval)
    g_cond_mode, g_val_mode = split_cond_val(g_mode, Nval)
    print(f"\nSplit into cond/val: g_cond_cart {g_cond_cart.shape}, "
          f"g_val_cart {g_val_cart.shape}")

    if args.save_not_filtered:
        save_elph_h5(
            args.save_not_filtered, g_cond_cart, g_val_cart, g_cond_mode, g_val_mode,
            kpoints_crystal=raw['kpoints_mat'], qpoints_crystal=raw['qpoints_mat'],
            qpoints_cart=qpoints_cart, phonon_modes=phonon_modes_dict,
            structure=structure, qp_rescaling=None,
            extra_attrs={'grid': 'fine', 'Nval': Nval, 'filtered': False,
                         'source': 'ElePhAny.jl (finite-difference DFT)',
                         'note': 'All available cond/val bands, no Nc/Nv windowing.'},
        )

    if args.eqp is not None and os.path.isfile(args.eqp):
        nc_target, nv_target = _get_band_window_from_eqp(args.eqp, Nval)
        print(f"\nMatching band window from {args.eqp}: "
              f"{nc_target} cond, {nv_target} val bands.")

        g_cond_cart, g_val_cart = _truncate_or_pad_bands(
            g_cond_cart, g_val_cart, nc_target, nv_target, label="ElePhAny")
        g_cond_mode, g_val_mode = _truncate_or_pad_bands(
            g_cond_mode, g_val_mode, nc_target, nv_target, label="ElePhAny")

        qp_ratio_c, qp_ratio_v, Eqp_c, Eqp_v, Edft_c, Edft_v = \
            _read_eqp_and_build_rescaling(args.eqp, Nk, nc_target, nv_target, Nval)
        qp_rescaling = dict(cond=qp_ratio_c, val=qp_ratio_v,
                             Eqp_cond=Eqp_c, Eqp_val=Eqp_v,
                             Edft_cond=Edft_c, Edft_val=Edft_v)

        save_elph_h5(
            args.elph_out, g_cond_cart, g_val_cart, g_cond_mode, g_val_mode,
            kpoints_crystal=raw['kpoints_mat'], qpoints_crystal=raw['qpoints_mat'],
            qpoints_cart=qpoints_cart, phonon_modes=phonon_modes_dict,
            structure=structure, qp_rescaling=qp_rescaling,
            extra_attrs={'grid': 'fine', 'Nval': Nval,
                         'source': 'ElePhAny.jl (finite-difference DFT)',
                         'interpolation': 'none (el-ph computed directly on the fine grid)'},
        )
        print(f"\n{args.elph_out} written: band-matched, QP-rescaled el-ph, "
              f"ready to use as elph_fine_h5_file in forces.inp.")
    else:
        if args.eqp is not None:
            print(f"\nNOTE: file not found for --eqp: '{args.eqp}' — skipping QP rescaling.")
        save_elph_h5(
            args.elph_out, g_cond_cart, g_val_cart, g_cond_mode, g_val_mode,
            kpoints_crystal=raw['kpoints_mat'], qpoints_crystal=raw['qpoints_mat'],
            qpoints_cart=qpoints_cart, phonon_modes=phonon_modes_dict,
            structure=structure, qp_rescaling=None,
            extra_attrs={'grid': 'fine', 'Nval': Nval,
                         'source': 'ElePhAny.jl (finite-difference DFT)',
                         'interpolation': 'none (el-ph computed directly on the fine grid)'},
        )
        print(f"\n{args.elph_out} written (no QP rescaling — pass --eqp pointing at "
              f"the fine-grid eqp.dat to also get that).")

    total_time = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {total_time:.1f} s.")


if __name__ == '__main__':
    main()
