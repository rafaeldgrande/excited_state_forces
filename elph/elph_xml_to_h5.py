"""
elph_xml_to_h5.py
==================
Combined electron-phonon (el-ph) pipeline: assembles QE DFPT elph XML files
into a coarse-grid HDF5 file, then (optionally) interpolates to a fine k-grid
using BerkeleyGW's dtmat coarse-to-fine overlap matrices. Writes elph_orig_kgrid.h5
and/or elph_interpolated_kgrid.h5 with identical dataset schemas (see "Output HDF5 layout").

This module merges what used to be two separate scripts (assemble_elph_h5.py +
interpolate_elph_bgw.py) into one, so a single coarse-grid dataset (kept in
memory) can flow straight into interpolation without a disk round-trip, while
still supporting each stage independently (--skip-interpolation to stop after
the coarse file; omit --elph_dir to resume interpolation from an existing
elph_orig_kgrid.h5).

Usage
-----
# Full pipeline: assemble XML -> elph_orig_kgrid.h5, interpolate -> elph_interpolated_kgrid.h5
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 \\
    --dtmat dtmat --wfn_to_interpolate WFN_fi.h5 --eqp eqp.dat

# Coarse only (no interpolation)
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 \\
    --skip-interpolation

# Resume: interpolate from a previously-written elph_orig_kgrid.h5
python elph_xml_to_h5.py --elph_coarse elph_orig_kgrid.h5 --wfn_origin WFN_co.h5 \\
    --dtmat dtmat --wfn_to_interpolate WFN_fi.h5

# No WFN_co.h5 available: fall back to scf.in / scf.out / pseudopotentials
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --qe_input scf.in \\
    --dtmat dtmat --wfn_to_interpolate WFN_fi.h5

# Manual Nval override (takes precedence over WFN_co.h5 / scf.in either way)
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 --Nval 13

Assembly stage (XML -> coarse grid)
------------------------------------
QE 7.5, ldisp=.true., electron_phonon='simple'.
         QE writes one elph.iq.ipert.xml per (q, perturbation) where
         "ipert" indexes the symmetry-adapted patterns chosen by ph.x
         (NOT necessarily simple Cartesian unit vectors).

         The directory containing the elph*.xml files (i.e. the phsave
         directory, e.g. "_ph0/<prefix>.phsave") is passed via --elph_dir.
         The displacement patterns U[ipert, alpha] are read from
         <elph_dir>/patterns.iq.xml and used to rotate
         <n,k+q|dV/du_pattern_ipert|m,k>  ->  <n,k+q|dV/du_cart_alpha|m,k>.

Cell parameters, reciprocal lattice, Nat, k-points, Nbnds and Nval are read
from a BerkeleyGW WFN_co.h5 (--wfn_origin) when given, else from the QE input
file (--qe_input, via ASE + scf.out/pseudopotentials for Nval).
The list of q-points and Nq are read from <elph_dir>/control_ph.xml
(Q-POINT_COORDINATES, units = 2pi/a).

If ANY ipert file is missing for a given iq, the entire q-point is skipped
and g remains zero for that iq.

Interpolation stage (coarse -> fine grid), one q at a time
------------------------------------------------------------
  <n, k_fi+q | dV(q) | m, k_fi>
      = sum_{a,b} <n, k_fi+q | a, k_co+q>
                * <a, k_co+q | dV(q) | b, k_co>
                * <b, k_co | m, k_fi>

The overlap matrices <n,k_fi|a,k_co> are dcn (conduction) and dvn (valence)
from the dtmat file (read via bgw_binary_io.read_dtmat).

BGW valence-band ordering convention
-------------------------------------
In BerkeleyGW, valence bands are indexed from the Fermi level *downward*:
  BGW v=0  ->  QE band Nval-1  (HOMO, 0-indexed)
  BGW v=1  ->  QE band Nval-2  (HOMO-1)
  ...
This means the valence block must be reversed along both band axes before
applying dvn. The output elph_val arrays use the same BGW ordering (v=0 =
HOMO), which is what the rest of the excited_forces code expects.

Output HDF5 layout (identical schema for elph_orig_kgrid.h5 and elph_interpolated_kgrid.h5 --
only the coefficients and k-points differ)
------------------------------------------------------------------------------
  elph_cond_mode   complex128  (Nq, Nmodes, Nk, Nc, Nc)  phonon-mode basis, optional
  elph_val_mode    complex128  (Nq, Nmodes, Nk, Nv, Nv)  phonon-mode basis, optional
  elph_cond_cart   complex128  (Nq, Npert,  Nk, Nc, Nc)  Cartesian basis
  elph_val_cart    complex128  (Nq, Npert,  Nk, Nv, Nv)  Cartesian basis
                   alpha = 3*iatom + {x,y,z}; conduction ic=0 -> LUMO;
                   valence iv=0 -> HOMO (BGW convention)

  Kpoints_in_elph_file  float64  (Nk, 3)  crystal (fractional) coords
  qpoints_cart          float64  (Nq, 3)  Cartesian 2pi/a
  qpoints_crystal       float64  (Nq, 3)  crystal (fractional) coords

  phonon_modes/          (group, present if matdyn.modes exists)
    qpoints              float64  (Nq_modes, 3)  Cartesian 2pi/a
    frequencies          float64  (Nq_modes, Nmodes)  cm^{-1}
    eigenvectors          complex128  (Nq_modes, Nmodes, Nat, 3)

  crystal/                (group -- structure info, from WFN_co.h5 or scf.in)
    atomic_numbers        int32    (Nat,)
    atomic_positions       float64  (Nat, 3)  Cartesian, Angstrom
    lattice_vectors        float64  (3, 3)    rows = a1,a2,a3, Angstrom

  QP_rescaling_matrix_cond/val, Eqp_cond/val, Edft_cond/val   (fine grid only,
    present when --eqp is given)

  Root attrs: grid ('coarse'/'fine'), Nval, and other provenance info.

Pattern -> Cartesian transform
-------------------------------
For a displacement pattern u_ipert expanded in Cartesian basis as
   delta_tau_alpha = U[ipert, alpha] * lambda_ipert
the chain rule gives
   g_pattern[ipert] = sum_alpha U[ipert, alpha] * g_cart[alpha].
Since U is unitary (orthonormal pattern basis), inverting yields
   g_cart[alpha] = sum_ipert conj(U[ipert, alpha]) * g_pattern[ipert].
This matches QE's own convention (cf. PHonon/PH/elphon.f90).
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import h5py
import xml.etree.ElementTree as ET
from ase.io import read as ase_read
from ase.data import chemical_symbols

# ── locate bgw_binary_io relative to this file ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bgw_binary_io import read_dtmat

TOL_K = 1e-5   # tolerance for k-point matching in crystal coords
BOHR_TO_ANGSTROM = 0.529177210903


def _require_path(path: str, flag: str, what: str | None = None, kind: str = 'file') -> None:
    """
    Raise a clear, actionable error if `path` does not exist, naming the CLI
    flag that controls it. Every file/directory this module opens that is
    controlled by a CLI flag goes through this check first.
    """
    exists = os.path.isdir(path) if kind == 'dir' else os.path.isfile(path)
    if not exists:
        noun = 'Directory' if kind == 'dir' else 'File'
        desc = f" ({what})" if what else ""
        raise FileNotFoundError(
            f"{noun} not found: '{path}'{desc}. Provide it with --{flag}. "
            f"Run 'python elph_xml_to_h5.py -h' to see all options.")


def _require_file(path: str, flag: str, what: str | None = None) -> None:
    _require_path(path, flag, what, kind='file')


# ══════════════════════════════════════════════════════════════════════════════
# Parse matdyn.modes (phonon frequencies and eigenvectors)
# ══════════════════════════════════════════════════════════════════════════════

def parse_matdyn_modes(path, nat):
    """
    Parse the matdyn.modes file produced by QE's matdyn.x.

    The file contains one block per q-point.  Each block has:
      - a header line:  q = qx qy qz
      - for each mode nu = 1 .. 3*nat:
          freq (nu) = ... [THz] = ... [cm-1]
          nat lines, each with 3 complex pairs:  ( Re Im  Re Im  Re Im )
          giving the displacement eigenvector components (x,y,z) for that atom.

    Returns
    -------
    qpoints     : (Nq, 3) float64
                  q-point coordinates as printed in matdyn.modes (Cartesian, 2pi/a)
    frequencies : (Nq, Nmodes) float64
                  phonon frequencies in cm^{-1}
    eigenvectors: (Nq, Nmodes, nat, 3) complex128
                  mode displacement vectors in real (Cartesian) space,
                  normalised so that for each (iq, nu):
                    sqrt( sum_{iat, xyz} |e_{iq,nu,iat,xyz}|^2 ) = 1.
                  eigenvectors[iq, nu, iat, :] = displacement of atom iat in mode nu
                  at q-point iq.
    """
    nmodes = 3 * nat

    with open(path) as f:
        text = f.read()

    # Split into q-point blocks using the "q = ..." header
    # Each block starts with a line containing " q = "
    q_blocks = re.split(r'(?=\s+q\s*=\s*[-\d])', text)
    q_blocks = [b for b in q_blocks if re.search(r'q\s*=', b)]

    nq = len(q_blocks)
    if nq == 0:
        raise ValueError(f"No q-point blocks found in {path}")

    qpoints      = np.zeros((nq, 3),               dtype=np.float64)
    frequencies  = np.zeros((nq, nmodes),           dtype=np.float64)
    eigenvectors = np.zeros((nq, nmodes, nat, 3),   dtype=np.complex128)

    for iq, block in enumerate(q_blocks):
        lines = block.strip().splitlines()

        # Parse q-point coordinates from the "q = ..." line
        q_match = re.search(r'q\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', lines[0])
        if q_match is None:
            raise ValueError(f"Could not parse q-point from line: {lines[0]}")
        qpoints[iq] = [float(q_match.group(i)) for i in (1, 2, 3)]

        # Parse freq + eigenvector blocks
        mode_idx = 0
        i = 1  # skip the q-line
        while i < len(lines):
            line = lines[i].strip()

            # Look for frequency line
            freq_match = re.search(
                r'freq\s*\(\s*(\d+)\s*\)\s*=\s*([-\d.]+)\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]',
                line)
            if freq_match:
                nu = int(freq_match.group(1)) - 1  # 0-indexed
                frequencies[iq, nu] = float(freq_match.group(3))  # cm^{-1}

                # Next nat lines are the eigenvector for this mode
                for iat in range(nat):
                    i += 1
                    evec_line = lines[i].strip()
                    # Format: ( Re Im  Re Im  Re Im )
                    nums = re.findall(r'[-\d.]+', evec_line)
                    if len(nums) != 6:
                        raise ValueError(
                            f"{path}: q-block {iq+1}, mode {nu+1}, atom {iat+1}: "
                            f"expected 6 numbers, got {len(nums)}: {evec_line}")
                    eigenvectors[iq, nu, iat, 0] = float(nums[0]) + 1j * float(nums[1])
                    eigenvectors[iq, nu, iat, 1] = float(nums[2]) + 1j * float(nums[3])
                    eigenvectors[iq, nu, iat, 2] = float(nums[4]) + 1j * float(nums[5])

                mode_idx += 1
            i += 1

        if mode_idx != nmodes:
            raise ValueError(
                f"{path}: q-block {iq+1} has {mode_idx} modes, expected {nmodes}")

    # ── Normalise each eigenvector to unit norm in the 3*nat-dimensional
    #    complex space:  ||e_{iq,nu}|| = sqrt(sum_{iat,xyz} |e|^2) = 1.
    #    matdyn.x usually prints normalised vectors, but we enforce it here
    #    to be safe.  Shape: (nq, nmodes, nat, 3) -> norm over last two axes.
    norms = np.sqrt(
        np.sum(np.abs(eigenvectors) ** 2, axis=(-2, -1), keepdims=True))
    # Guard against zero-norm vectors (shouldn't happen, but avoid div-by-zero)
    norms = np.where(norms < 1e-30, 1.0, norms)
    eigenvectors /= norms

    return qpoints, frequencies, eigenvectors


# ══════════════════════════════════════════════════════════════════════════════
# QE input fallback: cell/k-points/nbnd (via ASE + regex), used only when
# --wfn_origin is not given / not found
# ══════════════════════════════════════════════════════════════════════════════

def load_params_from_qe_input(qe_input_path):
    """
    Use ASE to read cell geometry and derive reciprocal lattice vectors.
    Parse nbnd and k-points directly from the file.

    qe_input_path : str
        Path to a QE pw.x input file (scf.in or bands.in both work — only
        the cell, &SYSTEM/nbnd, and K_POINTS block are used).

    Returns
    -------
    nat       : int        number of atoms
    npert     : int        3 * nat (Cartesian displacements, nosym=.true.)
    nbnds     : int        number of bands (from nbnd in &SYSTEM)
    kpts_nscf : (Nk, 3)   NSCF k-points in crystal coords
    nk        : int        number of k-points
    rec_vecs  : (3, 3)     reciprocal lattice vectors in units of 2pi/alat,
                           rows = b1, b2, b3
    bt_inv    : (3, 3)     (B^T)^{-1} for Cartesian-2pi/alat -> crystal conversion
    atomic_numbers       : (nat,) int      atomic numbers, QE input atom order
    atomic_positions_ang : (nat, 3) float64  Cartesian positions, Angstrom
    lattice_vectors_ang  : (3, 3) float64    rows = a1,a2,a3, Angstrom
    """
    # ── ASE: cell geometry ──────────────────────────────────────────────────
    atoms = ase_read(qe_input_path, format='espresso-in')
    nat   = len(atoms)
    npert = 3 * nat

    # Lattice constant = |a1| in Angstrom
    alat_ang = np.linalg.norm(atoms.get_cell()[0])

    # Reciprocal lattice vectors in 2pi/alat units:
    # ASE's cell.reciprocal() satisfies b_i . a_j = delta_ij  (no 2pi factor)
    # Multiplying by alat gives units of 2pi/alat.
    rec_vecs = np.array(atoms.get_cell().reciprocal()) * alat_ang
    bt_inv   = np.linalg.inv(rec_vecs.T)

    # ── nbnd: parse from &SYSTEM block ─────────────────────────────────────
    with open(qe_input_path) as f:
        content = f.read()
    m = re.search(r'\bnbnd\s*=\s*(\d+)', content, re.IGNORECASE)
    if m is None:
        raise ValueError(f"Could not find 'nbnd' in {qe_input_path}")
    nbnds = int(m.group(1))

    # ── k-points: parse K_POINTS block ─────────────────────────────────────
    kpts = []
    nk_expected = 0
    in_kblock = False
    for line in content.splitlines():
        s = line.strip()
        if s.upper().startswith('K_POINTS'):
            in_kblock = True
            continue
        if not in_kblock:
            continue
        if nk_expected == 0:
            nk_expected = int(s.split()[0])
            continue
        parts = s.split()
        if len(parts) >= 3:
            kpts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(kpts) == nk_expected:
            break
    kpts_nscf = np.array(kpts, dtype=np.float64)
    nk        = len(kpts_nscf)

    atomic_numbers       = np.array(atoms.get_atomic_numbers())
    atomic_positions_ang = np.array(atoms.get_positions())
    lattice_vectors_ang  = np.array(atoms.get_cell()[:])

    return (nat, npert, nbnds, kpts_nscf, nk, rec_vecs, bt_inv,
            atomic_numbers, atomic_positions_ang, lattice_vectors_ang)


# ── Nval: count electrons from the QE input (ATOMIC_SPECIES + pseudopotentials) ──

_QE_CARD_KEYWORDS = {
    'ATOMIC_SPECIES', 'ATOMIC_POSITIONS', 'K_POINTS', 'CELL_PARAMETERS',
    'OCCUPATIONS', 'CONSTRAINTS', 'ATOMIC_FORCES', 'ADDITIONAL_K_POINTS',
    'HUBBARD', 'SOLVENTS', 'TOTAL_CHARGE',
}


def _is_qe_card_header(line: str) -> bool:
    parts = line.strip().split()
    return bool(parts) and parts[0].upper() in _QE_CARD_KEYWORDS


def _parse_qe_species_and_atoms(qe_input_path: str):
    """
    Parse the ATOMIC_SPECIES and ATOMIC_POSITIONS cards of a QE pw.x input.

    Returns
    -------
    pseudo_dir       : str            value of pseudo_dir in &CONTROL (may be '.')
    species_pseudo   : dict[str, str] species label -> pseudopotential filename
    atom_species_list: list[str]      species label for each atom, in file order
    """
    with open(qe_input_path) as f:
        lines = f.readlines()
    content = ''.join(lines)

    m = re.search(r"pseudo_dir\s*=\s*['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
    pseudo_dir = os.path.expandvars(os.path.expanduser(m.group(1))) if m else '.'

    species_pseudo: dict[str, str] = {}
    atom_species_list: list[str] = []

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if s.upper().startswith('ATOMIC_SPECIES'):
            i += 1
            while i < n and lines[i].strip() and not _is_qe_card_header(lines[i]):
                parts = lines[i].split()
                if len(parts) >= 3:
                    species_pseudo[parts[0]] = parts[2]
                i += 1
            continue
        if s.upper().startswith('ATOMIC_POSITIONS'):
            i += 1
            while i < n and lines[i].strip() and not _is_qe_card_header(lines[i]):
                parts = lines[i].split()
                if len(parts) >= 4:
                    atom_species_list.append(parts[0])
                i += 1
            continue
        i += 1

    if not species_pseudo:
        raise ValueError(f"Could not find ATOMIC_SPECIES card in {qe_input_path}")
    if not atom_species_list:
        raise ValueError(f"Could not find ATOMIC_POSITIONS card in {qe_input_path}")

    return pseudo_dir, species_pseudo, atom_species_list


def _read_upf_zvalence(upf_path: str) -> float:
    """Read the Z_valence (number of valence electrons) from a UPF pseudopotential."""
    with open(upf_path, errors='ignore') as f:
        content = f.read()

    # UPF v2 / v2.0.1 (XML-like): <PP_HEADER ... z_valence="6.00" ... />
    m = re.search(r'z_valence\s*=\s*"([^"]+)"', content, re.IGNORECASE)
    if m is None:
        # UPF v1 (old fixed-format text): "   6.00000000000      Z valence"
        m = re.search(r'^\s*([\d.eE+-]+)\s+Z\s*valence', content,
                      re.IGNORECASE | re.MULTILINE)
    if m is None:
        raise ValueError(f"Could not parse z_valence from pseudopotential: {upf_path}")
    return float(m.group(1))


def _find_qe_output_file(qe_input_path: str) -> str | None:
    """
    Look for the QE pw.x *output* file corresponding to qe_input_path (e.g.
    scf.out next to scf.in). Its 'number of electrons' line is authoritative
    and lets us avoid re-deriving the electron count from pseudopotentials.
    """
    d = os.path.dirname(os.path.abspath(qe_input_path))
    base = os.path.splitext(os.path.basename(qe_input_path))[0]
    candidates = [os.path.join(d, base + '.out')]
    candidates += [os.path.join(d, name) for name in ('scf.out', 'pw.out', 'nscf.out')]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _read_nelec_from_qe_output(qe_output_path: str) -> float | None:
    """Parse the 'number of electrons = XXX.XX' line from a QE pw.x stdout file."""
    with open(qe_output_path, errors='ignore') as f:
        for line in f:
            m = re.search(r'number of electrons\s*=\s*([\d.]+)', line, re.IGNORECASE)
            if m is not None:
                return float(m.group(1))
    return None


def _count_electrons_from_pseudopotentials(qe_input_path: str,
                                            species_pseudo: dict[str, str]) -> float:
    """Fallback: sum Z_valence * n_atoms by reading each species' UPF file."""
    pseudo_dir, _, atom_species_list = _parse_qe_species_and_atoms(qe_input_path)
    qe_dir = os.path.dirname(os.path.abspath(qe_input_path))
    search_dirs = [pseudo_dir, os.path.join(qe_dir, pseudo_dir), qe_dir, '.']

    species_zval: dict[str, float] = {}
    for label, pseudo_fname in species_pseudo.items():
        upf_path = None
        for d in search_dirs:
            cand = os.path.join(d, pseudo_fname)
            if os.path.isfile(cand):
                upf_path = cand
                break
        if upf_path is None:
            raise FileNotFoundError(
                f"Could not find pseudopotential '{pseudo_fname}' for species "
                f"'{label}' (searched: {search_dirs}), and no QE output file "
                f"with a 'number of electrons' line was found either.")
        species_zval[label] = _read_upf_zvalence(upf_path)

    counts = Counter(atom_species_list)
    return sum(n * species_zval[label] for label, n in counts.items())


def get_nval_from_qe_input(qe_input_path: str) -> int:
    """
    Determine Nval (highest occupied/valence band index, QE nbnd convention)
    by counting electrons for the system in qe_input_path.

    Preferred source: the 'number of electrons = ...' line QE itself prints
    in the matching pw.x output file (e.g. scf.out next to scf.in) — this is
    authoritative and avoids re-deriving the count from pseudopotentials
    (which is unreliable in general: Z_valence depends on the specific
    pseudopotential, e.g. semicore states, not just the element).
    Falls back to reading Z_valence from the ATOMIC_SPECIES pseudopotential
    files only if no such output file can be found.

    Prints a summary of the atom counts per species, the total number of
    electrons, and the resulting Nval, per module usage instructions.
    """
    pseudo_dir, species_pseudo, atom_species_list = _parse_qe_species_and_atoms(qe_input_path)
    counts = Counter(atom_species_list)

    print(f"\nCounting electrons for QE input: {qe_input_path}")
    found_str = ', '.join(f"{n} atoms {label}" for label, n in counts.items())
    print(f"  found {found_str}")

    qe_output_path = _find_qe_output_file(qe_input_path)
    total_electrons = _read_nelec_from_qe_output(qe_output_path) if qe_output_path else None

    if total_electrons is not None:
        print(f"  Read total electron count directly from QE output: {qe_output_path}")
    else:
        print(f"  No QE output file with a 'number of electrons' line found next to "
              f"{qe_input_path} — falling back to summing Z_valence from the "
              f"ATOMIC_SPECIES pseudopotential files.")
        total_electrons = _count_electrons_from_pseudopotentials(qe_input_path, species_pseudo)

    total_electrons_int = int(round(total_electrons))
    if abs(total_electrons - total_electrons_int) > 1e-3:
        warnings.warn(
            f"Total number of electrons ({total_electrons}) is not an integer "
            f"to within 1e-3 — check the pseudopotentials.")
    if total_electrons_int % 2 != 0:
        warnings.warn(
            f"Total number of electrons ({total_electrons_int}) is odd — the "
            f"system may be spin-polarized or have fractional occupations; "
            f"Nval = electrons // 2 may not be correct.")

    Nval = total_electrons_int // 2
    print(f"  Total number of electrons is {total_electrons_int} and the "
          f"highest valence band index is Nv(={total_electrons_int}/2)={Nval}.")
    print(f"  If wrong reexecute the code with --Nval Nv with the correct "
          f"index of the highest occupied band to overwrite it.")

    return Nval


# ══════════════════════════════════════════════════════════════════════════════
# WFN.h5 reading (preferred source for cell/k-points/nbnd/Nval/atomic positions)
# ══════════════════════════════════════════════════════════════════════════════

def read_wfn_h5_header(wfn_h5_path: str, flag: str = 'wfn_origin') -> dict:
    """
    Read mean-field header info from a BerkeleyGW WFN.h5 file (WFN_co.h5 or
    WFN_fi.h5), following the mf_header layout documented at
    http://manual.berkeleygw.org/4.0/wfn_h5_spec/. h5py reverses the Fortran
    axis order relative to the BerkeleyGW source (confirmed against
    Common/wfn_io_hdf5_inc.f90 and a real WFN.h5 file).

    flag : name of the CLI flag that supplies wfn_h5_path (used only to build
        a clear error message if the file is missing).

    Nval (highest occupied/valence band index) is read directly from
    mf_header/kpoints/ifmax — the same pattern already used in
    main/bgw_interface_m.py:get_params_from_eigenvecs_file for eigenvectors.h5
    (which shares the same mf_header block): Nval = min(ifmax) over k-points
    for spin 0, assuming a semiconductor (warns if ifmax varies across k).

    Returns a dict with:
      kpts_crystal          (nrk, 3) float64   crystal (fractional) coords
      nbnd                  int                mnband
      rec_vecs               (3, 3) float64     = bvec, units 2pi/alat
      nat                    int
      atomic_numbers         (nat,) int         = atyp
      atomic_positions_ang   (nat, 3) float64    Cartesian, Angstrom
      lattice_vectors_ang    (3, 3) float64      rows = a1,a2,a3, Angstrom
      Nval                   int
    """
    _require_file(wfn_h5_path, flag, 'BerkeleyGW WFN.h5 (mean-field header)')
    with h5py.File(wfn_h5_path, 'r') as fh:
        mf = fh['mf_header']
        kpts_crystal = mf['kpoints/rk'][:]                  # (nrk, 3)
        nbnd         = int(mf['kpoints/mnband'][()])
        nspin        = int(mf['kpoints/nspin'][()])
        ifmax        = mf['kpoints/ifmax'][:]               # (nspin, nrk)

        alat  = float(mf['crystal/alat'][()])                # bohr
        avec  = mf['crystal/avec'][:]                         # (3,3), units alat
        bvec  = mf['crystal/bvec'][:]                         # (3,3), units 2pi/alat
        nat   = int(mf['crystal/nat'][()])
        atyp  = mf['crystal/atyp'][:]                         # (nat,) atomic numbers
        apos  = mf['crystal/apos'][:]                         # (nat,3), Cartesian, units alat

    atomic_positions_ang = apos * alat * BOHR_TO_ANGSTROM
    lattice_vectors_ang  = avec * alat * BOHR_TO_ANGSTROM

    ifmax_spin0 = ifmax[0]                                    # (nrk,)
    ifmax_values = sorted(set(int(v) for v in ifmax_spin0))
    Nval = min(ifmax_values)

    print(f"\nReading structure/Nval from WFN.h5: {wfn_h5_path}")
    counts = Counter(chemical_symbols[z] for z in atyp)
    found_str = ', '.join(f"{n} atoms {label}" for label, n in counts.items())
    print(f"  found {found_str}")
    if len(ifmax_values) > 1:
        warnings.warn(
            f"ifmax varies across k-points/spin in {wfn_h5_path} "
            f"({ifmax_values}) — system may be metallic. Using min(ifmax) = "
            f"{Nval} as Nval (semiconductor assumption).")
    if nspin > 1:
        warnings.warn(
            f"nspin={nspin} in {wfn_h5_path} — Nval read from spin index 0 only.")
    print(f"  Read highest occupied band index directly from WFN.h5 ifmax: "
          f"Nval = {Nval}.")
    print(f"  If wrong reexecute the code with --Nval Nv with the correct "
          f"index of the highest occupied band to overwrite it.")

    return dict(
        kpts_crystal=kpts_crystal,
        nbnd=nbnd,
        rec_vecs=bvec,
        nat=nat,
        atomic_numbers=np.asarray(atyp),
        atomic_positions_ang=atomic_positions_ang,
        lattice_vectors_ang=lattice_vectors_ang,
        Nval=Nval,
    )


# ══════════════════════════════════════════════════════════════════════════════
# k-point / crystal utilities
# ══════════════════════════════════════════════════════════════════════════════

def cart_to_crystal(kpts_cart, bt_inv):
    """Convert k-points from Cartesian 2pi/alat to crystal (fractional) coords."""
    return kpts_cart @ bt_inv.T


def wrap_to_bz(k):
    """Wrap crystal coordinates to [0, 1)."""
    return k - np.floor(k + 1e-10)


def find_kpt_index(k_query, kpts_ref_crys):
    """Return index in kpts_ref_crys matching k_query (mod lattice), or -1."""
    kq = wrap_to_bz(np.asarray(k_query, dtype=np.float64))
    for ik, kr in enumerate(kpts_ref_crys):
        diff = wrap_to_bz(np.asarray(kr, dtype=np.float64)) - kq
        diff -= np.round(diff)
        if np.linalg.norm(diff) < TOL_K:
            return ik
    return -1


def _build_kpt_map(kpts_query: np.ndarray,
                    kpts_ref: np.ndarray,
                    tol: float = 1e-5) -> np.ndarray:
    """
    For each row in kpts_query find the matching row index in kpts_ref
    (modulo reciprocal lattice vectors).  Returns array of ints; -1 = not found.
    """
    nq = len(kpts_query)
    idx_map = np.full(nq, -1, dtype=int)
    kref_w = wrap_to_bz(kpts_ref)               # (Nref, 3)
    for iq, kq in enumerate(kpts_query):
        kq_w = wrap_to_bz(kq)
        diff = kref_w - kq_w                    # (Nref, 3)
        diff -= np.round(diff)                  # fold to [-0.5, 0.5)
        dists = np.linalg.norm(diff, axis=1)
        best = int(np.argmin(dists))
        if dists[best] < tol:
            idx_map[iq] = best
    return idx_map


def check_extended_zone(kmap, kpts_nscf, kpts_xml_crys, g_xml):
    """
    Two checks performed after k-mapping:

    (1) BZ-copy check: verify that kmap[ik] points to the first-BZ copy of
        each NSCF k-point (not an extended-zone image).  With QE ldisp+nosym,
        irrek() appends extended-zone copies *after* the original Nk k-points,
        so find_kpt_index (which iterates in XML order) should always hit the
        first-BZ entry first.  If any match is NOT in the first BZ, we warn.

    (2) |g| consistency check: for k-points that appear more than once in the
        XML (extended-zone copies), verify that |g| is the same at every copy.
        The phase WILL differ between copies (by e^{iG.r}), but the magnitude
        must be identical.  Max deviation is reported.

    Prints a one-line summary.  Returns (n_in_bz, max_abs_diff_Ry).
    """
    NK_NSCF    = len(kmap)
    NK_XML     = len(kpts_xml_crys)
    n_in_bz    = 0
    n_bad_bz   = 0
    max_g_diff = 0.0
    n_copies_checked = 0

    for ik_nscf in range(NK_NSCF):
        idx = kmap[ik_nscf]
        if idx < 0:
            continue

        kn = kpts_nscf[ik_nscf]
        kx = kpts_xml_crys[idx]

        # (1) Is the matched XML k-point the first-BZ copy?
        if np.linalg.norm(kx - kn) < TOL_K * 100:
            n_in_bz += 1
        else:
            n_bad_bz += 1
            print(f"    WARNING: k[{ik_nscf}] matched to extended-zone copy "
                  f"{kx} instead of first-BZ {kn}  (diff={kx - kn})")

        # (2) Find ALL XML copies of this k-point and compare |g|
        g_ref_abs = np.abs(g_xml[idx])
        for ij in range(NK_XML):
            if ij == idx:
                continue
            diff_crys = kpts_xml_crys[ij] - kn
            diff_crys -= np.round(diff_crys)        # fold difference to [-0.5, 0.5)
            if np.linalg.norm(diff_crys) < TOL_K * 10:
                # This is another extended-zone copy
                abs_diff = np.max(np.abs(np.abs(g_xml[ij]) - g_ref_abs))
                max_g_diff = max(max_g_diff, abs_diff)
                n_copies_checked += 1

    status_bz = "OK" if n_bad_bz == 0 else f"WARNING: {n_bad_bz} not in 1st BZ"
    print(f"  [BZ-copy check]  {n_in_bz}/{NK_NSCF} matched to 1st-BZ copy -> {status_bz}")
    if n_copies_checked > 0:
        print(f"  [|g| consistency] max ||g|_copy - |g|_ref| = {max_g_diff:.2e} Ry/bohr "
              f"over {n_copies_checked} extended-zone copy-pairs "
              f"({'OK — phase varies but |g| is stable' if max_g_diff < 1e-3 else 'WARNING: large discrepancy'})")
    else:
        print(f"  [|g| consistency] no extended-zone copies found (lgamma or all unique)")

    return n_in_bz, max_g_diff


# ══════════════════════════════════════════════════════════════════════════════
# Read q-points from control_ph.xml
# ══════════════════════════════════════════════════════════════════════════════

def read_qpoints_control_ph(path):
    """
    Parse <Q_POINTS> from _ph0/<prefix>.phsave/control_ph.xml.

    Returns
    -------
    qpts_cart : (Nq, 3) float64    Cartesian 2pi/a, in the order ph.x indexes them
                                   (iq = 1 .. Nq -> patterns.iq.xml, elph.iq.*.xml)
    """
    tree = ET.parse(path)
    root = tree.getroot()
    qblk = root.find('Q_POINTS')
    if qblk is None:
        raise ValueError(f"No <Q_POINTS> block in {path}")

    nq = int(qblk.find('NUMBER_OF_Q_POINTS').text.strip())

    # Sanity check on units (QE writes "2 pi / a")
    units_el = qblk.find('UNITS_FOR_Q-POINT')
    if units_el is not None:
        units = units_el.attrib.get('UNITS', '')
        if '2 pi' not in units and '2pi' not in units:
            print(f"  WARNING: unexpected q-point units '{units}' in {path}")

    coords_text = qblk.find('Q-POINT_COORDINATES').text
    nums = np.fromstring(coords_text, sep=' ')
    if nums.size != 3 * nq:
        raise ValueError(
            f"{path}: Q-POINT_COORDINATES has {nums.size} values, "
            f"expected {3 * nq}")
    return nums.reshape(nq, 3).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Read displacement patterns from patterns.iq.xml
# ══════════════════════════════════════════════════════════════════════════════

def read_patterns_xml(xml_path, nat):
    """
    Parse one patterns.iq.xml and return the displacement-pattern matrix.

    The XML organises perturbations as REPRESENTION.k -> PERTURBATION.j.
    The flat ipert index used by elph.iq.ipert.xml runs over (k, j)
    in XML order (rep 1 perturbations first, then rep 2, ...).

    Each <DISPLACEMENT_PATTERN> contains 3*nat (Re, Im) pairs giving the
    Cartesian-basis components of that pattern (atom-major: a1x, a1y, a1z,
    a2x, ...).

    Returns
    -------
    U : (npert, 3*nat) complex128
        U[ipert, alpha] = alpha-th Cartesian component of the ipert-th
        symmetry-adapted displacement pattern (unitary in the noncollinear
        sense — orthonormal across ipert).
    """
    npert_total = 3 * nat
    tree = ET.parse(xml_path)
    root = tree.getroot()
    irreps = root.find('IRREPS_INFO')
    if irreps is None:
        raise ValueError(f"No <IRREPS_INFO> block in {xml_path}")

    n_irr = int(irreps.find('NUMBER_IRR_REP').text.strip())
    U = np.zeros((npert_total, npert_total), dtype=np.complex128)

    ipert = 0
    for k in range(1, n_irr + 1):
        rep = irreps.find(f'REPRESENTION.{k}')
        if rep is None:
            raise ValueError(f"{xml_path}: missing REPRESENTION.{k}")
        n_per_rep = int(rep.find('NUMBER_OF_PERTURBATIONS').text.strip())
        for j in range(1, n_per_rep + 1):
            pert = rep.find(f'PERTURBATION.{j}')
            if pert is None:
                raise ValueError(
                    f"{xml_path}: missing REPRESENTION.{k}/PERTURBATION.{j}")
            text = pert.find('DISPLACEMENT_PATTERN').text
            nums = np.fromstring(text, sep=' ')
            if nums.size != 2 * npert_total:
                raise ValueError(
                    f"{xml_path}: REP.{k}/PERT.{j} has {nums.size} values, "
                    f"expected {2 * npert_total}")
            U[ipert] = nums[::2] + 1j * nums[1::2]
            ipert += 1

    if ipert != npert_total:
        raise ValueError(
            f"{xml_path}: total perturbations {ipert} != 3*nat = {npert_total}")

    # Sanity: U should be unitary (rows are orthonormal in C^{3*nat})
    err = np.max(np.abs(U @ U.conj().T - np.eye(npert_total)))
    if err > 1e-8:
        print(f"  WARNING: pattern matrix not unitary (max dev = {err:.2e}) "
              f"in {os.path.basename(xml_path)}")

    return U


# ══════════════════════════════════════════════════════════════════════════════
# Read one elph.iq.ipert.xml
# ══════════════════════════════════════════════════════════════════════════════

def read_elph_xml(xml_path):
    """
    Parse one elph.iq.ipert.xml (works for q=0 and q!=0).

    Returns
    -------
    kpts_cart : (NK_xml, 3) float64    Cartesian 2pi/alat
    g_mat     : (NK_xml, nbnds, nbnds) complex128
                g_mat[ik, n, m] = <n, k+q | dV_SCF/d(tau^ipert) | m, k>  [Ry/bohr]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pel   = root.find('PARTIAL_EL_PHON')
    nk    = int(pel.find('NUMBER_OF_K').text)
    nbnds = int(pel.find('NUMBER_OF_BANDS').text)

    kpts_cart = np.empty((nk, 3),             dtype=np.float64)
    g_mat     = np.empty((nk, nbnds, nbnds),  dtype=np.complex128)

    for ik in range(nk):
        kp = pel.find(f'K_POINT.{ik + 1}')
        kpts_cart[ik] = [float(v) for v in kp.find('COORDINATES_XK').text.split()]
        text  = kp.find('PARTIAL_ELPH').text.replace(',', ' ')
        nums  = np.fromstring(text, sep='\n')
        cplx  = nums[::2] + 1j * nums[1::2]
        if len(cplx) != nbnds * nbnds:
            raise ValueError(
                f"{xml_path}: K_POINT.{ik+1} has {len(cplx)} values, "
                f"expected {nbnds**2}")
        g_mat[ik] = cplx.reshape(nbnds, nbnds)

    return kpts_cart, g_mat


# ══════════════════════════════════════════════════════════════════════════════
# Acoustic sum rule (Cartesian basis)
# ══════════════════════════════════════════════════════════════════════════════

def apply_acoustic_sum_rule(g_cart, nat):
    """
    Enforce sum_atoms <n,k+q | dV / d(tau^{atom, d}) | m,k> = 0  for d = x,y,z.

    A rigid translation of the whole crystal must leave the self-consistent
    potential invariant, so the sum of Cartesian displacement derivatives over
    all atoms (along any one Cartesian direction) should vanish identically.
    Numerical noise (incomplete k-grids, finite tr2_ph, etc.) breaks this
    slightly; we restore it by subtracting the mean across atoms:

        S_d[iq, ik, n, m] = sum_atoms g[iq, 3*atom + d, ik, n, m]
        g_corrected[iq, 3*atom + d, ik, n, m]
            = g[iq, 3*atom + d, ik, n, m] - S_d[iq, ik, n, m] / nat

    This is the same prescription used for the dynamical matrix's "simple"
    ASR: redistribute the residual equally across all atoms so the new
    per-direction sum is zero.

    Parameters
    ----------
    g_cart : (Nq, 3*nat, Nk, Nb, Nb) complex128, modified in place
    nat    : int

    Returns
    -------
    max_residual_before : float    max |S_d| across all (iq, d, ik, n, m) before correction
    max_residual_after  : float    same after correction (should be ~ machine eps)
    """
    Nq, npert, Nk, Nb1, Nb2 = g_cart.shape
    if npert != 3 * nat:
        raise ValueError(f"npert = {npert} but 3*nat = {3 * nat}")

    # View g as (Nq, nat, 3, Nk, Nb, Nb); axis 1 = atom, axis 2 = direction
    view = g_cart.reshape(Nq, nat, 3, Nk, Nb1, Nb2)

    # S_d[iq, d, ik, n, m] = sum over atoms (axis 1)
    S = view.sum(axis=1)                          # (Nq, 3, Nk, Nb, Nb)
    max_residual_before = float(np.max(np.abs(S)))

    # Subtract S/nat from every atom's slice (broadcast over atom axis)
    view -= (S / nat)[:, np.newaxis, :, :, :, :]

    # New residual (should be ~ 1e-15)
    max_residual_after = float(np.max(np.abs(view.sum(axis=1))))

    # view shares storage with g_cart, so g_cart is already updated
    return max_residual_before, max_residual_after


# ══════════════════════════════════════════════════════════════════════════════
# Cond/val band splitting (BGW convention), shared by coarse write & interpolation
# ══════════════════════════════════════════════════════════════════════════════

def split_cond_val(g_raw, Nval):
    """
    Split a (..., Nk, Nbnds, Nbnds) array (last two axes = raw QE band index,
    0-indexed) into conduction and valence blocks using BGW conventions.

      g_cond[..., a, b] = g_raw[..., Nval+a, Nval+b]        a=0 -> LUMO
      g_val[..., a, b]  = g_raw[..., Nval-1-a, Nval-1-b]    a=0 -> HOMO (flip)

    Returns (g_cond, g_val) with shapes (..., Nbnds-Nval, Nbnds-Nval) and
    (..., Nval, Nval).
    """
    g_cond = g_raw[..., Nval:, Nval:]
    g_val  = g_raw[..., :Nval, :Nval][..., ::-1, ::-1]
    return g_cond, g_val


# ══════════════════════════════════════════════════════════════════════════════
# Interpolation core (coarse -> fine, via dtmat)
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_elph(
    g_co_cond:    np.ndarray,
    g_co_val:     np.ndarray,
    kpts_co:      np.ndarray,
    qpts_co:      np.ndarray,
    dtmat_file:   str | None,
    kpts_fi_crys: np.ndarray | None = None,
    tol_k:        float = 1e-5,
    complex_flavor: bool = True,
    wfn_fi_same_wfn_co: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate el-ph matrix elements from coarse to fine k-grid.

    See module docstring for the interpolation formula and BGW valence-band
    ordering convention.

    g_co_cond, g_co_val : (Nq, Nb, Nk_co, nc_avail, nc_avail) / (..., nv_avail,
        nv_avail) complex128 — already split into cond/val blocks (BGW
        ordering) via split_cond_val(g_raw, Nval), where Nb is either Nmodes
        or Npert depending on which basis is being interpolated.
    kpts_co, qpts_co : coarse-grid k-points (crystal) and q-points (crystal),
        matching the first/third axis of g_co_cond/g_co_val.
    dtmat_file : path to the BerkeleyGW dtmat binary, or None if
        wfn_fi_same_wfn_co=True.
    kpts_fi_crys : (Nk_fi, 3) array, optional
        Fine k-point coordinates in crystal (fractional) coordinates.
        Required only for finite-q interpolation (q != 0).
        Can be read from WFN_fi.h5 -> mf_header/kpoints/rk.
    wfn_fi_same_wfn_co : bool, optional
        Set when WFN_fi == WFN_co, i.e. the el-ph coefficients were computed
        directly on the fine grid and no coarse-to-fine interpolation is
        needed. The coarse-to-fine wavefunction overlaps (normally read from
        dtmat) are replaced by identity matrices, and dtmat_file is ignored
        (may be None). kpts_fi_crys defaults to kpts_co when not given.

    Returns
    -------
    elph_fine_cond : (Nq, Nb, Nk_fi, Nc_fi, Nc_fi)  complex128
    elph_fine_val  : (Nq, Nb, Nk_fi, Nv_fi, Nv_fi)  complex128
    """
    Nq, Nmodes, Nk_co, nc_co_avail, _ = g_co_cond.shape
    nv_co_avail = g_co_val.shape[-1]
    print(f"\n{'='*64}")
    print(f"Interpolating coarse -> fine el-ph")
    print(f"{'='*64}")
    print(f"  g_co_cond shape : {g_co_cond.shape}  ({Nq} q, {Nmodes} modes, "
          f"{Nk_co} k_co, {nc_co_avail} cond bands)")
    print(f"  g_co_val  shape : {g_co_val.shape}  ({nv_co_avail} val bands)")

    # ── Load dtmat, or build an identity coarse-to-fine transformation ───
    if wfn_fi_same_wfn_co:
        print(f"\nwfn_fi_same_wfn_co=True: WFN_fi == WFN_co, using identity "
              f"coarse-to-fine transformation (dtmat not read).")
        if dtmat_file is not None:
            print(f"  (ignoring dtmat_file={dtmat_file})")

        nk_co_dt  = Nk_co
        nk_fi     = Nk_co
        nc_co     = nc_co_avail
        nv_co     = nv_co_avail
        nc_fi     = nc_co
        nv_fi     = nv_co
        nspin     = 1
        npts      = 1
        kco_dt    = kpts_co             # (Nk_co, 3), same ordering as g_co_cond/val

        if kpts_fi_crys is None:
            # Fine k-points are the coarse k-points themselves.
            kpts_fi_crys = kpts_co
            fine_to_co_base = np.arange(Nk_co, dtype=int)
        else:
            fine_to_co_base = _build_kpt_map(kpts_fi_crys, kpts_co, tol=tol_k)
            if not np.all(fine_to_co_base >= 0):
                n_bad = int(np.sum(fine_to_co_base < 0))
                raise ValueError(
                    f"wfn_fi_same_wfn_co=True but {n_bad}/{len(kpts_fi_crys)} "
                    f"fine k-points could not be matched to the coarse grid. "
                    f"WFN_fi and WFN_co must share the same k-grid.")

        # dcn/dvn: identity overlap for every fine k-point (broadcast, no copy)
        dcn = np.broadcast_to(
            np.eye(nc_fi, dtype=complex)[:, :, None, None, None],
            (nc_fi, nc_co, nspin, nk_fi, npts))
        dvn = np.broadcast_to(
            np.eye(nv_fi, dtype=complex)[:, :, None, None, None],
            (nv_fi, nv_co, nspin, nk_fi, npts))
        fi2co      = (fine_to_co_base + 1)[None, :]   # 1-indexed, (npts=1, nk_fi)
        intp_coefs = np.ones((npts, nk_fi))

        print(f"  nk_fi = nk_co = {nk_fi}    "
              f"nc_fi = nc_co = {nc_fi}    nv_fi = nv_co = {nv_fi}")
    else:
        if dtmat_file is None:
            raise ValueError(
                "dtmat_file is required unless wfn_fi_same_wfn_co=True.")
        _require_file(dtmat_file, 'dtmat',
                      'BerkeleyGW dtmat binary from absorption.<flavour>.x; '
                      'or pass --wfn_fi_same_wfn_co if no interpolation is needed')
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

    # ── Consistency checks / warnings ────────────────────────────────────
    if nk_co_dt != Nk_co:
        warnings.warn(
            f"Coarse k-point count mismatch: coarse el-ph has {Nk_co}, dtmat "
            f"has {nk_co_dt}. They may come from different runs. Will match "
            f"k-points by coordinates.")

    if nc_co_avail < nc_co:
        warnings.warn(
            f"dtmat requests {nc_co} coarse conduction bands, but the coarse "
            f"el-ph only has {nc_co_avail}. Using {nc_co_avail} cond bands "
            f"(missing higher bands zero-padded).")
    elif nc_co_avail > nc_co:
        print(f"  NOTE: coarse el-ph has {nc_co_avail} conduction bands but "
              f"dtmat only uses {nc_co} — truncating to the lowest {nc_co} "
              f"cond bands (closest to LUMO).")
    if nv_co_avail < nv_co:
        warnings.warn(
            f"dtmat requests {nv_co} coarse valence bands, but the coarse "
            f"el-ph only has {nv_co_avail}. Using {nv_co_avail} val bands "
            f"(missing lower bands zero-padded).")
    elif nv_co_avail > nv_co:
        print(f"  NOTE: coarse el-ph has {nv_co_avail} valence bands but "
              f"dtmat only uses {nv_co} — truncating to the highest {nv_co} "
              f"val bands (closest to HOMO).")

    if nspin > 1:
        warnings.warn(
            f"nspin={nspin} in dtmat — current implementation uses spin=0 only.")

    # ── Build coarse k-point map (elph -> dtmat) ──────────────────────────
    elph_to_dt_co = _build_kpt_map(kpts_co, kco_dt, tol=tol_k)
    n_matched = np.sum(elph_to_dt_co >= 0)
    if n_matched < Nk_co:
        warnings.warn(
            f"Only {n_matched}/{Nk_co} coarse k-points matched between "
            f"the coarse el-ph and dtmat. Unmatched k-points will be skipped (g_fine=0).")
    print(f"\n  Coarse k-point matching: {n_matched}/{Nk_co} matched")

    # ── Resize the pre-split cond/val blocks to match dtmat's band request ──
    #    (dcn/dvn's coarse dimension is exactly nc_co/nv_co, so g_co_cond/val
    #    must end up with exactly that many bands, whichever direction the
    #    mismatch goes.)
    if nc_co_avail > nc_co:
        g_co_cond = g_co_cond[..., :nc_co, :nc_co]
    elif nc_co_avail < nc_co:
        pad = np.zeros((*g_co_cond.shape[:3], nc_co, nc_co), dtype=g_co_cond.dtype)
        pad[..., :nc_co_avail, :nc_co_avail] = g_co_cond
        g_co_cond = pad

    if nv_co_avail > nv_co:
        g_co_val = g_co_val[..., :nv_co, :nv_co]
    elif nv_co_avail < nv_co:
        pad = np.zeros((*g_co_val.shape[:3], nv_co, nv_co), dtype=g_co_val.dtype)
        pad[..., :nv_co_avail, :nv_co_avail] = g_co_val
        g_co_val = pad

    # ── Handle finite-q: build fine k-point map k_fi -> k_fi+q ───────────
    ik_fi_q_map = np.arange(nk_fi, dtype=int)   # default: q=0, no shift

    _any_finite_q = not np.all(np.abs(qpts_co) < tol_k)

    if _any_finite_q:
        if kpts_fi_crys is None:
            raise ValueError(
                "Fine k-point coordinates are required for finite-q "
                "interpolation but were not found. Provide a valid WFN.h5 "
                "with --wfn_to_interpolate (read from its mf_header/kpoints/rk). "
                "Run 'python elph_xml_to_h5.py -h' to see all options.")
        print(f"\n  Finite-q detected — will build k_fi -> k_fi+q maps per q.")
    else:
        print(f"\n  q = Gamma (all q-points are 0,0,0) — no k-shift needed.")

    # ── Precompute dtmat-coarse -> elph-coarse index map ───────────────────
    dt_to_elph_co = np.full(nk_co_dt, -1, dtype=int)
    for ik_co_dt in range(nk_co_dt):
        match = _build_kpt_map(kco_dt[ik_co_dt][None, :], kpts_co, tol=tol_k)
        dt_to_elph_co[ik_co_dt] = match[0]
    n_co_matched = np.sum(dt_to_elph_co >= 0)
    if n_co_matched < nk_co_dt:
        warnings.warn(
            f"{nk_co_dt - n_co_matched}/{nk_co_dt} dtmat coarse k-points have no "
            f"match in the coarse el-ph grid. Their contributions will be skipped.")
    print(f"  dtmat->elph coarse k-map: {n_co_matched}/{nk_co_dt} matched")

    fi2co_elph = np.full((npts, nk_fi), -1, dtype=int)
    for ivert in range(npts):
        for ik_fi in range(nk_fi):
            ik_co_dt = int(fi2co[ivert, ik_fi]) - 1   # 0-indexed in dtmat
            fi2co_elph[ivert, ik_fi] = dt_to_elph_co[ik_co_dt]

    # ── Allocate output arrays ────────────────────────────────────────────
    elph_fine_cond = np.zeros((Nq, Nmodes, nk_fi, nc_fi, nc_fi), dtype=np.complex128)
    elph_fine_val  = np.zeros((Nq, Nmodes, nk_fi, nv_fi, nv_fi), dtype=np.complex128)

    print(f"\n  Output shapes:")
    print(f"    elph_fine_cond : {elph_fine_cond.shape}  (Nq, Nmodes, Nk_fi, nc_fi, nc_fi)")
    print(f"    elph_fine_val  : {elph_fine_val.shape}  (Nq, Nmodes, Nk_fi, nv_fi, nv_fi)")

    # ── Main interpolation loop ───────────────────────────────────────────
    print(f"\nInterpolating ... ({nk_fi} fine k-points, {Nq} q-points, "
          f"{Nmodes} modes)")

    for iq in range(Nq):

        if _any_finite_q:
            q_crys = qpts_co[iq]                                     # (3,)
            k_fi_q = wrap_to_bz(kpts_fi_crys + q_crys[None, :])      # (Nk_fi, 3)
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

            g_co_cond_block = np.zeros((Nmodes, nc_co, nc_co), dtype=np.complex128)
            g_co_val_block  = np.zeros((Nmodes, nv_co, nv_co), dtype=np.complex128)

            for ivert in range(npts):
                w          = intp_coefs[ivert, ik_fi]     # interpolation weight
                ik_co_elph = fi2co_elph[ivert, ik_fi]     # elph coarse k index
                if ik_co_elph < 0:
                    continue   # coarse k-point not matched; skip vertex
                g_co_cond_block += w * g_co_cond[iq, :, ik_co_elph, :, :]
                g_co_val_block  += w * g_co_val[ iq, :, ik_co_elph, :, :]

            D_right_cond = dcn[:, :, 0, ik_fi,   0]   # (nc_fi, nc_co)
            D_left_cond  = dcn[:, :, 0, ik_fi_q, 0]   # (nc_fi, nc_co) at k_fi+q
            D_right_val  = dvn[:, :, 0, ik_fi,   0]   # (nv_fi, nv_co)
            D_left_val   = dvn[:, :, 0, ik_fi_q, 0]   # (nv_fi, nv_co) at k_fi+q

            # G_fine[n, m] = sum_{a,b} D_left[n,a]^* * G_co[a,b] * D_right[m,b]
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

    print(f"\nDone.")
    print(f"  elph_fine_cond : {elph_fine_cond.shape}")
    print(f"    max|Re| = {np.max(np.abs(np.real(elph_fine_cond))):.4e} Ry/bohr")
    print(f"    max|Im| = {np.max(np.abs(np.imag(elph_fine_cond))):.4e} Ry/bohr")
    print(f"  elph_fine_val  : {elph_fine_val.shape}")
    print(f"    max|Re| = {np.max(np.abs(np.real(elph_fine_val))):.4e} Ry/bohr")
    print(f"    max|Im| = {np.max(np.abs(np.imag(elph_fine_val))):.4e} Ry/bohr")

    return elph_fine_cond, elph_fine_val


# ══════════════════════════════════════════════════════════════════════════════
# QP rescaling: read eqp.dat and build per-k rescaling matrices
# ══════════════════════════════════════════════════════════════════════════════

def _read_eqp_and_build_rescaling(
    eqp_file: str,
    nk: int,
    nc: int,
    nv: int,
    Nval: int,
    tol_deg: float = 1e-5,
):
    """
    Read a fine-grid eqp.dat (output of inteqp.x) and compute QP rescaling matrices.

    The rescaling matrix satisfies:
        ratio[ik, n, m] = (Eqp[ik,n] - Eqp[ik,m]) / (Edft[ik,n] - Edft[ik,m])
    with ratio = 1.0 for degenerate pairs (|Edft difference| < tol_deg).

    Band ordering conventions match elph_fine arrays:
        conduction: ic=0 -> LUMO  (iband_file = Nval+1)
        valence:    iv=0 -> HOMO  (iband_file = Nval)

    Parameters
    ----------
    eqp_file : path to eqp.dat
    nk       : number of fine k-points (= nkpt_fi from dtmat)
    nc       : number of conduction bands (= ncb_fi from dtmat)
    nv       : number of valence bands    (= nvb_fi from dtmat)
    Nval     : number of occupied bands in the DFT/DFPT run (QE nbnd convention)
    tol_deg  : energy threshold (eV) for degenerate-pair masking

    Returns
    -------
    QP_rescaling_cond : (nk, nc, nc) float64
    QP_rescaling_val  : (nk, nv, nv) float64
    Eqp_cond  : (nk, nc) float64  in eV
    Eqp_val   : (nk, nv) float64  in eV
    Edft_cond : (nk, nc) float64  in eV
    Edft_val  : (nk, nv) float64  in eV
    """
    Eqp_cond  = np.zeros((nk, nc))
    Eqp_val   = np.zeros((nk, nv))
    Edft_cond = np.zeros((nk, nc))
    Edft_val  = np.zeros((nk, nv))

    print(f'\nReading QP energies from {eqp_file} ...')
    ik = -1
    with open(eqp_file, 'r') as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] != '1':
                ik += 1
                if ik >= nk:
                    break
            else:
                iband_file = int(parts[1])
                edft = float(parts[2])
                eqp  = float(parts[3])
                if iband_file > Nval:
                    ic = iband_file - Nval - 1   # 0-indexed: ic=0 -> LUMO
                    if ic < nc:
                        Edft_cond[ik, ic] = edft
                        Eqp_cond[ik, ic]  = eqp
                else:
                    iv = Nval - iband_file       # 0-indexed: iv=0 -> HOMO
                    if iv < nv:
                        Edft_val[ik, iv] = edft
                        Eqp_val[ik, iv]  = eqp

    print(f'  Read {ik + 1} k-points; Eqp_cond shape {Eqp_cond.shape}, Eqp_val shape {Eqp_val.shape}')

    def _build_ratio(Eqp, Edft):
        dEqp  = Eqp[:, :, None] - Eqp[:, None, :]    # (nk, nb, nb)
        dEdft = Edft[:, :, None] - Edft[:, None, :]
        mask  = np.abs(dEdft) > tol_deg
        return np.where(mask, dEqp / np.where(mask, dEdft, 1.0), 1.0)

    QP_rescaling_cond = _build_ratio(Eqp_cond, Edft_cond)
    QP_rescaling_val  = _build_ratio(Eqp_val,  Edft_val)
    print(f'  QP_rescaling_cond shape {QP_rescaling_cond.shape}, '
          f'QP_rescaling_val shape {QP_rescaling_val.shape}')

    return QP_rescaling_cond, QP_rescaling_val, Eqp_cond, Eqp_val, Edft_cond, Edft_val


# ══════════════════════════════════════════════════════════════════════════════
# Unified HDF5 writer/reader — shared by elph_orig_kgrid.h5 and elph_interpolated_kgrid.h5
# ══════════════════════════════════════════════════════════════════════════════

def save_elph_h5(
    out_path:            str,
    elph_cond_cart:       np.ndarray,
    elph_val_cart:        np.ndarray,
    elph_cond_mode:       np.ndarray | None = None,
    elph_val_mode:        np.ndarray | None = None,
    kpoints_crystal:      np.ndarray | None = None,
    qpoints_crystal:      np.ndarray | None = None,
    qpoints_cart:         np.ndarray | None = None,
    phonon_modes:         dict | None = None,
    structure:            dict | None = None,
    qp_rescaling:         dict | None = None,
    compress:             bool = True,
    extra_attrs:          dict | None = None,
) -> None:
    """
    Save el-ph arrays to an HDF5 file. Used for BOTH elph_orig_kgrid.h5 and
    elph_interpolated_kgrid.h5 — the two files share an identical dataset schema; only the
    numeric coefficients and k-points differ (coarse vs. fine grid).

    Parameters
    ----------
    elph_cond_cart, elph_val_cart : (Nq, Npert, Nk, Nc/Nv, Nc/Nv) Cartesian basis
    elph_cond_mode, elph_val_mode : (Nq, Nmodes, Nk, Nc/Nv, Nc/Nv) phonon-mode
        basis, or None if matdyn.modes was not available
    kpoints_crystal : (Nk, 3) crystal (fractional) coords — coarse or fine
    qpoints_crystal, qpoints_cart : (Nq, 3) DFPT q-points
    phonon_modes : dict with keys 'qpoints', 'frequencies', 'eigenvectors'
        (from parse_matdyn_modes), or None
    structure : dict with keys 'atomic_numbers', 'atomic_positions_ang',
        'lattice_vectors_ang', 'nat', 'source' (from load_params_from_qe_input
        or read_wfn_h5_header), or None
    qp_rescaling : dict with keys 'cond', 'val', 'Eqp_cond', 'Eqp_val',
        'Edft_cond', 'Edft_val' (from _read_eqp_and_build_rescaling), or None
        — fine-grid only in practice
    extra_attrs : optional dict of additional root-level attributes
        (e.g. 'grid': 'coarse'/'fine', source file paths)
    """
    kw = dict(compression='gzip', compression_opts=4) if compress else {}

    with h5py.File(out_path, 'w') as fh:
        if elph_cond_mode is not None:
            ds = fh.create_dataset('elph_cond_mode', data=elph_cond_mode, **kw)
            ds.attrs['axes'] = 'elph_cond_mode[iq, nu, ik, ic, ic_prime]'
            ds.attrs['note'] = '<c, k+q | dV(q) | c_prime, k>, phonon-mode basis, ic=0 -> LUMO'

            ds = fh.create_dataset('elph_val_mode', data=elph_val_mode, **kw)
            ds.attrs['axes'] = 'elph_val_mode[iq, nu, ik, iv, iv_prime]'
            ds.attrs['note'] = '<v, k+q | dV(q) | v_prime, k>, phonon-mode basis, iv=0 -> HOMO'
        else:
            print("  g_mode not available — skipping phonon-mode basis datasets.")

        ds = fh.create_dataset('elph_cond_cart', data=elph_cond_cart, **kw)
        ds.attrs['axes'] = 'elph_cond_cart[iq, alpha, ik, ic, ic_prime]'
        ds.attrs['note'] = '<c, k+q | dV(q) | c_prime, k>, Cartesian basis, alpha=3*iatom+idir'

        ds = fh.create_dataset('elph_val_cart', data=elph_val_cart, **kw)
        ds.attrs['axes'] = 'elph_val_cart[iq, alpha, ik, iv, iv_prime]'
        ds.attrs['note'] = '<v, k+q | dV(q) | v_prime, k>, Cartesian basis, alpha=3*iatom+idir'

        if kpoints_crystal is not None:
            ds = fh.create_dataset('Kpoints_in_elph_file', data=kpoints_crystal, **kw)
            ds.attrs['axes'] = 'Kpoints_in_elph_file[ik, xyz]'
            ds.attrs['units'] = 'crystal (fractional) coordinates'

        if qpoints_cart is not None:
            fh.create_dataset('qpoints_cart', data=qpoints_cart)
            fh['qpoints_cart'].attrs['coords'] = 'Cartesian 2pi/a'

        if qpoints_crystal is not None:
            fh.create_dataset('qpoints_crystal', data=qpoints_crystal)
            fh['qpoints_crystal'].attrs['coords'] = 'crystal (fractional)'

        if phonon_modes is not None:
            pmg = fh.create_group('phonon_modes')
            pmg.attrs['source'] = 'matdyn.modes'
            pmg.attrs['Nq']     = len(phonon_modes['qpoints'])
            pmg.attrs['Nmodes'] = phonon_modes['frequencies'].shape[1]

            pmg.create_dataset('qpoints', data=phonon_modes['qpoints'])
            pmg['qpoints'].attrs['coords'] = 'Cartesian 2pi/a (as printed by matdyn.x)'

            pmg.create_dataset('frequencies', data=phonon_modes['frequencies'], **kw)
            pmg['frequencies'].attrs['axes']  = 'frequencies[iq, nu]'
            pmg['frequencies'].attrs['units'] = 'cm-1'

            pmg.create_dataset('eigenvectors', data=phonon_modes['eigenvectors'], **kw)
            pmg['eigenvectors'].attrs['axes']  = 'eigenvectors[iq, nu, iat, xyz]'
            pmg['eigenvectors'].attrs['units'] = 'dimensionless (real-space displacement pattern, NOT mass-weighted)'
            pmg['eigenvectors'].attrs['note']  = (
                'Real-space phonon displacement vectors from diagonalisation of the '
                'dynamical matrix, as printed by matdyn.x, then normalised to unit norm. '
                'These are NOT orthogonal in the Euclidean inner product '
                '— orthogonality holds only under the mass-weighted inner product.')

        if structure is not None:
            cg = fh.create_group('crystal')
            cg.attrs['nat']    = structure['nat']
            cg.attrs['source'] = structure['source']
            cg.create_dataset('atomic_numbers', data=structure['atomic_numbers'])
            ds = cg.create_dataset('atomic_positions', data=structure['atomic_positions_ang'])
            ds.attrs['units'] = 'Angstrom (Cartesian)'
            ds = cg.create_dataset('lattice_vectors', data=structure['lattice_vectors_ang'])
            ds.attrs['axes']  = 'lattice_vectors[i, xyz], rows = a1,a2,a3'
            ds.attrs['units'] = 'Angstrom'

        if qp_rescaling is not None:
            ds = fh.create_dataset('QP_rescaling_matrix_cond', data=qp_rescaling['cond'], **kw)
            ds.attrs['axes'] = 'QP_rescaling_matrix_cond[ik, ic1, ic2]'
            ds.attrs['description'] = ('(Eqp_cond[ik,ic1]-Eqp_cond[ik,ic2])/'
                                       '(Edft_cond[ik,ic1]-Edft_cond[ik,ic2]); '
                                       '1.0 for degenerate pairs')

            ds = fh.create_dataset('QP_rescaling_matrix_val', data=qp_rescaling['val'], **kw)
            ds.attrs['axes'] = 'QP_rescaling_matrix_val[ik, iv1, iv2]'
            ds.attrs['description'] = ('(Eqp_val[ik,iv1]-Eqp_val[ik,iv2])/'
                                       '(Edft_val[ik,iv1]-Edft_val[ik,iv2]); '
                                       '1.0 for degenerate pairs')

            ds = fh.create_dataset('Eqp_cond', data=qp_rescaling['Eqp_cond'], **kw)
            ds.attrs['axes'] = 'Eqp_cond[ik, ic]'; ds.attrs['units'] = 'eV'
            ds.attrs['note'] = 'ic=0 -> LUMO'

            ds = fh.create_dataset('Eqp_val', data=qp_rescaling['Eqp_val'], **kw)
            ds.attrs['axes'] = 'Eqp_val[ik, iv]'; ds.attrs['units'] = 'eV'
            ds.attrs['note'] = 'iv=0 -> HOMO'

            ds = fh.create_dataset('Edft_cond', data=qp_rescaling['Edft_cond'], **kw)
            ds.attrs['axes'] = 'Edft_cond[ik, ic]'; ds.attrs['units'] = 'eV'

            ds = fh.create_dataset('Edft_val', data=qp_rescaling['Edft_val'], **kw)
            ds.attrs['axes'] = 'Edft_val[ik, iv]'; ds.attrs['units'] = 'eV'

            print(f"  Saved QP rescaling matrices and energies.")

        fh.attrs['Nq']     = elph_cond_cart.shape[0]
        fh.attrs['Npert']  = elph_cond_cart.shape[1]
        fh.attrs['Nk']     = elph_cond_cart.shape[2]
        fh.attrs['Nc']     = elph_cond_cart.shape[3]
        fh.attrs['Nv']     = elph_val_cart.shape[3]
        if elph_cond_mode is not None:
            fh.attrs['Nmodes'] = elph_cond_mode.shape[1]
        if extra_attrs:
            for k, v in extra_attrs.items():
                fh.attrs[k] = v

    print(f"\nSaved el-ph to: {out_path}")


def load_coarse_elph_h5(path: str) -> dict:
    """
    Load back the split cond/val blocks + k/q-points + phonon_modes from an
    existing elph_orig_kgrid.h5 (written by save_elph_h5), for the "resume:
    interpolate from a previously-assembled coarse file" CLI path (i.e. when
    --elph_dir is not given).
    """
    out = {}
    with h5py.File(path, 'r') as fh:
        out['elph_cond_cart'] = fh['elph_cond_cart'][:]
        out['elph_val_cart']  = fh['elph_val_cart'][:]
        out['elph_cond_mode'] = fh['elph_cond_mode'][:] if 'elph_cond_mode' in fh else None
        out['elph_val_mode']  = fh['elph_val_mode'][:]  if 'elph_val_mode'  in fh else None
        out['kpoints_crystal'] = fh['Kpoints_in_elph_file'][:]
        out['qpoints_crystal'] = fh['qpoints_crystal'][:] if 'qpoints_crystal' in fh else None
        out['qpoints_cart']    = fh['qpoints_cart'][:]    if 'qpoints_cart'    in fh else None
        if 'phonon_modes' in fh:
            out['phonon_modes'] = dict(
                qpoints=fh['phonon_modes/qpoints'][:],
                frequencies=fh['phonon_modes/frequencies'][:],
                eigenvectors=fh['phonon_modes/eigenvectors'][:],
            )
        else:
            out['phonon_modes'] = None
        if 'crystal' in fh:
            out['structure'] = dict(
                nat=int(fh['crystal'].attrs['nat']),
                source=fh['crystal'].attrs['source'],
                atomic_numbers=fh['crystal/atomic_numbers'][:],
                atomic_positions_ang=fh['crystal/atomic_positions'][:],
                lattice_vectors_ang=fh['crystal/lattice_vectors'][:],
            )
        else:
            out['structure'] = None
        out['Nval'] = int(fh.attrs['Nval'])
    return out


def _write_qpoints_dat(qpts_cart, qpts_crystal, suffix: str, here: str) -> None:
    """
    Write qpoints_cart_<suffix>.dat and qpoints_crystal_<suffix>.dat report
    files (suffix = 'co' for the coarse grid, 'fi' for the fine grid).
    """
    if qpts_cart is None or qpts_crystal is None:
        print(f"  q-points not available — skipping qpoints_*_{suffix}.dat.")
        return

    cart_path = os.path.join(here, f'qpoints_cart_{suffix}.dat')
    crys_path = os.path.join(here, f'qpoints_crystal_{suffix}.dat')

    with open(cart_path, 'w') as f:
        f.write('# q-points in Cartesian basis, units: 2*pi/a\n')
        f.write(f'# {"iq":>4}   {"qx":>14}   {"qy":>14}   {"qz":>14}\n')
        for iq, q in enumerate(qpts_cart):
            f.write(f'  {iq+1:4d}   {q[0]:14.9f}   {q[1]:14.9f}   {q[2]:14.9f}\n')
    print(f"Written: {cart_path}")

    with open(crys_path, 'w') as f:
        f.write('# q-points in crystal (fractional) coordinates, units: (b1, b2, b3)\n')
        f.write(f'# {"iq":>4}   {"q1":>14}   {"q2":>14}   {"q3":>14}\n')
        for iq, q in enumerate(qpts_crystal):
            f.write(f'  {iq+1:4d}   {q[0]:14.9f}   {q[1]:14.9f}   {q[2]:14.9f}\n')
    print(f"Written: {crys_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI / main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    cli = argparse.ArgumentParser(
        description='Assemble QE elph XML files into HDF5, and/or interpolate '
                     'to a fine k-grid using BerkeleyGW dtmat.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Necessary arguments: core file paths for a standard run ────────────
    g_necessary = cli.add_argument_group(
        'necessary arguments',
        'Core file paths driving a standard run (assembly + interpolation).')

    g_necessary.add_argument('--elph_dir',
                      help="Directory containing elph.iq.ipert.xml, patterns.iq.xml, and "
                           "control_ph.xml (QE phsave dir, e.g. _ph0/<prefix>.phsave). "
                           "If given, the assembly stage runs and --elph_coarse is (re)written. "
                           "If omitted, --elph_coarse must point to an existing coarse file "
                           "(resume: interpolation-only).")
    g_necessary.add_argument('--modes_file', default='matdyn.modes',
                      help="matdyn.modes (from QE DFPT calculations); optional")
    g_necessary.add_argument('--elph_coarse', default='elph_orig_kgrid.h5',
                      help="Output path for the coarse el-ph file when --elph_dir is given; "
                           "required input path (must already exist) when --elph_dir is omitted.")
    g_necessary.add_argument('--elph_fine', default='elph_interpolated_kgrid.h5',
                      help="Output HDF5 filename for the fine-grid (interpolated) el-ph file.")
    g_necessary.add_argument('--dtmat', default='dtmat',
                      help="Path to dtmat binary file produced by absorption.<flavour>.x. It "
                           "contains the projections <psi_n,kfi|psi_m,kco> and interpolation "
                           "coefficients from coarse to fine grids.")
    g_necessary.add_argument('--wfn_to_interpolate', default='WFN_fi.h5',
                      help="Path to WFN_fi.h5 (needed for finite-q interpolation).")
    g_necessary.add_argument('--eqp', default='eqp.dat',
                      help="Path to fine-grid eqp.dat (output of inteqp.x). When found, "
                           "QP rescaling matrices and Eqp/Edft energies are computed and "
                           "saved into --elph_fine.")

    # ── Alternative arguments: alternate sources/overrides + optional modifiers ──
    g_alt = cli.add_argument_group(
        'alternative arguments',
        'Alternative ways to supply information already covered by the '
        'necessary arguments above, plus optional run modifiers.')

    # Alternative sources for cell/k-points/nbnd/Nval (grouped: pick one, or override)
    g_alt.add_argument('--wfn_origin',
                      help="Path to WFN_co.h5. If given and found, cell, k-points, nbnd, "
                           "and Nval are read directly from its mf_header (authoritative, "
                           "no pseudopotential/scf.in guessing needed). Falls back to "
                           "--qe_input if not given or not found.")
    g_alt.add_argument('--qe_input', default='scf.in',
                      help="QE pw.x input file (scf.in or bands.in), used as a fallback "
                           "source for cell/k-points/nbnd/Nval when --wfn_origin is not given "
                           "or not found.")
    g_alt.add_argument('--Nval', type=int,
                      help="Manual override for Nval (highest occupied band index, QE nbnd "
                           "convention), bypassing both --wfn_origin and --qe_input.")

    # Alternative interpolation mode (replaces the need for --dtmat)
    g_alt.add_argument('--wfn_fi_same_wfn_co', action='store_true',
                      help="Set when WFN_fi == WFN_co, i.e. el-ph coefficients were "
                           "computed directly on the fine grid. The coarse-to-fine "
                           "transformation is taken to be the identity and --dtmat "
                           "is ignored (no dtmat file needed).")

    # Alternative run mode (stop after the coarse stage)
    g_alt.add_argument('--skip-interpolation', dest='skip_interpolation', action='store_true',
                      help="Stop after writing --elph_coarse. --dtmat/--wfn_to_interpolate/--eqp are "
                           "ignored and --elph_fine is NOT written.")

    # Optional toggles
    g_alt.add_argument('--no-ASR', dest='asr', action='store_false', default=True,
                      help="Disable the acoustic sum rule (enforces sum_atoms "
                           "g[iq, 3*atom + d, ik, n, m] = 0 for each Cartesian direction d). "
                           "Only used when --elph_dir is given. The value shown below is the "
                           "default of 'asr' (True = ASR applied); pass --no-ASR to turn it off.")
    g_alt.add_argument('--real', action='store_true',
                      help="Use real-flavor dtmat instead of complex.")

    args = cli.parse_args()

    t0 = datetime.now()
    HERE = os.getcwd()

    # ── Resolve Nval + cell/k-points/nbnd/structure ────────────────────────
    # Only needed in full when the assembly stage runs; when resuming from an
    # existing --elph_coarse, cell/k-points/nbnd aren't needed at all (they're
    # already baked into the coarse file), and Nval/structure fall back to
    # what's stored there if no override is given.
    if args.elph_dir is not None:
        _require_path(args.elph_dir, 'elph_dir',
                      'QE phsave directory, e.g. _ph0/<prefix>.phsave', kind='dir')

        wfn_origin_used = args.wfn_origin is not None and os.path.isfile(args.wfn_origin)
        if wfn_origin_used:
            wfn_info  = read_wfn_h5_header(args.wfn_origin)
            kpts_nscf = wfn_info['kpts_crystal']
            NBNDS     = wfn_info['nbnd']
            rec_vecs  = wfn_info['rec_vecs']
            nat       = wfn_info['nat']
            structure = dict(
                nat=nat, source=f'WFN_co.h5 ({args.wfn_origin})',
                atomic_numbers=wfn_info['atomic_numbers'],
                atomic_positions_ang=wfn_info['atomic_positions_ang'],
                lattice_vectors_ang=wfn_info['lattice_vectors_ang'],
            )
            Nval_auto = wfn_info['Nval']
        else:
            if args.wfn_origin is not None:
                print(f"WARNING: file not found for --wfn_origin: '{args.wfn_origin}'. "
                      f"Falling back to --qe_input. Run 'python elph_xml_to_h5.py "
                      f"-h' to see all options.")
            _require_file(args.qe_input, 'qe_input', 'QE pw.x input file, e.g. scf.in')
            print(f"Reading parameters from {args.qe_input} via ASE ...")
            (nat, _npert, NBNDS, kpts_nscf, _nk, rec_vecs, _bt_inv,
             atomic_numbers, atomic_positions_ang, lattice_vectors_ang) = \
                load_params_from_qe_input(args.qe_input)
            structure = dict(
                nat=nat, source=f'QE input ({args.qe_input})',
                atomic_numbers=atomic_numbers,
                atomic_positions_ang=atomic_positions_ang,
                lattice_vectors_ang=lattice_vectors_ang,
            )
            Nval_auto = get_nval_from_qe_input(args.qe_input)

        bt_inv  = np.linalg.inv(rec_vecs.T)
        NPERT   = 3 * nat
        NK_NSCF = len(kpts_nscf)
        Nval    = args.Nval if args.Nval is not None else Nval_auto
        if args.Nval is not None:
            print(f"Using manually specified --Nval = {Nval} (overrides automatic detection).")

        print(f"  nat    = {nat}")
        print(f"  NPERT  = {NPERT}  (3 x nat, nosym=.true.)")
        print(f"  NBNDS  = {NBNDS}")
        print(f"  NK     = {NK_NSCF}  k-points")
        print(f"  Nval   = {Nval}")
        print(f"  Reciprocal lattice vectors (2pi/alat, rows = b1,b2,b3):")
        for i, b in enumerate(rec_vecs):
            print(f"    b{i+1} = [{b[0]:10.7f}  {b[1]:10.7f}  {b[2]:10.7f}]")

        print(f"\nAll parsed command-line variables:")
        for _k, _v in vars(args).items():
            print(f"  {_k:22s} = {'missing' if _v is None else _v}")

        # ── Assembly stage: XML -> coarse grid ──────────────────────────────
        PHSAVE       = os.path.abspath(args.elph_dir)
        MATDYN_MODES = os.path.abspath(args.modes_file)
        CTRL_PH      = os.path.join(PHSAVE, 'control_ph.xml')
        _require_file(CTRL_PH, 'elph_dir',
                      'control_ph.xml inside the phsave directory')

        print(f"\nReading q-points from {os.path.relpath(CTRL_PH, HERE)} ...")
        qpts_cart = read_qpoints_control_ph(CTRL_PH)
        NQ_LOAD   = len(qpts_cart)
        if NQ_LOAD == 0:
            print("ERROR: no q-points found in control_ph.xml. Aborting.")
            sys.exit(1)
        print(f"  Found {NQ_LOAD} q-points (Cartesian, 2pi/a):")
        for iq, q in enumerate(qpts_cart):
            print(f"    iq={iq+1:2d}:  [{q[0]:10.7f}  {q[1]:10.7f}  {q[2]:10.7f}]")

        g_all = np.zeros((NQ_LOAD, NPERT, NK_NSCF, NBNDS, NBNDS), dtype=np.complex128)

        n_complete = 0
        n_skipped  = 0

        for iq_idx in range(NQ_LOAD):
            iq = iq_idx + 1

            print(f"\n{'='*62}")
            print(f"iq = {iq:2d}  q = [{qpts_cart[iq_idx][0]:9.6f}  "
                  f"{qpts_cart[iq_idx][1]:9.6f}  {qpts_cart[iq_idx][2]:9.6f}]")
            print(f"{'='*62}")

            missing = [ipert for ipert in range(1, NPERT + 1)
                       if not os.path.exists(
                           os.path.join(PHSAVE, f'elph.{iq}.{ipert}.xml'))]
            if missing:
                print(f"  SKIP: missing ipert file(s): {missing}  -> g = 0 for this q.")
                n_skipped += 1
                continue

            patt_path = os.path.join(PHSAVE, f'patterns.{iq}.xml')
            if not os.path.exists(patt_path):
                print(f"  SKIP: missing {os.path.basename(patt_path)}  -> g = 0 for this q.")
                n_skipped += 1
                continue
            U = read_patterns_xml(patt_path, nat)
            dev_cart = np.max(np.abs(U - np.eye(NPERT)))
            print(f"  patterns.{iq}.xml: NPERT x NPERT = {U.shape}, "
                  f"max|U - I| = {dev_cart:.2e} "
                  f"({'~ Cartesian' if dev_cart < 1e-8 else 'symmetry-adapted'})")

            g_pat = np.zeros((NPERT, NK_NSCF, NBNDS, NBNDS), dtype=np.complex128)
            kpt_map_done = False
            kmap = np.full(NK_NSCF, -1, dtype=int)

            for ipert in range(1, NPERT + 1):
                xml_path = os.path.join(PHSAVE, f'elph.{iq}.{ipert}.xml')
                print(f"  ipert={ipert:2d}  {os.path.basename(xml_path)} ...",
                      end=' ', flush=True)
                t_start = datetime.now()

                kpts_xml_cart, g_xml = read_elph_xml(xml_path)
                kpts_xml_crys = cart_to_crystal(kpts_xml_cart, bt_inv)

                if not kpt_map_done:
                    n_missing_k = 0
                    for ik_nscf in range(NK_NSCF):
                        idx = find_kpt_index(kpts_nscf[ik_nscf], kpts_xml_crys)
                        if idx >= 0:
                            kmap[ik_nscf] = idx
                        else:
                            n_missing_k += 1
                            print(f"\n    WARNING: NSCF k[{ik_nscf}] = "
                                  f"{kpts_nscf[ik_nscf]} not found in XML.")
                    n_matched = NK_NSCF - n_missing_k
                    print(f"\n  k-map: {n_matched}/{NK_NSCF} matched  "
                          f"(XML has {len(kpts_xml_cart)} k-pts)")

                    check_extended_zone(kmap, kpts_nscf, kpts_xml_crys, g_xml)

                    kpt_map_done = True

                for ik_nscf in range(NK_NSCF):
                    idx = kmap[ik_nscf]
                    if idx >= 0:
                        g_pat[ipert - 1, ik_nscf] = g_xml[idx]

                dt = (datetime.now() - t_start).total_seconds()
                print(f"  done in {dt:.1f} s")

            g_all[iq_idx] = np.einsum('Pa,Pknm->aknm', np.conj(U), g_pat)
            print(f"  -> transformed to Cartesian basis  "
                  f"(max|delta| from raw pattern values: "
                  f"{np.max(np.abs(g_all[iq_idx] - g_pat)):.2e})")

            n_complete += 1

        print(f"\n{'='*62}")
        print(f"Summary: {n_complete} q-points complete, {n_skipped} skipped (g=0).")

        if args.asr:
            print("\nApplying acoustic sum rule (default; pass --no-ASR to disable).")
            before, after = apply_acoustic_sum_rule(g_all, nat)
            print(f"  max |sum_atoms g_d|  before ASR = {before:.4e} Ry/bohr")
            print(f"  max |sum_atoms g_d|  after  ASR = {after:.4e} Ry/bohr "
                  f"({'OK' if after < 1e-10 else 'WARNING: residual is non-negligible'})")
        else:
            print("\nSkipping acoustic sum rule (--no-ASR set).  g is left as-is.")
            view = g_all.reshape(NQ_LOAD, nat, 3, NK_NSCF, NBNDS, NBNDS)
            residual = float(np.max(np.abs(view.sum(axis=1))))
            print(f"  diagnostic only: max |sum_atoms g_d| = {residual:.4e} Ry/bohr")

        # ── matdyn.modes: phonon frequencies/eigenvectors + Cartesian -> mode projection ──
        phonon_modes_dict = None
        has_g_mode = False
        g_mode = None
        if os.path.isfile(MATDYN_MODES):
            print(f"\nParsing {os.path.relpath(MATDYN_MODES, HERE)} ...")
            modes_qpts, modes_freq, modes_evec = parse_matdyn_modes(MATDYN_MODES, nat)
            nq_modes = len(modes_qpts)
            nmodes   = 3 * nat
            print(f"  Found {nq_modes} q-points, {nmodes} modes/q-point")
            print(f"  Frequency range: {modes_freq.min():.2f} - {modes_freq.max():.2f} cm-1")

            TOL_Q = 1e-5
            elph_to_matdyn = np.full(NQ_LOAD, -1, dtype=int)
            for iq_elph in range(NQ_LOAD):
                q_elph = qpts_cart[iq_elph]
                for iq_md in range(nq_modes):
                    diff = modes_qpts[iq_md] - q_elph
                    if np.linalg.norm(diff) < TOL_Q:
                        elph_to_matdyn[iq_elph] = iq_md
                        break
            n_matched_q   = np.sum(elph_to_matdyn >= 0)
            n_unmatched_q = NQ_LOAD - n_matched_q
            print(f"  q-point matching (elph <-> matdyn.modes): "
                  f"{n_matched_q}/{NQ_LOAD} matched")

            if nq_modes != NQ_LOAD:
                print(f"  WARNING: matdyn.modes has {nq_modes} q-points but elph has "
                      f"{NQ_LOAD}. The two grids do not match.")
            if n_unmatched_q > 0:
                print(f"  WARNING: {n_unmatched_q}/{NQ_LOAD} elph q-point(s) have no "
                      f"matching entry in matdyn.modes — g_mode will be ZERO for these.")

            if n_matched_q == 0:
                print("  WARNING: no q-points matched — g_mode will be all zeros. "
                      "Skipping mode-basis rotation.")
            else:
                # alpha = 3*iat + xyz (atom-major), matching g_all's second axis
                evec_flat = modes_evec.reshape(nq_modes, nmodes, NPERT)
                g_mode = np.zeros((NQ_LOAD, nmodes, NK_NSCF, NBNDS, NBNDS),
                                  dtype=np.complex128)
                for iq_elph in range(NQ_LOAD):
                    iq_md = elph_to_matdyn[iq_elph]
                    if iq_md < 0:
                        continue
                    if np.all(g_all[iq_elph] == 0):
                        continue
                    g_mode[iq_elph] = np.einsum(
                        'va,aknm->vknm', evec_flat[iq_md], g_all[iq_elph])
                has_g_mode = True
                print(f"\n  g_mode shape: {g_mode.shape}  (Nq, Nmodes, Nk, Nbnds, Nbnds)")
                print(f"  max|Re(g_mode)| = {np.max(np.abs(np.real(g_mode))):.4f} Ry/bohr")
                print(f"  max|Im(g_mode)| = {np.max(np.abs(np.imag(g_mode))):.4f} Ry/bohr")

            phonon_modes_dict = dict(qpoints=modes_qpts, frequencies=modes_freq,
                                      eigenvectors=modes_evec)
        else:
            print(f"\nmatdyn.modes not found — skipping phonon eigenvectors.")

        qpts_crystal = cart_to_crystal(qpts_cart, bt_inv)

        # ── Split into cond/val (BGW convention) and write elph_orig_kgrid.h5 ──
        g_cond_cart, g_val_cart = split_cond_val(g_all, Nval)
        g_cond_mode = g_val_mode = None
        if has_g_mode:
            g_cond_mode, g_val_mode = split_cond_val(g_mode, Nval)

        save_elph_h5(
            args.elph_coarse, g_cond_cart, g_val_cart, g_cond_mode, g_val_mode,
            kpoints_crystal=kpts_nscf, qpoints_crystal=qpts_crystal, qpoints_cart=qpts_cart,
            phonon_modes=phonon_modes_dict, structure=structure,
            extra_attrs={'grid': 'coarse', 'Nval': Nval, 'n_complete': n_complete,
                         'n_skipped': n_skipped, 'asr_applied': bool(args.asr),
                         'source_elph_dir': PHSAVE},
        )

        kpts_co_for_interp  = kpts_nscf
        qpts_co_for_interp  = qpts_crystal
        qpts_cart_for_interp = qpts_cart

    else:
        # ── Resume: interpolation-only from an existing --elph_coarse ──────
        _require_file(args.elph_coarse, 'elph_coarse',
                      'existing coarse el-ph file; alternatively pass --elph_dir '
                      'to run the assembly stage from scratch')
        print(f"\n--elph_dir not given: resuming from existing coarse file "
              f"{args.elph_coarse}")
        coarse = load_coarse_elph_h5(args.elph_coarse)
        g_cond_cart = coarse['elph_cond_cart']
        g_val_cart  = coarse['elph_val_cart']
        g_cond_mode = coarse['elph_cond_mode']
        g_val_mode  = coarse['elph_val_mode']
        has_g_mode  = g_cond_mode is not None
        kpts_co_for_interp   = coarse['kpoints_crystal']
        qpts_co_for_interp   = coarse['qpoints_crystal']
        qpts_cart_for_interp = coarse['qpoints_cart']
        phonon_modes_dict    = coarse['phonon_modes']
        structure            = coarse['structure']

        if args.Nval is not None:
            Nval = args.Nval
            print(f"Using manually specified --Nval = {Nval} (overrides value stored "
                  f"in {args.elph_coarse}: {coarse['Nval']}).")
        elif args.wfn_origin is not None and os.path.isfile(args.wfn_origin):
            Nval = read_wfn_h5_header(args.wfn_origin)['Nval']
        elif os.path.isfile(args.qe_input):
            Nval = get_nval_from_qe_input(args.qe_input)
        else:
            Nval = coarse['Nval']
            print(f"  Using Nval = {Nval} stored in {args.elph_coarse} "
                  f"(no --Nval/--wfn_origin/--qe_input available).")

    _write_qpoints_dat(qpts_cart_for_interp, qpts_co_for_interp, 'co', HERE)

    if args.skip_interpolation:
        print(f"\n--skip-interpolation set: stopping after coarse el-ph. "
              f"{args.elph_fine} NOT written.")
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════
    # Interpolation stage (coarse -> fine, via dtmat)
    # ══════════════════════════════════════════════════════════════════════
    kpts_fi = None
    wfn_to_interpolate_path = args.wfn_to_interpolate
    if wfn_to_interpolate_path is None:
        _dtmat_dir = os.path.dirname(os.path.abspath(args.dtmat))
        _candidate = os.path.join(_dtmat_dir, 'WFN_fi.h5')
        if os.path.isfile(_candidate):
            wfn_to_interpolate_path = _candidate
            print(f"Auto-discovered WFN_fi.h5: {wfn_to_interpolate_path}")

    if wfn_to_interpolate_path and os.path.isfile(wfn_to_interpolate_path):
        print(f"\nReading fine k-points from {wfn_to_interpolate_path} ...")
        with h5py.File(wfn_to_interpolate_path, 'r') as fh:
            rk = fh['mf_header/kpoints/rk'][:]  # h5py reads Fortran rk(3,nrk) as (nrk, 3)
            kpts_fi = rk if rk.shape[-1] == 3 else rk.T
        print(f"  {len(kpts_fi)} fine k-points loaded, shape {kpts_fi.shape}")
    elif args.wfn_fi_same_wfn_co:
        print(f"File not found for --wfn_to_interpolate: '{wfn_to_interpolate_path}' — fine k-points "
              f"will default to the coarse grid (--wfn_fi_same_wfn_co).")
        kpts_fi = kpts_co_for_interp
    else:
        print(f"WARNING: file not found for --wfn_to_interpolate: '{wfn_to_interpolate_path}'. "
              f"'Kpoints_in_elph_file' will NOT be saved in the output, and "
              f"finite-q interpolation will fail if any q != Gamma is present. "
              f"Run 'python elph_xml_to_h5.py -h' to see all options.")

    _common = dict(
        kpts_co=kpts_co_for_interp,
        qpts_co=qpts_co_for_interp,
        dtmat_file=None if args.wfn_fi_same_wfn_co else args.dtmat,
        kpts_fi_crys=kpts_fi,
        complex_flavor=not args.real,
        wfn_fi_same_wfn_co=args.wfn_fi_same_wfn_co,
    )

    elph_cond_mode_fi = elph_val_mode_fi = None
    if has_g_mode:
        print("\n" + "="*64)
        print("Interpolating phonon-mode basis (g_mode) ...")
        elph_cond_mode_fi, elph_val_mode_fi = interpolate_elph(
            g_cond_mode, g_val_mode, **_common)
    else:
        print("\nWARNING: phonon-mode basis not available in the coarse el-ph — "
              "skipping. Only Cartesian basis (g) will be saved.")

    print("\n" + "="*64)
    print("Interpolating Cartesian basis (g) ...")
    elph_cond_cart_fi, elph_val_cart_fi = interpolate_elph(
        g_cond_cart, g_val_cart, **_common)

    qp_rescaling = None
    if args.eqp is not None and os.path.isfile(args.eqp):
        nk_fi = elph_cond_cart_fi.shape[2]
        nc_fi = elph_cond_cart_fi.shape[3]
        nv_fi = elph_val_cart_fi.shape[3]
        qp_ratio_c, qp_ratio_v, Eqp_c, Eqp_v, Edft_c, Edft_v = \
            _read_eqp_and_build_rescaling(args.eqp, nk_fi, nc_fi, nv_fi, Nval)
        qp_rescaling = dict(cond=qp_ratio_c, val=qp_ratio_v,
                             Eqp_cond=Eqp_c, Eqp_val=Eqp_v,
                             Edft_cond=Edft_c, Edft_val=Edft_v)
    elif args.eqp is not None:
        print(f"\nNOTE: file not found for --eqp: '{args.eqp}' — skipping QP "
              f"rescaling. Run 'python elph_xml_to_h5.py -h' to see all options.")

    save_elph_h5(
        args.elph_fine, elph_cond_cart_fi, elph_val_cart_fi,
        elph_cond_mode_fi, elph_val_mode_fi,
        kpoints_crystal=kpts_fi, qpoints_crystal=qpts_co_for_interp,
        qpoints_cart=qpts_cart_for_interp, phonon_modes=phonon_modes_dict,
        structure=structure, qp_rescaling=qp_rescaling,
        extra_attrs={'grid': 'fine', 'Nval': Nval,
                     'interpolation': 'BerkeleyGW dtmat coarse-to-fine',
                     'source_elph_coarse': args.elph_coarse,
                     'source_dtmat': ('identity (wfn_fi_same_wfn_co)'
                                       if args.wfn_fi_same_wfn_co else args.dtmat)},
    )

    _write_qpoints_dat(qpts_cart_for_interp, qpts_co_for_interp, 'fi', HERE)

    total_time = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {total_time:.1f} s.")
    print(f"  elph_cond_cart : {elph_cond_cart_fi.shape}")
    print(f"  elph_val_cart  : {elph_val_cart_fi.shape}")
