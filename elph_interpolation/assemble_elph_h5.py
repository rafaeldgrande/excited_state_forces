"""

Usage:
python assemble_elph_h5.py <in.xml> <out.h5>


assemble_elph_h5.py
===================
Read QE DFPT elph XML files, transform to a Cartesian-displacement basis,
and assemble into a single HDF5 file.

System:  MoS₂, QE 7.5, ldisp=.true., electron_phonon='simple'.
         QE writes one elph.iq.ipert.xml per (q, perturbation) where
         "ipert" indexes the symmetry-adapted patterns chosen by ph.x
         (NOT necessarily simple Cartesian unit vectors).

         The displacement patterns U[ipert, alpha] are read from
         _ph0/mos2.phsave/patterns.iq.xml and used to rotate
         <n,k+q|dV/du_pattern_ipert|m,k>  →  <n,k+q|dV/du_cart_alpha|m,k>.

Cell parameters, reciprocal lattice, Nat, and Npert are read from scf.in via ASE.
Nbnds is parsed from scf.in (nbnd = ...).
Nk is taken from the k-points block of scf.in.
The list of q-points and Nq are read from _ph0/mos2.phsave/control_ph.xml
(Q-POINT_COORDINATES, units = 2pi/a).

If ANY ipert file is missing for a given iq, the entire q-point is skipped
and g remains zero for that iq.

Output HDF5 layout
------------------
  g             complex128  (Nq, Npert, Nk, Nbnds, Nbnds)
                g[iq, alpha, ik, ibnd_n, ibnd_m]
                = <n, k+q | dV_SCF/d(tau^{alpha}) | m, k>   [Ry/bohr]
                where alpha runs over CARTESIAN atomic DOFs:
                  alpha = 0,1,2 → atom1-x,y,z
                  alpha = 3,4,5 → atom2-x,y,z
                  ...

  kpoints_dft_crystal  float64  (Nk, 3)  crystal (fractional) coords, from scf.in
  kpoints_dft_cart     float64  (Nk, 3)  Cartesian 2pi/a (computed from crystal × b)
  qpoints_cart         float64  (Nq, 3)  Cartesian 2pi/a, from control_ph.xml
  qpoints_crystal      float64  (Nq, 3)  crystal (fractional) coords (computed)

  phonon_modes/          (group, present if matdyn.modes exists)
    qpoints              float64  (Nq_modes, 3)  Cartesian 2pi/a
    frequencies          float64  (Nq_modes, Nmodes)  cm^{-1}
    eigenvectors         complex128  (Nq_modes, Nmodes, Nat, 3)
                         dynamical-matrix eigenvectors from matdyn.x

  g_mode               complex128  (Nq, Nmodes, Nk, Nbnds, Nbnds)
                         g_mode[iq, nu, ik, ibnd_n, ibnd_m]
                         = sum_{s,alpha} e_{nu,s,alpha}(q) * g_cart[iq,(s,alpha),ik,n,m]
                         El-ph deformation potential projected onto dynamical-matrix
                         eigenvector directions.  Units: Ry/bohr.
                         (present only if matdyn.modes q-points match the elph q-points)

XML formats
-----------
  elph.iq.ipert.xml (same for q=0 and q≠0):
    <PARTIAL_EL_PHON>
      <NUMBER_OF_K>...</NUMBER_OF_K>
      <NUMBER_OF_BANDS>...</NUMBER_OF_BANDS>
      <K_POINT.1>
        <COORDINATES_XK> kx ky kz </COORDINATES_XK>    Cartesian, 2pi/alat
        <PARTIAL_ELPH perturbation="N">
          Re Im                                        per (ibnd, jbnd) pair
          ...
        </PARTIAL_ELPH>
      </K_POINT.1>
      ...
    </PARTIAL_EL_PHON>

  patterns.iq.xml:
    <IRREPS_INFO>
      <NUMBER_IRR_REP>...</NUMBER_IRR_REP>
      <REPRESENTION.k>
        <NUMBER_OF_PERTURBATIONS>...</NUMBER_OF_PERTURBATIONS>
        <PERTURBATION.j>
          <DISPLACEMENT_PATTERN>
            Re Im                  3*nat lines of pairs (Cartesian basis)
            ...
          </DISPLACEMENT_PATTERN>
        </PERTURBATION.j>
        ...
      </REPRESENTION.k>
      ...
    </IRREPS_INFO>

Pattern → Cartesian transform
-----------------------------
For a displacement pattern u_ipert expanded in Cartesian basis as
   delta_tau_alpha = U[ipert, alpha] * lambda_ipert
the chain rule gives
   g_pattern[ipert] = sum_alpha U[ipert, alpha] * g_cart[alpha].
Since U is unitary (orthonormal pattern basis), inverting yields
   g_cart[alpha] = sum_ipert conj(U[ipert, alpha]) * g_pattern[ipert].
This matches QE's own convention (cf. PHonon/PH/elphon.f90).
"""

import numpy as np
import h5py
import xml.etree.ElementTree as ET
import os, re, sys, argparse
from datetime import datetime
from ase.io import read as ase_read

# ── Paths ──────────────────────────────────────────────────────────────────────
# HERE    = os.path.dirname(os.path.abspath(__file__))

import os
HERE = os.getcwd()

PHSAVE  = os.path.join(HERE, '_ph0/mos2.phsave')
SCF_IN  = os.path.join(HERE, 'scf.in')
CTRL_PH = os.path.join(PHSAVE, 'control_ph.xml')

TOL_K   = 1e-5   # tolerance for k-point matching in crystal coords
MATDYN_MODES = os.path.join(HERE, 'matdyn.modes')


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
    #    to be safe.  Shape: (nq, nmodes, nat, 3) → norm over last two axes.
    norms = np.sqrt(
        np.sum(np.abs(eigenvectors) ** 2, axis=(-2, -1), keepdims=True))
    # Guard against zero-norm vectors (shouldn't happen, but avoid div-by-zero)
    norms = np.where(norms < 1e-30, 1.0, norms)
    eigenvectors /= norms

    return qpoints, frequencies, eigenvectors


# ══════════════════════════════════════════════════════════════════════════════
# Parameters from scf.in via ASE + regex
# ══════════════════════════════════════════════════════════════════════════════

def load_params_from_scf_in(scf_in_path):
    """
    Use ASE to read cell geometry and derive reciprocal lattice vectors.
    Parse nbnd and k-points directly from the file.

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
    """
    # ── ASE: cell geometry ──────────────────────────────────────────────────
    atoms = ase_read(scf_in_path, format='espresso-in')
    nat   = len(atoms)
    npert = 3 * nat

    # Lattice constant = |a1| in Angstrom
    alat_ang = np.linalg.norm(atoms.get_cell()[0])

    # Reciprocal lattice vectors in 2pi/alat units:
    # ASE's cell.reciprocal() satisfies b_i · a_j = delta_ij  (no 2pi factor)
    # Multiplying by alat gives units of 2pi/alat.
    rec_vecs = np.array(atoms.get_cell().reciprocal()) * alat_ang
    bt_inv   = np.linalg.inv(rec_vecs.T)

    # ── nbnd: parse from &SYSTEM block ─────────────────────────────────────
    with open(scf_in_path) as f:
        content = f.read()
    m = re.search(r'\bnbnd\s*=\s*(\d+)', content, re.IGNORECASE)
    if m is None:
        raise ValueError("Could not find 'nbnd' in scf.in")
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

    return nat, npert, nbnds, kpts_nscf, nk, rec_vecs, bt_inv


# ══════════════════════════════════════════════════════════════════════════════
# Read q-points from control_ph.xml
# ══════════════════════════════════════════════════════════════════════════════

def read_qpoints_control_ph(path):
    """
    Parse <Q_POINTS> from _ph0/<prefix>.phsave/control_ph.xml.

    Returns
    -------
    qpts_cart : (Nq, 3) float64    Cartesian 2pi/a, in the order ph.x indexes them
                                   (iq = 1 .. Nq → patterns.iq.xml, elph.iq.*.xml)
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

    The XML organises perturbations as REPRESENTION.k → PERTURBATION.j.
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
# k-point utilities
# ══════════════════════════════════════════════════════════════════════════════

def cart_to_crystal(kpts_cart, bt_inv):
    """Convert k-points from Cartesian 2pi/alat to crystal (fractional) coords."""
    return kpts_cart @ bt_inv.T


def wrap_to_bz(k):
    """Wrap crystal coords to [0, 1)."""
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


def check_extended_zone(kmap, kpts_nscf, kpts_xml_crys, g_xml):
    """
    Two checks performed after k-mapping:

    (1) BZ-copy check: verify that kmap[ik] points to the first-BZ copy of
        each NSCF k-point (not an extended-zone image).  With QE ldisp+nosym,
        irrek() appends extended-zone copies *after* the original 36 k-points,
        so find_kpt_index (which iterates in XML order) should always hit the
        first-BZ entry first.  If any match is NOT in the first BZ, we warn.

    (2) |g| consistency check: for k-points that appear more than once in the
        XML (extended-zone copies), verify that |g| is the same at every copy.
        The phase WILL differ between copies (by e^{iG·r}), but the magnitude
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

        kn = kpts_nscf[ik_nscf, :2]
        kx = kpts_xml_crys[idx, :2]

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
            diff_crys = kpts_xml_crys[ij, :2] - kn
            diff_crys -= np.round(diff_crys)        # fold difference to [-0.5, 0.5)
            if np.linalg.norm(diff_crys) < TOL_K * 10:
                # This is another extended-zone copy
                abs_diff = np.max(np.abs(np.abs(g_xml[ij]) - g_ref_abs))
                max_g_diff = max(max_g_diff, abs_diff)
                n_copies_checked += 1

    status_bz = "OK" if n_bad_bz == 0 else f"WARNING: {n_bad_bz} not in 1st BZ"
    print(f"  [BZ-copy check]  {n_in_bz}/{NK_NSCF} matched to 1st-BZ copy → {status_bz}")
    if n_copies_checked > 0:
        print(f"  [|g| consistency] max ||g|_copy - |g|_ref| = {max_g_diff:.2e} Ry/bohr "
              f"over {n_copies_checked} extended-zone copy-pairs "
              f"({'OK — phase varies but |g| is stable' if max_g_diff < 1e-3 else 'WARNING: large discrepancy'})")
    else:
        print(f"  [|g| consistency] no extended-zone copies found (lgamma or all unique)")

    return n_in_bz, max_g_diff


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
# Read one elph.iq.ipert.xml
# ══════════════════════════════════════════════════════════════════════════════

def read_elph_xml(xml_path):
    """
    Parse one elph.iq.ipert.xml (works for q=0 and q≠0).

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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── 0. Parse CLI ──────────────────────────────────────────────────────────
    cli = argparse.ArgumentParser(
        description="Assemble QE elph XML files into HDF5, in Cartesian basis.")
    cli.add_argument(
        '--no-ASR', dest='asr', action='store_false',
        help="Disable the acoustic sum rule (default: ASR is applied so that "
             "sum_atoms g[iq, 3*atom + d, ik, n, m] = 0 for each Cartesian "
             "direction d). Use this to keep the raw, uncorrected couplings.")
    cli.set_defaults(asr=True)
    args = cli.parse_args()

    t0 = datetime.now()

    # ── 1. Load parameters from scf.in ────────────────────────────────────────
    print("Reading parameters from scf.in via ASE ...")
    nat, NPERT, NBNDS, kpts_nscf, NK_NSCF, rec_vecs, bt_inv = \
        load_params_from_scf_in(SCF_IN)

    print(f"  nat    = {nat}")
    print(f"  NPERT  = {NPERT}  (3 × nat, nosym=.true.)")
    print(f"  NBNDS  = {NBNDS}")
    print(f"  NK     = {NK_NSCF}  k-points (from K_POINTS block)")
    print(f"  Reciprocal lattice vectors (2pi/alat, rows = b1,b2,b3):")
    for i, b in enumerate(rec_vecs):
        print(f"    b{i+1} = [{b[0]:10.7f}  {b[1]:10.7f}  {b[2]:10.7f}]")

    # ── 2. Read q-points from control_ph.xml ──────────────────────────────────
    print(f"\nReading q-points from {os.path.relpath(CTRL_PH, HERE)} ...")
    qpts_cart = read_qpoints_control_ph(CTRL_PH)
    NQ_LOAD   = len(qpts_cart)
    if NQ_LOAD == 0:
        print("ERROR: no q-points found in control_ph.xml. Aborting.")
        sys.exit(1)
    print(f"  Found {NQ_LOAD} q-points (Cartesian, 2pi/a):")
    for iq, q in enumerate(qpts_cart):
        print(f"    iq={iq+1:2d}:  [{q[0]:10.7f}  {q[1]:10.7f}  {q[2]:10.7f}]")

    OUT_H5 = os.path.join(HERE, f'elph.h5')

    # ── 3. Allocate output (Cartesian basis, initialised to zero) ─────────────
    g_all = np.zeros((NQ_LOAD, NPERT, NK_NSCF, NBNDS, NBNDS), dtype=np.complex128)

    # ── 4. Loop over q-points ──────────────────────────────────────────────────
    n_complete = 0   # q-points with all files present
    n_skipped  = 0   # q-points with at least one missing file

    for iq_idx in range(NQ_LOAD):
        iq = iq_idx + 1   # 1-indexed filename prefix

        print(f"\n{'='*62}")
        print(f"iq = {iq:2d}  q = [{qpts_cart[iq_idx][0]:9.6f}  "
              f"{qpts_cart[iq_idx][1]:9.6f}  {qpts_cart[iq_idx][2]:9.6f}]")
        print(f"{'='*62}")

        # Check that ALL ipert files exist before touching g_all
        missing = [ipert for ipert in range(1, NPERT + 1)
                   if not os.path.exists(
                       os.path.join(PHSAVE, f'elph.{iq}.{ipert}.xml'))]
        if missing:
            print(f"  SKIP: missing ipert file(s): {missing}  → g = 0 for this q.")
            n_skipped += 1
            continue   # g_all[iq_idx] stays zero

        # ── 4a. Read displacement patterns for this q ─────────────────────────
        patt_path = os.path.join(PHSAVE, f'patterns.{iq}.xml')
        if not os.path.exists(patt_path):
            print(f"  SKIP: missing {os.path.basename(patt_path)}  → g = 0 for this q.")
            n_skipped += 1
            continue
        U = read_patterns_xml(patt_path, nat)
        # Diagnose how far this pattern basis is from plain Cartesian
        dev_cart = np.max(np.abs(U - np.eye(NPERT)))
        print(f"  patterns.{iq}.xml: NPERT×NPERT = {U.shape}, "
              f"max|U - I| = {dev_cart:.2e} "
              f"({'≈ Cartesian' if dev_cart < 1e-8 else 'symmetry-adapted'})")

        # ── 4b. Read all elph files in pattern basis ──────────────────────────
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

            # Build k-point map once per q (same for all ipert)
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

                # Extended-zone checks (uses g_xml from this first ipert)
                check_extended_zone(kmap, kpts_nscf, kpts_xml_crys, g_xml)

                kpt_map_done = True

            # Pack pattern-basis g into temporary buffer
            for ik_nscf in range(NK_NSCF):
                idx = kmap[ik_nscf]
                if idx >= 0:
                    g_pat[ipert - 1, ik_nscf] = g_xml[idx]

            dt = (datetime.now() - t_start).total_seconds()
            print(f"  done in {dt:.1f} s")

        # ── 4c. Pattern → Cartesian transform ──────────────────────────────────
        #   g_cart[alpha, k, n, m] = sum_ipert conj(U[ipert, alpha]) * g_pat[ipert, k, n, m]
        g_all[iq_idx] = np.einsum('Pa,Pknm->aknm', np.conj(U), g_pat)
        print(f"  → transformed to Cartesian basis  "
              f"(max|Δ| from raw pattern values: "
              f"{np.max(np.abs(g_all[iq_idx] - g_pat)):.2e})")

        n_complete += 1

    # ── 5. Acoustic sum rule (Cartesian basis) ────────────────────────────────
    print(f"\n{'='*62}")
    print(f"Summary: {n_complete} q-points complete, {n_skipped} skipped (g=0).")

    if args.asr:
        print("\nApplying acoustic sum rule (default; pass --no-ASR to disable).")
        print("  Enforcing  sum_atoms <n,k+q | dV/d(tau^{atom,d}) | m,k> = 0")
        print("  for each Cartesian direction d ∈ {x, y, z}.")
        print("  Method: for each (iq, d, ik, n, m), compute")
        print("    S_d = sum_atoms g[iq, 3*atom + d, ik, n, m]")
        print("  and subtract S_d / nat from every atom's slice, so the new")
        print("  per-direction sum over atoms vanishes.")
        before, after = apply_acoustic_sum_rule(g_all, nat)
        print(f"  max |sum_atoms g_d|  before ASR = {before:.4e} Ry/bohr")
        print(f"  max |sum_atoms g_d|  after  ASR = {after:.4e} Ry/bohr "
              f"({'OK' if after < 1e-10 else 'WARNING: residual is non-negligible'})")
    else:
        print("\nSkipping acoustic sum rule (--no-ASR set).  g is left as-is.")
        # Report the residual so the user knows the magnitude of the violation
        view = g_all.reshape(NQ_LOAD, nat, 3, NK_NSCF, NBNDS, NBNDS)
        residual = float(np.max(np.abs(view.sum(axis=1))))
        print(f"  diagnostic only: max |sum_atoms g_d| = {residual:.4e} Ry/bohr")

    # ── 6. Parse matdyn.modes (phonon frequencies + eigenvectors) ────────────
    has_modes = False
    if os.path.isfile(MATDYN_MODES):
        print(f"\nParsing {os.path.relpath(MATDYN_MODES, HERE)} ...")
        modes_qpts, modes_freq, modes_evec = parse_matdyn_modes(MATDYN_MODES, nat)
        has_modes = True
        nq_modes  = len(modes_qpts)
        nmodes    = 3 * nat
        print(f"  Found {nq_modes} q-points, {nmodes} modes/q-point")
        print(f"  Frequency range: {modes_freq.min():.2f} – {modes_freq.max():.2f} cm⁻¹")
        print(f"  Eigenvector array shape: {modes_evec.shape}  "
              f"(Nq, Nmodes, Nat, 3)  dtype={modes_evec.dtype}")
    else:
        print(f"\nmatdyn.modes not found — skipping phonon eigenvectors.")

    # ── 6b. Cartesian → phonon-mode projection of el-ph couplings ─────────────
    #
    # matdyn.x prints real-space displacement eigenvectors e_{nu,s,alpha}(q).
    # These are the eigenvectors of the dynamical matrix D(q), but printed
    # in the UN-mass-weighted (real-space) basis:
    #
    #   D_{s alpha, s' beta}(q) = (1/sqrt(M_s M_{s'})) * C_{s alpha, s' beta}(q)
    #
    # The dynamical-matrix eigenvectors (in mass-weighted coordinates) ARE
    # orthonormal, but the real-space displacements printed by matdyn.x are
    # NOT orthogonal — they differ from the mass-weighted eigenvectors by
    # factors of sqrt(M_s).  Orthogonality holds only in the mass-weighted
    # inner product:  sum_{s,alpha} M_s * e*_{nu,s,alpha} e_{mu,s,alpha} = const * delta_{nu,mu}.
    #
    # The contraction performed here is simply a projection onto the
    # real-space displacement patterns:
    #
    #   g_mode[iq, nu, ik, n, m]
    #       = sum_{s, alpha} e_{nu, s, alpha}(q) * g_cart[iq, (s,alpha), ik, n, m]
    #
    # where g_cart = <n,k+q | dV_SCF/d(tau^{s,alpha}) | m,k>   [Ry/bohr].
    #
    # Because the real-space eigenvectors are NOT orthonormal, this is NOT
    # a unitary rotation — the Frobenius norm of g is NOT preserved.
    #
    # To obtain the standard el-ph vertex g^{std}_{mn,nu}(k,q) one needs
    # the mass-weighted eigenvectors ẽ_{nu,s,alpha} = e_{nu,s,alpha} * sqrt(M_s)
    # (which ARE orthonormal) and the 1/sqrt(2 omega_nu) prefactor:
    #
    #   g^{std}_{mn,nu}(k,q) = (1 / sqrt(2 omega_nu))
    #                          * sum_{s,alpha} ẽ_{nu,s,alpha} / M_s
    #                          * <n,k+q | dV_SCF/d(tau^{s,alpha}) | m,k>
    #
    # See the attrs['note'] on the HDF5 dataset for the precise definition
    # of what is stored.
    #
    has_g_mode = False
    if has_modes:
        print(f"\n{'─'*62}")
        print("Rotating el-ph couplings: Cartesian → phonon-mode basis")
        print(f"{'─'*62}")

        # ── Match matdyn q-points to the elph q-point list ────────────────
        # Both are Cartesian 2pi/a.  Build a map: elph_iq_idx → matdyn_iq_idx
        TOL_Q = 1e-5
        elph_to_matdyn = np.full(NQ_LOAD, -1, dtype=int)
        for iq_elph in range(NQ_LOAD):
            q_elph = qpts_cart[iq_elph]
            for iq_md in range(nq_modes):
                diff = modes_qpts[iq_md] - q_elph
                if np.linalg.norm(diff) < TOL_Q:
                    elph_to_matdyn[iq_elph] = iq_md
                    break

        n_matched_q = np.sum(elph_to_matdyn >= 0)
        print(f"  q-point matching (elph ↔ matdyn.modes): "
              f"{n_matched_q}/{NQ_LOAD} matched")
        for iq_elph in range(NQ_LOAD):
            iq_md = elph_to_matdyn[iq_elph]
            q = qpts_cart[iq_elph]
            if iq_md >= 0:
                print(f"    elph iq={iq_elph+1} → matdyn iq={iq_md+1}  "
                      f"q=[{q[0]:9.6f} {q[1]:9.6f} {q[2]:9.6f}]")
            else:
                print(f"    elph iq={iq_elph+1} → NOT MATCHED  "
                      f"q=[{q[0]:9.6f} {q[1]:9.6f} {q[2]:9.6f}]")

        if n_matched_q == 0:
            print("  WARNING: no q-points matched — skipping mode-basis rotation.")
        else:
            # ── Reshape eigenvectors: (Nq, nmodes, nat, 3) → (Nq, nmodes, 3*nat)
            # The flat alpha index must be atom-major: alpha = 3*iat + xyz,
            # which is what .reshape(..., 3*nat) does when the last two axes
            # are (nat, 3) in C-contiguous order — i.e. iat is the slow index
            # and xyz is the fast index.  This matches g_all's second axis.
            evec_flat = modes_evec.reshape(nq_modes, nmodes, NPERT)

            # ── Perform the projection for each matched q-point ───────────
            # g_mode[iq, nu, ik, n, m] = sum_alpha e[nu, alpha] * g_cart[iq, alpha, ik, n, m]
            g_mode = np.zeros((NQ_LOAD, nmodes, NK_NSCF, NBNDS, NBNDS),
                              dtype=np.complex128)

            for iq_elph in range(NQ_LOAD):
                iq_md = elph_to_matdyn[iq_elph]
                if iq_md < 0:
                    continue  # no matching eigenvectors; g_mode stays zero
                if np.all(g_all[iq_elph] == 0):
                    continue  # skipped q-point; g_mode stays zero

                # einsum: e[nu, alpha] * g[alpha, k, n, m] → g_mode[nu, k, n, m]
                g_mode[iq_elph] = np.einsum(
                    'va,aknm->vknm',
                    evec_flat[iq_md],
                    g_all[iq_elph])

            has_g_mode = True

            # ── Diagnostics ───────────────────────────────────────────────
            print(f"\n  g_mode shape: {g_mode.shape}  "
                  f"(Nq, Nmodes, Nk, Nbnds, Nbnds)")
            print(f"  max|Re(g_mode)| = "
                  f"{np.max(np.abs(np.real(g_mode))):.4f} Ry/bohr")
            print(f"  max|Im(g_mode)| = "
                  f"{np.max(np.abs(np.imag(g_mode))):.4f} Ry/bohr")

    # ── 7. Save to HDF5 ───────────────────────────────────────────────────────
    print(f"\nSaving to {OUT_H5} ...")

    with h5py.File(OUT_H5, 'w') as fh:
        ds = fh.create_dataset('g', data=g_all,
                               compression='gzip', compression_opts=4)
        ds.attrs['axes']  = 'g[iq, alpha_cart, ik, ibnd_n, ibnd_m]'
        ds.attrs['units'] = 'Ry/bohr'
        ds.attrs['basis'] = 'Cartesian atomic displacements (rotated from QE pattern basis via patterns.iq.xml)'
        ds.attrs['note']  = ('<n,k+q | dV_SCF/d(tau^{alpha}) | m,k> with alpha = '
                             'Cartesian atomic DOF: 0,1,2 = atom1-x,y,z; 3,4,5 = atom2-x,y,z; ...')

        # ── k-points: store in both crystal and Cartesian (2pi/a) ─────────
        # k_cart = k_crystal · B,  rows of B (= rec_vecs) are b1,b2,b3 (2pi/a)
        kpts_nscf_cart = kpts_nscf @ rec_vecs

        fh.create_dataset('kpoints_dft_crystal', data=kpts_nscf)
        fh['kpoints_dft_crystal'].attrs['coords'] = 'crystal (fractional), from scf.in K_POINTS block'

        fh.create_dataset('kpoints_dft_cart', data=kpts_nscf_cart)
        fh['kpoints_dft_cart'].attrs['coords']    = 'Cartesian 2pi/a (computed from crystal × reciprocal lattice)'

        # ── q-points: store in both Cartesian and crystal ─────────────────
        # qpts_cart from control_ph.xml is already 2pi/a; convert to crystal:
        qpts_crystal = cart_to_crystal(qpts_cart, bt_inv)

        fh.create_dataset('qpoints_cart', data=qpts_cart)
        fh['qpoints_cart'].attrs['coords']    = 'Cartesian 2pi/a, from _ph0/.../control_ph.xml'

        fh.create_dataset('qpoints_crystal', data=qpts_crystal)
        fh['qpoints_crystal'].attrs['coords'] = 'crystal (fractional), computed from Cartesian via (B^T)^{-1}'

        # ── phonon_modes group (from matdyn.modes) ──────────────────────
        if has_modes:
            pmg = fh.create_group('phonon_modes')
            pmg.attrs['source'] = 'matdyn.modes'
            pmg.attrs['Nq']     = nq_modes
            pmg.attrs['Nmodes'] = nmodes
            pmg.attrs['nat']    = nat

            pmg.create_dataset('qpoints', data=modes_qpts)
            pmg['qpoints'].attrs['coords'] = 'Cartesian 2pi/a (as printed by matdyn.x)'

            pmg.create_dataset('frequencies', data=modes_freq,
                               compression='gzip', compression_opts=4)
            pmg['frequencies'].attrs['axes']  = 'frequencies[iq, nu]'
            pmg['frequencies'].attrs['units'] = 'cm-1'

            pmg.create_dataset('eigenvectors', data=modes_evec,
                               compression='gzip', compression_opts=4)
            pmg['eigenvectors'].attrs['axes']  = 'eigenvectors[iq, nu, iat, xyz]'
            pmg['eigenvectors'].attrs['units'] = 'dimensionless (real-space displacement pattern, NOT mass-weighted)'
            pmg['eigenvectors'].attrs['normalization'] = (
                'Each eigenvector is normalised to unit norm in the 3*Nat-dimensional '
                'complex space: sum_{s,alpha} |e_{nu,s,alpha}|^2 = 1 for every (iq, nu).')
            pmg['eigenvectors'].attrs['note']  = (
                'Real-space phonon displacement vectors from diagonalisation of the '
                'dynamical matrix, as printed by matdyn.x, then normalised to unit norm.  '
                'These are NOT orthogonal in the Euclidean inner product '
                '— orthogonality holds only under the mass-weighted inner product '
                'sum_{s,alpha} M_s e*_{nu,s,alpha} e_{mu,s,alpha}.  '
                'eigenvectors[iq, nu, iat, :] gives the (x,y,z) displacement of '
                'atom iat in mode nu at q-point iq.')

        # ── el-ph couplings in phonon-mode basis ─────────────────────────
        if has_g_mode:
            ds_gm = fh.create_dataset('g_mode', data=g_mode,
                                      compression='gzip', compression_opts=4)
            ds_gm.attrs['axes']  = 'g_mode[iq, nu, ik, ibnd_n, ibnd_m]'
            ds_gm.attrs['units'] = 'Ry/bohr'
            ds_gm.attrs['basis'] = 'phonon real-space displacement basis (NOT unitary — see note)'
            ds_gm.attrs['definition'] = (
                'g_mode[iq, nu, ik, n, m] = sum_{s,alpha} e_{nu,s,alpha}(q) '
                '* g_cart[iq, (s,alpha), ik, n, m]  where e are the real-space '
                'displacement eigenvectors printed by matdyn.x.')
            ds_gm.attrs['note'] = (
                'This is the el-ph deformation potential projected onto the '
                'real-space displacement patterns from matdyn.x.  IMPORTANT: '
                'the real-space eigenvectors e_{nu,s,alpha} are NOT orthonormal '
                '— they are eigenvectors of the dynamical matrix D(q) but '
                'printed without the 1/sqrt(M_s) mass-weighting that makes '
                'them orthogonal.  Orthogonality holds only in the mass-weighted '
                'inner product: sum_{s,alpha} M_s e*_{nu,s,alpha} e_{mu,s,alpha} '
                '∝ delta_{nu,mu}.  Therefore this projection is NOT a unitary '
                'rotation and the Frobenius norm of g is NOT preserved.  '
                'To obtain the standard el-ph vertex g^{std}_{mn,nu}(k,q), '
                'one needs the mass-weighted eigenvectors '
                'ẽ_{nu,s,alpha} = e_{nu,s,alpha} * sqrt(M_s) (which ARE '
                'orthonormal) and the phonon frequency prefactor: '
                'g^{std} = (1/sqrt(2 omega_nu)) * '
                'sum_{s,alpha} (ẽ_{nu,s,alpha} / M_s) * g_cart_{(s,alpha),mn}.')
            ds_gm.attrs['eigenvector_source'] = 'phonon_modes/eigenvectors (matdyn.modes)'
            ds_gm.attrs['g_cart_source'] = 'g (Cartesian basis, after ASR if applied)'

        fh.attrs['Nq']          = NQ_LOAD
        fh.attrs['Npert']       = NPERT
        fh.attrs['Nk']          = NK_NSCF
        fh.attrs['Nbnds']       = NBNDS
        fh.attrs['nat']         = nat
        fh.attrs['basis']       = 'Cartesian'
        fh.attrs['ipert_note']  = ('Second axis indexes Cartesian atomic DOFs (0-indexed in HDF5): '
                                   '0=atom1-x, 1=atom1-y, 2=atom1-z, 3=atom2-x, ...')
        fh.attrs['n_complete']  = n_complete
        fh.attrs['n_skipped']   = n_skipped
        fh.attrs['asr_applied'] = bool(args.asr)
        fh.attrs['asr_note']    = (
            'Acoustic sum rule: sum_atoms g[iq, 3*atom + d, ik, n, m] = 0 for d in {x,y,z}. '
            'When applied, the per-direction mean over atoms is subtracted from each atom.')

    total_time = (datetime.now() - t0).total_seconds()
    print(f"Done in {total_time:.1f} s.")
    print(f"\nOutput: {OUT_H5}")
    print(f"  g shape : {g_all.shape}  (Nq, Npert, Nk, Nbnds, Nbnds)")
    print(f"  dtype   : {g_all.dtype}")
    print(f"  max|Re| : {np.max(np.abs(np.real(g_all))):.4f} Ry/bohr")
    print(f"  max|Im| : {np.max(np.abs(np.imag(g_all))):.4f} Ry/bohr")
    zero_qs = [iq+1 for iq in range(NQ_LOAD) if np.all(g_all[iq] == 0)]
    print(f"  Zero q-points (skipped): iq = {zero_qs}")
    if has_modes:
        print(f"\n  phonon_modes/")
        print(f"    qpoints      : {modes_qpts.shape}")
        print(f"    frequencies   : {modes_freq.shape}  (cm⁻¹)")
        print(f"    eigenvectors  : {modes_evec.shape}  (Nq, Nmodes, Nat, 3)")
    if has_g_mode:
        print(f"\n  g_mode shape : {g_mode.shape}  (Neq, Nmodes, Nk, Nbnds, Nbnds)")
        print(f"  g_mode dtype : {g_mode.dtype}")
        print(f"  g_mode max|Re| : {np.max(np.abs(np.real(g_mode))):.4f} Ry/bohr")
        print(f"  g_mode max|Im| : {np.max(np.abs(np.imag(g_mode))):.4f} Ry/bohr")
        zero_mode_qs = [iq+1 for iq in range(NQ_LOAD) if np.all(g_mode[iq] == 0)]
        print(f"  g_mode zero q-points: iq = {zero_mode_qs}")
