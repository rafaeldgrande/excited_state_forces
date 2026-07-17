
import argparse
import numpy as np
from ase.io import read
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors

'''
Build a file with excited-state forces drawn as vectors, to be visualized in
XCrySDen (.axsf) or VESTA (.vesta).

Reads atomic positions and cell vectors from a Quantum ESPRESSO input file
(via ASE) and per-atom excited-state forces from a forces file, then writes
either:
  - an XCrySDen AXSF file (PRIMVEC / PRIMCOORD blocks with a force vector
    appended to each atom line); or
  - a VESTA file (CELLP / STRUC / VECTR / VECTT blocks), since VESTA does not
    read force vectors from XSF/AXSF's trailing-column convention -- that is
    XCrySDen-specific. VESTA vectors are a native .vesta feature normally
    used for spin/displacement vectors, repurposed here for forces.

Usage:
    python visualize_forces.py scf_input_file forces_file flavor_forces output_file [--format {axsf,vesta}]

flavor_forces:
    1 -> RPA_diag
    2 -> RPA_diag_offdiag
    3 -> RPA_diag + Kernel
'''


def get_atoms_from_qe_file(file_scf_input):
    atoms = read(file_scf_input, format="espresso-in")
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    ATOMS = [
        f"{symbol}   {x:.10f}   {y:.10f}   {z:.10f}    "
        for symbol, (x, y, z) in zip(symbols, positions)
    ]
    CELL_LATT = [
        f"{vx:.10f}   {vy:.10f}   {vz:.10f}" for vx, vy, vz in atoms.get_cell()
    ]
    return ATOMS, CELL_LATT


def read_excited_forces(excited_state_forces_file, flavor_forces):
    data = np.genfromtxt(excited_state_forces_file, dtype=complex, usecols=flavor_forces + 1)

    max_imag_part = np.max(np.abs(np.imag(data)))
    if max_imag_part > 1e-6:
        print('Warning: Imaginary part of forces is non-zero. Just considering the real part!')

    data = np.real(data.reshape(-1, 3))

    return data


def write_axsf(output_file, atoms_qe, cell_latt, forces):
    Nat = len(atoms_qe)
    with open(output_file, "w") as arq_out:
        arq_out.write("""ANIMSTEPS  1
CRYSTAL
PRIMVEC \n""")

        for ilat in range(3):
            arq_out.write(f"""    {cell_latt[ilat]}\n""")

        arq_out.write(f"""PRIMCOORD    1
{Nat}   1\n""")

        for iatom in range(Nat):
            fx, fy, fz = forces[iatom]
            arq_out.write(f"""    {atoms_qe[iatom]}  {fx}   {fy}    {fz}\n""")


def write_vesta(output_file, atoms, forces, title="excited state forces"):
    """
    Write a VESTA file with the crystal structure and per-atom force vectors
    (VECTR / VECTT blocks). VESTA has no XSF-style force convention of its
    own, so this uses the same mechanism normally used to plot spin /
    displacement vectors on atoms.
    """
    symbols = atoms.get_chemical_symbols()
    frac = atoms.get_scaled_positions(wrap=False)
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    Nat = len(atoms)
    labels = [f"{sym}{i + 1}" for i, sym in enumerate(symbols)]

    max_force = np.max(np.linalg.norm(forces, axis=1))
    if max_force < 1e-12:
        max_force = 1.0

    with open(output_file, "w") as f:
        f.write("#VESTA_FORMAT_VERSION 3.5.4\n\n\nCRYSTAL\n\n")
        f.write(f"TITLE\n{title}\n\n")

        f.write("GROUP\n1 1 P 1\n")
        f.write("SYMOP\n")
        f.write(" 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1   1\n")
        f.write(" -1.0 -1.0 -1.0  0  0\n")

        f.write("CELLP\n")
        f.write(f"  {a:.6f}  {b:.6f}  {c:.6f}  {alpha:.6f}  {beta:.6f}  {gamma:.6f}\n")
        f.write("  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n")

        f.write("STRUC\n")
        for i in range(Nat):
            x, y, z = frac[i]
            f.write(
                f"  {i + 1} {symbols[i]:<8s} {labels[i]:<6s} 1.0000   "
                f"{x:.6f}   {y:.6f}   {z:.6f}    1a       1\n"
            )
            f.write("                            0.000000   0.000000   0.000000  0.00\n")
        f.write("  0 0 0 0 0 0 0\n")

        f.write("THERI 0\n")
        for i in range(Nat):
            f.write(f"  {i + 1} {labels[i]:<6s} 0.000000\n")
        f.write("  0 0 0 0 0 0\n")

        f.write("SHAPE\n")
        f.write("  0       0       0       0   0.000000  0    0    0    0    0    0    0    0\n")
        f.write("BOUND\n")
        f.write("       0        1        0        1        0        1\n")
        f.write("  0   0   0   0  0\n")
        f.write("SBOND\n  0 0 0 0\n")

        f.write("SITET\n")
        for i in range(Nat):
            z = atomic_numbers[symbols[i]]
            r_rad = covalent_radii[z]
            rc, gc, bc = (jmol_colors[z] * 255).astype(int)
            f.write(
                f"  {i + 1} {labels[i]:<6s} {r_rad:.4f} {rc} {gc} {bc} {rc} {gc} {bc} 204  0\n"
            )
        f.write("  0 0 0 0 0 0\n")

        # Vectors: one definition per atom (scaled to a visually sane length),
        # assigned to that same atom.
        f.write("VECTR\n")
        for i in range(Nat):
            fx, fy, fz = forces[i]
            f.write(f"   {i + 1}   {fx:.6f}   {fy:.6f}   {fz:.6f}    0\n")
            f.write(f"   {i + 1}    0    0    0    0\n")
            f.write(" 0 0 0 0 0\n")
        f.write(" 0 0 0 0 0\n")

        # Trailing flag on the VECTT line is VESTA's "penetrate atoms"
        # switch: 2 = vector starts at the atom's sphere surface (summed
        # with its radius) instead of piercing through the center; 1 = the
        # default, penetrating behavior. (Determined empirically by editing
        # one vector's properties in VESTA and diffing the saved file.)
        f.write("VECTT\n")
        for i in range(Nat):
            f.write(f"   {i + 1}  0.500000 255    0    0 2\n")
        f.write("  0 0 0 0 0\n")

        f.write("SPLAN\n  0    0    0    0\n")
        f.write("LBLAT\n -1\n")
        f.write("LBLSP\n -1\n")
        f.write("DLATM\n -1\n")
        f.write("DLBND\n -1\n")
        f.write("DLPLY\n -1\n")
        f.write("GLBL\n 0.000000\n")

        f.write("ATOMT\n")
        seen = {}
        for i in range(Nat):
            sym = symbols[i]
            if sym in seen:
                continue
            seen[sym] = True
            z = atomic_numbers[sym]
            r_rad = covalent_radii[z]
            rc, gc, bc = (jmol_colors[z] * 255).astype(int)
            f.write(f"   {len(seen)}        {sym:<6s} {r_rad:.4f} {rc}  {gc}  {bc}  {rc}  {gc}  {bc} 204\n")

        f.write("SCENE\n")
        f.write("  0.000000  0.000000  0.000000  0.000000\n")
        f.write("  0.000000  0.000000  0.000000  0.000000\n")
        f.write("  0.000000  0.000000  0.000000  0.000000\n")
        f.write("  0.000000  0.000000  0.000000  1.000000\n")
        f.write("  0.000  0.000\n")
        f.write("  1.000\n")
        f.write("  0.000000  0.000000  0.000000  0.000000\n")
        f.write("HBOND 0 2\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a file with excited-state forces, to be visualized in XCrySDen (.axsf) or VESTA (.vesta)."
    )
    parser.add_argument("scf_input_file", help="Quantum ESPRESSO input file with atomic positions and cell")
    parser.add_argument("forces_file", help="File with excited-state forces")
    parser.add_argument(
        "flavor_forces",
        type=int,
        choices=[1, 2, 3],
        help="1 = RPA_diag, 2 = RPA_diag_offdiag, 3 = RPA_diag + Kernel",
    )
    parser.add_argument("output_file", help="Path to write the output file")
    parser.add_argument(
        "--format", choices=["axsf", "vesta"], default="axsf",
        help="Output format: 'axsf' for XCrySDen (default), 'vesta' for VESTA",
    )
    args = parser.parse_args()

    forces = read_excited_forces(args.forces_file, args.flavor_forces)

    if args.format == "axsf":
        ATOMS, CELL_LATT = get_atoms_from_qe_file(args.scf_input_file)
        write_axsf(args.output_file, ATOMS, CELL_LATT, forces)
    else:
        atoms = read(args.scf_input_file, format="espresso-in")
        write_vesta(args.output_file, atoms, forces)
