
import argparse
import numpy as np
from ase.io import read

'''
Build a .axsf file with excited-state forces, to be visualized in XCrySDen.

Reads atomic positions and cell vectors from a Quantum ESPRESSO input file
(via ASE) and per-atom excited-state forces from a forces file, then writes
an XCrySDen AXSF file (PRIMVEC / PRIMCOORD blocks with a force vector
appended to each atom line).

Usage:
    python visualize_forces.py scf_input_file forces_file flavor_forces output_file

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a .axsf file with excited-state forces, to be visualized in XCrySDen."
    )
    parser.add_argument("scf_input_file", help="Quantum ESPRESSO input file with atomic positions and cell")
    parser.add_argument("forces_file", help="File with excited-state forces")
    parser.add_argument(
        "flavor_forces",
        type=int,
        choices=[1, 2, 3],
        help="1 = RPA_diag, 2 = RPA_diag_offdiag, 3 = RPA_diag + Kernel",
    )
    parser.add_argument("output_file", help="Path to write the output .axsf file")
    args = parser.parse_args()

    forces = read_excited_forces(args.forces_file, args.flavor_forces)

    ATOMS, CELL_LATT = get_atoms_from_qe_file(args.scf_input_file)
    Nat = len(ATOMS)

    with open(args.output_file, "w") as arq_out:
        arq_out.write("""ANIMSTEPS  1
CRYSTAL
PRIMVEC \n""")

        for ilat in range(3):
            arq_out.write(f"""    {CELL_LATT[ilat]}\n""")

        arq_out.write(f"""PRIMCOORD    1
{Nat}   1\n""")

        for iatom in range(Nat):
            fx, fy, fz = forces[iatom]
            arq_out.write(f"""    {ATOMS[iatom]}  {fx}   {fy}    {fz}\n""")
