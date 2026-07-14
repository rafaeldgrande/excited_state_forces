import argparse

'''
Apply atomic displacements to a Quantum ESPRESSO input file.

Reads the ATOMIC_POSITIONS block from a QE input file, adds a per-atom
displacement vector (read from a separate displacements file) to the
corresponding atomic coordinates, and writes the result to a new QE
input file. All other lines of the original file (control blocks, cell
parameters, k-points, etc.) are copied through unchanged.

Displacements file format (whitespace-separated, one atom per line):

    <atom_index> <dx> <dy> <dz>

where <atom_index> is the 1-based index of the atom in the
ATOMIC_POSITIONS block (matching its order in the original QE input),
and <dx> <dy> <dz> are the Cartesian displacements in the same units as
the ATOMIC_POSITIONS block (e.g. Angstrom). Atoms not listed in the
displacements file are left unchanged.

Usage:
    python apply_displacements.py original_file displacements_file new_file

Arguments:
    original_file        Path to the original QE input file (e.g. scf.in)
    displacements_file    Path to the file with per-atom displacements
    new_file              Path to write the modified QE input file
'''

def read_qe_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    atomic_positions_start = None
    atomic_positions_end = None
    atomic_positions = []

    for i, line in enumerate(lines):
        if line.strip().startswith('ATOMIC_POSITIONS'):
            atomic_positions_start = i + 1
        elif atomic_positions_start and line.strip() == '':
            atomic_positions_end = i
            break
        elif atomic_positions_start:
            atomic_positions.append(line)

    return lines, atomic_positions, atomic_positions_start, atomic_positions_end

def read_displacements(file_path):
    displacements = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            atom_index = int(parts[0])
            displacement = [float(parts[1]), float(parts[2]), float(parts[3])]
            displacements[atom_index] = displacement
    return displacements

def apply_displacements(atomic_positions, displacements):
    modified_positions = []
    for i, line in enumerate(atomic_positions):
        parts = line.split()
        atom_index = i + 1
        if atom_index in displacements:
            displacement = displacements[atom_index]
            x = float(parts[1]) + displacement[0]
            y = float(parts[2]) + displacement[1]
            z = float(parts[3]) + displacement[2]
            modified_positions.append(f"{parts[0]} {x:.8f} {y:.8f} {z:.8f}\n")
        else:
            modified_positions.append(line)
    return modified_positions

def write_qe_input(file_path, lines, modified_positions, atomic_positions_start, atomic_positions_end):
    with open(file_path, 'w') as file:
        file.writelines(lines[:atomic_positions_start])
        file.writelines(modified_positions)
        file.writelines(lines[atomic_positions_end:])

def modify_qe_input(original_input_file, displacement_file, modified_input_file):
    lines, atomic_positions, start, end = read_qe_input(original_input_file)
    displacements = read_displacements(displacement_file)
    modified_positions = apply_displacements(atomic_positions, displacements)
    write_qe_input(modified_input_file, lines, modified_positions, start, end)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply per-atom displacements to a Quantum ESPRESSO input file."
    )
    parser.add_argument("original_file", help="Path to the original QE input file (e.g. scf.in). Assuming positions in angstroms and cartersian basis.")
    parser.add_argument("displacements_file", help="Path to the file with per-atom displacements")
    parser.add_argument("new_file", help="Path to write the modified QE input file")
    args = parser.parse_args()

    print(f'Applying displacements from file {args.displacements_file} to file {args.original_file} and saving to {args.new_file}')

    modify_qe_input(args.original_file, args.displacements_file, args.new_file)

    print('Finished!')

