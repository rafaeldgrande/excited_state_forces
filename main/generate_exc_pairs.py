
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description=(
        'Generate exciton_pairs.dat for excited_forces.py.\n\n'
        'One-file mode (default): generates pairs (i, j) with i ≤ j from the same '
        'exciton set, including diagonal (i, i). Hermitian symmetry handles (j, i).\n'
        'Two-file mode (--eigenvalues-2-file): generates all (i, j) cross-pairs '
        'with i from file 1 and j from file 2 (finite-q phonon workflow).'
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--Emax', type=float, default=5.0,
                    help='Maximum exciton energy to include (eV, default: 5.0)')
parser.add_argument('--maxEdiff', type=float, default=1.0,
                    help='Maximum |E_i - E_j| for off-diagonal pairs (eV, default: 1.0)')
parser.add_argument('--eigenvalues-1-file', dest='eigenvalues_1_file',
                    type=str, default='eigenvalues.dat',
                    help='Eigenvalues file; column 0 = energy in eV (default: eigenvalues.dat)')
parser.add_argument('--eigenvalues-2-file', dest='eigenvalues_2_file',
                    type=str, default=None,
                    help='Second eigenvalues file for two-set pairs (enables two-file mode)')
parser.add_argument('--output', type=str, default='exciton_pairs.dat',
                    help='Output file (default: exciton_pairs.dat)')
args = parser.parse_args()

Emax    = args.Emax
maxEdiff = args.maxEdiff

print('--- generate_exc_pairs ---')
print(f'  Emax     : {Emax} eV')
print(f'  maxEdiff : {maxEdiff} eV')
print(f'  output   : {args.output}')

# Load file 1
energies_1 = np.loadtxt(args.eigenvalues_1_file)[:, 0]
within_1 = energies_1 < Emax
print(f'\nFile 1 : {args.eigenvalues_1_file}')
print(f'  Total excitons : {len(energies_1)}')
print(f'  E < {Emax} eV   : {within_1.sum()}  '
      f'(E range: {energies_1[within_1].min():.3f} – {energies_1[within_1].max():.3f} eV)')

pairs = []

if args.eigenvalues_2_file is None:
    # ── One-file mode ──────────────────────────────────────────────────────────
    if maxEdiff == 0:
        print('\nMode: single file — diagonal only (maxEdiff = 0)')
        for i in range(len(energies_1)):
            if energies_1[i] >= Emax:
                continue
            pairs.append((i + 1, i + 1))
        print(f'  Diagonal pairs: {len(pairs)}')
    else:
        print('\nMode: single file')
        print('  Generating (i, i) diagonal pairs and (i, j) off-diagonal pairs with i < j')
        n_diag = 0
        n_offdiag = 0
        for i in range(len(energies_1)):
            if energies_1[i] >= Emax:
                continue
            pairs.append((i + 1, i + 1))
            n_diag += 1
            for j in range(i + 1, len(energies_1)):
                if energies_1[j] >= Emax:
                    continue
                if abs(energies_1[i] - energies_1[j]) < maxEdiff:
                    pairs.append((i + 1, j + 1))
                    n_offdiag += 1
        print(f'  Diagonal pairs    : {n_diag}')
        print(f'  Off-diagonal pairs: {n_offdiag}')

else:
    # ── Two-file mode ──────────────────────────────────────────────────────────
    energies_2 = np.loadtxt(args.eigenvalues_2_file)[:, 0]
    within_2 = energies_2 < Emax
    print(f'\nFile 2 : {args.eigenvalues_2_file}')
    print(f'  Total excitons : {len(energies_2)}')
    print(f'  E < {Emax} eV   : {within_2.sum()}  '
          f'(E range: {energies_2[within_2].min():.3f} – {energies_2[within_2].max():.3f} eV)')

    if maxEdiff == 0:
        print('\nMode: two files — diagonal only (maxEdiff = 0)')
        print('  Generating (i, i) pairs with i from both files (same index)')
        for i in range(min(len(energies_1), len(energies_2))):
            if energies_1[i] >= Emax or energies_2[i] >= Emax:
                continue
            pairs.append((i + 1, i + 1))
        print(f'  Diagonal pairs: {len(pairs)}')
    else:
        print('\nMode: two files (finite-q phonon)')
        print('  Generating all (i, j) cross-pairs with i from file 1, j from file 2')
        for i in range(len(energies_1)):
            if energies_1[i] >= Emax:
                continue
            for j in range(len(energies_2)):
                if energies_2[j] >= Emax:
                    continue
                if abs(energies_1[i] - energies_2[j]) < maxEdiff:
                    pairs.append((i + 1, j + 1))

print(f'\nTotal pairs generated: {len(pairs)}')

with open(args.output, 'w') as f:
    for pair in pairs:
        f.write(f'{pair[0]} {pair[1]}\n')

print(f'Written to {args.output}')
