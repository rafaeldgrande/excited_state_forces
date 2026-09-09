import numpy as np
import re
from ase.data import atomic_masses, atomic_numbers

ry2ev = 13.605703976
bohr2ang = 0.529177249

'''
Replays a chronological history of (position, force, energy) configurations
through either BFGS or nonlinear Conjugate Gradient (Polak-Ribiere+),
selected via the 'method' config key ('bfgs' or 'cg'). Both reuse the same
configuration history and reporting/displacement-writing machinery; only
the core update differs:

BFGS -- builds an inverse-Hessian estimate H_n (Nocedal & Wright eq. 6.17,
with eq. 6.20 initial scaling) from the first n configurations. Requires
the curvature condition (s_k . y_k > 0) to update H; violating pairs are
skipped, H kept from the previous step. Predicted energy at the quadratic
minimum reached by a Newton step Delta_x = H_n F_n:
    Delta_E = -1/2 F_n^T H_n F_n

CG -- builds a Polak-Ribiere+ search direction d_n from consecutive forces
only (d_k = F_k + max(0, F_k.(F_k-F_{k-1})/(F_{k-1}.F_{k-1})) d_{k-1}),
which unlike BFGS is always well-defined regardless of the curvature
condition. To still report a predicted energy, a scalar effective
curvature kappa_eff = (s_k.y_k)/(s_k.s_k) is tracked from the most recent
curvature-condition-passing pair (frozen otherwise), giving the standard
1D line-search quadratic minimum along d_n:
    Delta_E = -1/2 (F_n.d_n)^2 / (kappa_eff |d_n|^2)

Both run independently for three force channels (total = DFT+exciton,
DFT-only, exciton-only), and both can write out the actual next
displacement (from the full history) in the same atom_index/dx/dy/dz
format apply_displacements.py expects, with a max_disp cap (uniform
rescale, direction preserved) since neither method's raw step is
guaranteed to be a safe size.

Excited-state forces are read from forces_cart.out (the raw per-atom force
block, not the full excited_forces.out log), which has no embedded exciton
energy -- so excited_forces_files points to a file with one
'forces_cart.out_path  exciton_energy_eV' pair per line instead of a bare
list of paths.
'''

list_position_files = 'list_position_files.dat'
excited_forces_files = 'excited_forces_files.dat'  # lines: "forces_cart.out_path  exciton_energy_eV"
flavor = 2
minimize_just_excited_state = False
exciton_per_unit_cell = 1
method = 'bfgs'
output_file = None                 # default: f'{method}_prediction_vs_step.dat'
output_displacements_file = None   # default: f'displacements_{method}.dat'
do_not_move_CM = True
max_disp = 0.5

curvature_tol = 1e-8


def true_or_false(text, default_value):
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False
    return default_value


def read_input(input_file):
    global list_position_files, excited_forces_files, flavor
    global minimize_just_excited_state, exciton_per_unit_cell, method
    global output_file, output_displacements_file, do_not_move_CM, max_disp

    try:
        arq_in = open(input_file)
    except FileNotFoundError:
        print(f'WARNING! - Input file {input_file} not found! Using default values.')
        return

    print(f'Reading input file {input_file}')
    for line in arq_in:
        linha = line.split()
        if len(linha) >= 2:
            if linha[0] == 'list_position_files':
                list_position_files = linha[1]
            elif linha[0] == 'excited_forces_files':
                excited_forces_files = linha[1]
            elif linha[0] == 'flavor':
                flavor = int(linha[1])
            elif linha[0] == 'minimize_just_excited_state':
                minimize_just_excited_state = true_or_false(linha[1], minimize_just_excited_state)
            elif linha[0] == 'exciton_per_unit_cell':
                exciton_per_unit_cell = float(linha[1])
            elif linha[0] == 'method':
                method = linha[1].lower()
            elif linha[0] == 'output_file':
                output_file = linha[1]
            elif linha[0] == 'output_displacements_file':
                output_displacements_file = linha[1]
            elif linha[0] == 'do_not_move_CM':
                do_not_move_CM = true_or_false(linha[1], do_not_move_CM)
            elif linha[0] == 'max_disp':
                max_disp = float(linha[1])
    arq_in.close()

    if method not in ('bfgs', 'cg'):
        raise ValueError(f"method = '{method}' not recognized -- must be 'bfgs' or 'cg'.")


def read_list_file(path):
    with open(path) as arq:
        return [line.strip() for line in arq if line.strip() and not line.strip().startswith('#')]


def derive_qe_input_path(qe_output_path):
    if qe_output_path.endswith('.out'):
        return qe_output_path[:-len('.out')] + '.in'
    raise ValueError(
        f"Cannot derive the QE input file for '{qe_output_path}': "
        "expected a path ending in '.out' (input assumed to sit next to it, same basename with '.in')."
    )


def parse_positions_from_qe_input(qe_input_path):
    with open(qe_input_path) as arq:
        lines = arq.readlines()

    symbols, positions = [], []
    for i, line in enumerate(lines):
        if line.strip().startswith('ATOMIC_POSITIONS'):
            units = line.strip()
            if 'angstrom' not in units.lower():
                raise ValueError(
                    f"{qe_input_path}: ATOMIC_POSITIONS units are '{units}', expected angstrom."
                )
            for line2 in lines[i + 1:]:
                parts = line2.split()
                if len(parts) < 4:
                    break
                symbols.append(parts[0])
                positions.extend([float(parts[1]), float(parts[2]), float(parts[3])])
            break

    return symbols, np.array(positions)


force_line_re = re.compile(
    r'atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*([\-\d.Ee]+)\s+([\-\d.Ee]+)\s+([\-\d.Ee]+)'
)
total_energy_re = re.compile(r'!\s+total energy\s*=\s*([\-\d.Ee]+)\s*Ry')


def parse_dft_forces_and_energy(qe_output_path, Natoms):
    with open(qe_output_path) as arq:
        text = arq.read()

    matches = force_line_re.findall(text)
    if len(matches) < Natoms:
        print(f'WARNING! {qe_output_path}: found only {len(matches)} force lines, expected {Natoms}. '
              'Setting DFT forces to zero.')
        forces = np.zeros(3 * Natoms)
    else:
        last_block = matches[-Natoms:]
        forces = np.zeros(3 * Natoms)
        for iatom, fx, fy, fz in last_block:
            i = int(iatom) - 1
            forces[3 * i:3 * i + 3] = [float(fx), float(fy), float(fz)]

    energy_matches = total_energy_re.findall(text)
    if not energy_matches:
        print(f'WARNING! {qe_output_path}: total energy line not found. Setting DFT energy to NaN.')
        energy_ry = np.nan
    else:
        energy_ry = float(energy_matches[-1])

    return forces, energy_ry * ry2ev


exc_force_line_re = re.compile(r'^\s*(\d+)\s+([xyz])\s+(.+)$', re.MULTILINE)


def read_excited_forces_list(path):
    '''excited_forces_files.dat now carries the exciton energy alongside each
    forces_cart.out path (that file has no embedded energy line, unlike the
    older excited_forces.out), one 'file energy' pair per line:
        forces_cart_step0.out  4.139289
        forces_cart_step1.out  4.087213
        ...
    '''
    entries = []
    with open(path) as arq:
        for line in arq:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"{path}: expected 'file energy' per line, got: '{line}'"
                )
            entries.append((parts[0], float(parts[1])))
    return entries


def parse_excited_forces_and_energy(excited_forces_path, flavor, Natoms, exciton_energy):
    with open(excited_forces_path) as arq:
        text = arq.read()

    dir_index = {'x': 0, 'y': 1, 'z': 2}
    forces = np.zeros(3 * Natoms)
    n_found = 0
    for iatom_str, direction, rest in exc_force_line_re.findall(text):
        iatom = int(iatom_str)
        if iatom < 1 or iatom > Natoms:
            continue
        values = rest.split()
        if len(values) < flavor:
            continue
        forces[3 * (iatom - 1) + dir_index[direction]] = np.real(complex(values[flavor - 1]))
        n_found += 1

    if n_found < 3 * Natoms:
        print(f'WARNING! {excited_forces_path}: found {n_found} force components, expected {3 * Natoms}.')

    return forces, exciton_energy


def write_displacements(displacements, arq_name, Natoms):
    print(f"Modulus of displacement = {np.linalg.norm(displacements):.6f} angstroms")
    print(f"Writing displacements in {arq_name} file")
    with open(arq_name, 'w') as arq_out:
        for iatom in range(Natoms):
            r = displacements[3 * iatom:3 * (iatom + 1)]
            arq_out.write(f"{iatom + 1}    {r[0]:.8f}     {r[1]:.8f}    {r[2]:.8f} \n")


def load_history(list_position_files, excited_forces_files, flavor,
                  minimize_just_excited_state, exciton_per_unit_cell):

    position_files = read_list_file(list_position_files)
    exc_entries = read_excited_forces_list(excited_forces_files)
    exc_force_files = [entry[0] for entry in exc_entries]
    exc_energies_input = [entry[1] for entry in exc_entries]

    if len(position_files) != len(exc_force_files):
        raise ValueError(f"{list_position_files} and {excited_forces_files} have different lengths.")

    symbols, Natoms = None, None
    positions_all, dft_forces_all, exc_forces_all = [], [], []
    dft_energies, exc_energies = [], []

    for pos_file, exc_file, exc_energy_in in zip(position_files, exc_force_files, exc_energies_input):
        qe_input_path = derive_qe_input_path(pos_file)
        symbols_i, positions_i = parse_positions_from_qe_input(qe_input_path)

        if Natoms is None:
            Natoms = len(symbols_i)
            symbols = symbols_i
        elif len(symbols_i) != Natoms:
            raise ValueError(f"{qe_input_path}: {len(symbols_i)} atoms, expected {Natoms}.")

        dft_forces, dft_energy = parse_dft_forces_and_energy(pos_file, Natoms)
        dft_forces = dft_forces * ry2ev / bohr2ang

        exc_forces, exc_energy = parse_excited_forces_and_energy(exc_file, flavor, Natoms, exc_energy_in)
        exc_forces = exc_forces * exciton_per_unit_cell

        positions_all.append(positions_i)
        dft_forces_all.append(dft_forces)
        exc_forces_all.append(exc_forces)
        dft_energies.append(dft_energy)
        exc_energies.append(exc_energy)

    return (Natoms, symbols, np.array(positions_all), np.array(dft_forces_all), np.array(exc_forces_all),
            np.array(dft_energies), np.array(exc_energies), position_files)


def bfgs_predictions(positions, forces, ref_energies, curvature_tol=1e-8, channel_name=''):
    N, dim = positions.shape
    H = np.eye(dim)
    first_update_done = False
    n_skipped = 0

    # (n_configs_used, actual_energy_last, predicted_energy, delta_E, trace_H, delta_x_predicted)
    results = []

    for k in range(N - 1):
        s = positions[k + 1] - positions[k]
        y = forces[k] - forces[k + 1]
        sy = np.dot(s, y)

        if sy <= curvature_tol:
            print(f'WARNING [{channel_name}]: curvature condition violated between configs '
                  f'{k + 1} and {k + 2} (s.y = {sy:.4e} <= 0). Skipping BFGS update, H unchanged.')
            n_skipped += 1
        else:
            if not first_update_done:
                H = (sy / np.dot(y, y)) * np.eye(dim)
                first_update_done = True
            rho = 1.0 / sy
            I = np.eye(dim)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        n_configs = k + 2
        F_last = forces[k + 1]
        delta_x = H @ F_last
        delta_E = -0.5 * np.dot(F_last, delta_x)
        predicted_energy = ref_energies[k + 1] + delta_E
        results.append((n_configs, ref_energies[k + 1], predicted_energy, delta_E, np.trace(H), delta_x))

    print(f'[{channel_name}] Skipped {n_skipped}/{N - 1} pairs for curvature-condition violations.\n')
    return results, H


def cg_predictions(positions, forces, ref_energies, curvature_tol=1e-8, channel_name=''):
    N, dim = positions.shape
    d = forces[0].copy()
    kappa_eff = None
    n_skipped = 0

    # (n_configs_used, actual_energy_last, predicted_energy, delta_E, kappa_eff, delta_x_predicted)
    results = []

    for k in range(N - 1):
        F_k, F_k1 = forces[k], forces[k + 1]

        s = positions[k + 1] - positions[k]
        y = F_k - F_k1
        sy = np.dot(s, y)

        if sy <= curvature_tol:
            print(f'WARNING [{channel_name}]: curvature condition violated between configs '
                  f'{k + 1} and {k + 2} (s.y = {sy:.4e} <= 0). Keeping previous kappa_eff '
                  '(this does NOT block the CG direction update).')
            n_skipped += 1
        else:
            kappa_eff = sy / np.dot(s, s)

        if k == 0:
            beta = 0.0  # first direction is plain steepest descent, d already = F_1
        else:
            beta = max(0.0, np.dot(F_k1, F_k1 - F_k) / np.dot(F_k, F_k))
        d = F_k1 + beta * d

        n_configs = k + 2
        dd = np.dot(d, d)
        if kappa_eff is None or kappa_eff <= 0 or dd <= 0:
            delta_E = 0.0
            delta_x = None
        else:
            Fd = np.dot(F_k1, d)
            delta_E = -0.5 * Fd ** 2 / (kappa_eff * dd)
            delta_x = (Fd / (kappa_eff * dd)) * d
        predicted_energy = ref_energies[k + 1] + delta_E
        results.append((n_configs, ref_energies[k + 1], predicted_energy, delta_E,
                         kappa_eff if kappa_eff is not None else np.nan, delta_x))

    print(f'[{channel_name}] {n_skipped}/{N - 1} pairs had kappa_eff frozen '
          '(curvature-condition violation); CG direction updates were unaffected.\n')
    return results, d, kappa_eff


def run_predictions(method, positions, forces, ref_energies, curvature_tol, channel_name):
    '''Dispatches to bfgs_predictions or cg_predictions; returns (results, final_state)
    where final_state is whatever compute_final_displacement needs for this method.'''
    if method == 'bfgs':
        results, H_final = bfgs_predictions(positions, forces, ref_energies, curvature_tol, channel_name)
        return results, {'H': H_final}
    else:
        results, d_final, kappa_final = cg_predictions(positions, forces, ref_energies, curvature_tol, channel_name)
        return results, {'d': d_final, 'kappa': kappa_final}


def compute_final_displacement(method, final_state, F_final):
    if method == 'bfgs':
        return final_state['H'] @ F_final
    else:
        d, kappa = final_state['d'], final_state['kappa']
        if kappa is None or kappa <= 0:
            raise ValueError('Never got a valid (curvature-condition-passing) secant pair -- '
                              'cannot form a CG step length. Check the input data.')
        dd = np.dot(d, d)
        alpha_opt = np.dot(F_final, d) / (kappa * dd)
        return alpha_opt * d


if __name__ == '__main__':

    print(100 * '#')
    print('Optimizer replay over configuration history')
    print(100 * '#')

    read_input('optimizer_test.inp')

    if output_file is None:
        output_file = f'{method}_prediction_vs_step.dat'
    if output_displacements_file is None:
        output_displacements_file = f'displacements_{method}.dat'

    print('Parameters:')
    print(f'method: {method}')
    print(f'list_position_files: {list_position_files}')
    print(f'excited_forces_files: {excited_forces_files}')
    print(f'flavor: {flavor}\n')

    Natoms, symbols, positions, dft_forces, exc_forces, dft_e, exc_e, position_files = load_history(
        list_position_files, excited_forces_files, flavor,
        minimize_just_excited_state, exciton_per_unit_cell
    )
    tot_forces = exc_forces if minimize_just_excited_state else dft_forces + exc_forces
    tot_e = dft_e + exc_e
    print(f'Loaded {len(position_files)} configurations, Natoms = {Natoms}, 3N = {3 * Natoms}.\n')

    results_total, state_total = run_predictions(method, positions, tot_forces, tot_e, curvature_tol, f'total ({method})')
    results_dft, _ = run_predictions(method, positions, dft_forces, dft_e, curvature_tol, f'DFT ({method})')
    results_exc, _ = run_predictions(method, positions, exc_forces, exc_e, curvature_tol, f'exciton ({method})')

    hessian_col = 'TraceH' if method == 'bfgs' else 'Kappa'
    with open(output_file, 'w') as arq:
        arq.write('# Nconfigs_used'
                   f'  TotEnergy_actual TotEnergy_{method} DeltaE_total {hessian_col}_total'
                   f'  DFTEnergy_actual DFTEnergy_{method} DeltaE_dft {hessian_col}_dft'
                   f'  ExcEnergy_actual ExcEnergy_{method} DeltaE_exc {hessian_col}_exc\n')
        for row_tot, row_dft, row_exc in zip(results_total, results_dft, results_exc):
            n_configs = row_tot[0]
            arq.write(f'{n_configs}  '
                      f'{row_tot[1]:.8f} {row_tot[2]:.8f} {row_tot[3]:.8f} {row_tot[4]:.6e}  '
                      f'{row_dft[1]:.8f} {row_dft[2]:.8f} {row_dft[3]:.8f} {row_dft[4]:.6e}  '
                      f'{row_exc[1]:.8f} {row_exc[2]:.8f} {row_exc[3]:.8f} {row_exc[4]:.6e}\n')

    print(f'\nWrote {output_file}')

    F_final = tot_forces[-1]
    displacement = compute_final_displacement(method, state_total, F_final)

    if do_not_move_CM:
        Masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
        Tot_mass = np.sum(Masses)
        CM_disp = np.zeros(3)
        for iatom in range(Natoms):
            CM_disp += displacement[3 * iatom:3 * (iatom + 1)] * Masses[iatom] / Tot_mass
        for iatom in range(Natoms):
            displacement[3 * iatom:3 * (iatom + 1)] -= CM_disp

    per_atom_norms = np.linalg.norm(displacement.reshape(Natoms, 3), axis=1)
    max_atom_disp = per_atom_norms.max()
    if max_atom_disp > max_disp:
        scale = max_disp / max_atom_disp
        print(f'\nRaw {method.upper()} step has a max per-atom displacement of {max_atom_disp:.4f} Ang '
              f'(atom {per_atom_norms.argmax() + 1}), above max_disp = {max_disp:.4f} Ang. '
              f'Scaling the whole displacement vector by {scale:.4f} (direction preserved).')
        displacement = displacement * scale

    print(f'\nDisplacement from the {method.upper()} step using the full history (total channel, '
          f'{len(position_files)} configs), from the last configuration ({position_files[-1]}):')
    write_displacements(displacement, output_displacements_file, Natoms)
    print(f'Apply it with apply_displacements.py against {derive_qe_input_path(position_files[-1])} '
          'to get the next structure to evaluate.')

    print('\nFinished!')
