# relaxation_scripts

Tools for driving excited-state structural relaxations from GW/BSE + DFPT
forces: computing the next atomic displacement from available force/energy
data, and applying it to a QE input. 

---

## `displacement_from_forces_history.py`

Computes the next displacement (and reports a predicted-energy diagnostic)
from a **history** of previously evaluated configurations, by replaying. 
Runs the chosen method independently on three force channels (total = DFT+exciton,
DFT-only, exciton-only).

BFGS builds an inverse-Hessian estimate `H_n` from the first `n`
configurations. See more in https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm
CG builds a search direction from consecutive forces only. https://en.wikipedia.org/wiki/Conjugate_gradient_method 

Config file: `optimizer_test.inp`

| key | default | meaning |
|---|---|---|
| `method` | `bfgs` | `bfgs` or `cg` |
| `list_position_files` | `list_position_files.dat` | list of QE output files (see below) |
| `excited_forces_files` | `excited_forces_files.dat` | list of excited-state-forces output files (see below) |
| `flavor` | `2` | which excited-state-forces column to use: 1=RPA_diag, 2=RPA_diag_offdiag, 3=RPA_diag_Kernel |
| `minimize_just_excited_state` | `False` | use only exciton forces (ignore DFT forces) for the "total" channel |
| `exciton_per_unit_cell` | `1` | scales the excited-state forces (e.g. for multi-exciton-per-cell setups) |
| `do_not_move_CM` | `True` | project the final displacement so it doesn't shift the center of mass |
| `max_disp` | `0.5` | cap (Angstrom) on the largest single-atom displacement; the whole 3N vector is uniformly rescaled (direction preserved) if exceeded |
| `output_file` | `{method}_prediction_vs_step.dat` | per-step energy/curvature report |
| `output_displacements_file` | `displacements_{method}.dat` | the actual proposed next displacement (see format below) |

Input files it reads:
- **`list_position_files.dat`**: plain text, one QE output file path per
  line.
- **`excited_forces_files.dat`**: plain text, one `path  energy` pair per
  line (whitespace-separated) — same order/length as
  `list_position_files.dat`; line `i` of each pairs up as configuration
  `i`.
- Each **position-list entry** is a QE `pw.x` output file (e.g. `scf.out`)
  from a **single-point `calculation='scf'` run** — DFT forces and the
  total energy are parsed from it (`atom N type M force = ...` lines, last
  block if there's more than one; `!    total energy = ... Ry`, last
  match). Atomic **positions** are read from the sibling QE **input** file
  next to it (same path with `.out` -> `.in`), since a plain `scf` run
  never moves the atoms — this is why only the input, not the output, has
  a reliable `ATOMIC_POSITIONS angstrom` block. Non-angstrom units raise
  an error (not handled).
- Each **excited-forces-list entry's `path`** is the bare numeric
  excited-state-forces table, not a full run log:
  `forces_cart.out` for the pre-2026-07 code, or
  `exc_forces_<iexc>_<jexc>_cart.dat` (e.g. `exc_forces_1_1_cart.dat` for
  the default diagonal single-exciton relaxation setup, `iexc=jexc=1`) for
  the current `excited_forces.py` (`main/`) — both share the same
  `# Atom  dir  <flavor columns...>` header and `atom_index  x|y|z
  <values...>` row format, so no code changes are needed to read either,
  only the path in this list. The **`energy`** on the same line is the
  exciton energy (eV) for that configuration — this file has no energy
  info of its own in either code version, so it must be supplied here,
  read off the corresponding run's stdout log (`Exciton energy (eV):
  <value>` for the current code; `Omega = <value>` for the pre-2026-07
  code — printed only when a single exciton is loaded, i.e. `iexc=jexc`
  and no exciton pairs list).
- **Caveat for the current code**: `report_forces(..., None, suffix='cart', ...)`
  in `main/excited_forces.py` always passes `F_kernel=None` for the
  cart-basis file, so its 3rd column (`RPA_diag_plus_Kernel`) is
  currently always identical to column 1 (`RPA_diag`) regardless of the
  `Calculate_Kernel` setting — `flavor=3` is effectively a no-op in this
  file until that's wired up.

Output `displacements_{method}.dat` format (also used/expected by
`apply_displacements.py`): one line per atom,
`atom_index(1-based)  dx  dy  dz` in Angstrom.

---

## `apply_displacements.py`

Applies a displacement file to a QE input file's `ATOMIC_POSITIONS` block
(assumed Cartesian, Angstrom); everything else in the file (control
blocks, cell parameters, k-points) is copied through unchanged. Not
config-file driven — plain CLI:

```
python apply_displacements.py original_file displacements_file new_file
```

`displacements_file` format: whitespace-separated,
`atom_index(1-based) dx dy dz`, one line per atom (this is exactly what
`displacement_from_forces_history.py` and `displacements_from_forces.py`
write out). Atoms not listed are left unchanged.

---

## `displacements_from_forces.py`

Unlike `displacement_from_forces_history.py`, this does
not use a history of configurations.

Config file: `harmonic_approx.inp`

| key | default | meaning |
|---|---|---|
| `file_out_QE` | `out` | QE output file with DFT forces (optional; if missing, DFT forces are set to 0) |
| `excited_state_forces_file` | `forces_cart.out` | bare numeric excited-state-forces table (not the full log) |
| `flavor` | `2` | 1=RPA_diag, 2=RPA_diag_offdiag, 3=RPA_diag_Kernel |
| `eigvecs_file` | `eigvecs` | DFPT dynamical-matrix eigenvectors/frequencies, from `dynmat.x`'s `fileig` |
| `qe_input` | `qe.in` | QE input file, for atom symbols/masses (needs an `nat = N` line + `ATOMIC_POSITIONS` block) |
| `do_not_move_CM` | `True` | project out center-of-mass motion |
| `avoid_saddle_points` | `True` | for negative-eigenvalue (unstable) modes, flip the displacement to go the other way instead of following the raw Newton step |
| `limit_disp_eigvec_basis` | `0.5` | cap (Angstrom, per eigenmode) on positive-curvature mode displacements |
| `limit_disp_neg_freq` | `0.0` | cap (Angstrom, per eigenmode) on negative-curvature mode displacements (`0.0` = don't move along unstable modes at all) |
| `exciton_per_unit_cell` | `1` | scales the excited-state forces |
| `minimize_just_excited_state` | `False` | use only exciton forces, ignore DFT forces |
| `dont_project_forces_on_eigvecs` | `False` | if `True`, skip the eigenvector/K machinery entirely; displacement is just `A * F_total`, scaled so `max(|x_i|) = max_disp_parallel_to_forces` |
| `max_disp_parallel_to_forces` | `0.1` | used only when `dont_project_forces_on_eigvecs = True` |

Input files: `file_out_QE` (QE output, DFT forces via `grep force`),
`excited_state_forces_file` (bare numeric table, no header/log text
tolerated — plain `np.loadtxt`), `eigvecs_file` (`dynmat.x` output:
`freq` lines + eigenvector blocks), `qe_input` (needs a literal `nat = N`
line, used to bound how many lines of the following `ATOMIC_POSITIONS`
block to read).

Outputs (when `dont_project_forces_on_eigvecs = False`):
`displacements_Newton_method.dat` (the real one to use) plus three
diagnostic variants for comparison — `displacements_parallel_ftot.dat`,
`displacements_parallel_fex.dat`, `displacements_parallel_fdft.dat`
(what the displacement would be if it were purely parallel to the total,
exciton-only, or DFT-only force, respectively). With
`dont_project_forces_on_eigvecs = True`, only
`displacements_parallel_Ftot_limited.dat` is written.

---

## `rand_disp_finite_temp.py`

Generates a single random thermally-displaced structure by sampling each
DFT phonon mode's amplitude from a Boltzmann distribution
(`sigma = sqrt(k_B T / lambda_i)`), using the same ground-state DFPT
eigenvectors/eigenvalues as `displacements_from_forces.py`. Used to build
an ensemble of displaced structures (e.g. for finite-temperature sampling
or as scattered training data for a regression-based force-constant fit).

**Not config-file driven** — edit the module-level variables directly at
the top of the script:

| variable | default | meaning |
|---|---|---|
| `eigvecs_file` | `'eigvecs'` | same `dynmat.x` output format as above |
| `atomic_pos_file` | `'Atoms_info'` | plain text, one atom per line: `Symbol  x  y  z` (Angstrom) |
| `T` | `300` | temperature in K |
| `seed` | `1234` | *(declared but not currently passed to `np.random` — the RNG is not actually seeded, so runs are not reproducible as-is)* |

Output: `atomic_disp_rand_displacements`, `Symbol x y z` per line (the
full displaced structure, not a delta — different convention from the
other scripts' `atom_index dx dy dz` displacement files, so it is **not**
directly usable as input to `apply_displacements.py`).
