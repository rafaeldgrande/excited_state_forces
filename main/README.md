# main

Core module for computing excited-state forces and exciton-phonon matrix elements from many-body perturbation theory (GW/BSE + DFPT).

## Theory

### Notation

| Symbol | Meaning |
|--------|---------|
| $A, B$ | Exciton states |
| $A_{\mathbf{k}cv}$ | BSE exciton coefficient for k-point $\mathbf{k}$, conduction band $c$, valence band $v$ |
| $g^{\nu}_{\mathbf{k}cc'}$ | Electron-phonon (el-ph) matrix element between conduction bands $c, c'$ at $\mathbf{k}$ for mode $\nu$ |
| $g^{\nu}_{\mathbf{k}vv'}$ | El-ph matrix element between valence bands $v, v'$ at $\mathbf{k}$ for mode $\nu$ |
| $\mathbf{Q}$ | Center-of-mass momentum of an exciton |
| $\mathbf{q}$ | Phonon momentum |
| $\varepsilon^{\rm QP}$, $\varepsilon^{\rm DFT}$ | Quasiparticle and DFT single-particle energies |

### Excited-State Forces ($\mathbf{q}=0$ phonons)

The excited-state force on mode $\nu$ for exciton $|A\rangle$ is given by the Hellman-Feynman theorem applied to the BSE Hamiltonian:

$$F_\nu = -\left\langle A \left| \frac{\partial H^{\rm BSE}}{\partial Q_\nu} \right| A \right\rangle$$

**RPA_diag approximation** (diagonal in band indices; eq. 1 of [arXiv:2502.05144](https://arxiv.org/abs/2502.05144)):

$$F_\nu^{\rm RPA\_diag} = -\sum_{\mathbf{k}cv} |A_{\mathbf{k}cv}|^2 \left( g^{\nu}_{\mathbf{k}cc} - g^{\nu}_{\mathbf{k}vv} \right)$$

**Full RPA approximation** (off-diagonal in band indices; eq. 3 of [arXiv:2502.05144](https://arxiv.org/abs/2502.05144)):

$$F_\nu^{\rm RPA} = -\sum_{\mathbf{k}cvc'v'} A^*_{\mathbf{k}cv}\, A_{\mathbf{k}c'v'} \left( g^{\nu}_{\mathbf{k}cc'}\,\delta_{vv'} - g^{\nu}_{\mathbf{k}vv'}\,\delta_{cc'} \right)$$

Both forms are computed simultaneously. The difference $F^{\rm RPA} - F^{\rm RPA\_diag}$ measures the contribution of off-diagonal el-ph matrix elements (band mixing under the phonon perturbation).

An optional kernel correction $F^{\rm Kernel}_\nu = -\langle A|\partial K^{eh}/\partial Q_\nu|A\rangle$ can be enabled with `Calculate_Kernel True`.

### Exciton-Phonon Matrix Elements (finite-$\mathbf{q}$ phonons)

For a phonon with finite momentum $\mathbf{q} = \mathbf{Q}_B - \mathbf{Q}_A$, the exciton-phonon matrix element between exciton $|A(\mathbf{Q}_A)\rangle$ and $|B(\mathbf{Q}_B)\rangle$ is:

$$\left\langle A(\mathbf{Q}_A) \left| \frac{\partial H^{\rm BSE}}{\partial Q_\nu(\mathbf{q})} \right| B(\mathbf{Q}_B) \right\rangle = \sum_{\mathbf{k}cvc'v'} A^*_{\mathbf{k}cv}(\mathbf{Q}_A)\, B_{\mathbf{k}c'v'}(\mathbf{Q}_B) \left( g^{\nu}_{\mathbf{k}cc'}(\mathbf{q})\,\delta_{vv'} - g^{\nu}_{\mathbf{k}+\mathbf{Q}_A,vv'}(\mathbf{q})\,\delta_{cc'} \right)$$

where the valence-band el-ph matrix element is evaluated at $\mathbf{k}+\mathbf{Q}_A$ to account for the Q-shift of the valence states in the finite-momentum BSE.

### El-Ph Renormalization to QP Level

The DFPT el-ph matrix elements are computed at the DFT level. The code promotes them to the quasiparticle (GW) level using the approximation:

$$g^{\nu,\rm QP}_{\mathbf{k}ij} = g^{\nu,\rm DFT}_{\mathbf{k}ij} \times \frac{\varepsilon^{\rm QP}_{\mathbf{k}i} - \varepsilon^{\rm QP}_{\mathbf{k}j}}{\varepsilon^{\rm DFT}_{\mathbf{k}i} - \varepsilon^{\rm DFT}_{\mathbf{k}j}}$$

This renormalization is applied by default (set `no_renorm_elph True` to skip it).

---

## Input Files

| File | Source | Description |
|------|--------|-------------|
| `forces.inp` | user | Configuration file (see parameters table below) |
| `elph_fine.h5` | `interpolate_elph_bgw.py` | Pre-interpolated el-ph matrix elements on the fine BSE k-grid |
| `eigenvectors.h5` | BerkeleyGW `absorption` | BSE exciton eigenvectors and system parameters |
| `eqp1.dat` (or `eqp.dat`) | BerkeleyGW `sigma` | Quasiparticle energy levels |
| `exciton_pairs.dat` | user (optional) | List of exciton pairs $(i,j)$ to compute; one pair per line |
| `eigenvectors_A.h5` | BerkeleyGW `absorption` | Exciton A eigenvectors (finite-q mode only) |
| `eigenvectors_B.h5` | BerkeleyGW `absorption` | Exciton B eigenvectors (finite-q mode only) |

### `forces.inp` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iexc` | int | 1 | Exciton index $i$ for $\langle i\|dH\|j\rangle$ |
| `jexc` | int | same as `iexc` | Exciton index $j$ |
| `eqp_file` | str | `eqp1.dat` | QP energies file |
| `exciton_file` | str | `eigenvectors.h5` | BSE eigenvectors file |
| `elph_fine_h5_file` | str | `elph_fine.h5` | Pre-interpolated el-ph HDF5 file |
| `ncbnds_sum` | int | all | Number of conduction bands to include in the sum |
| `nvbnds_sum` | int | all | Number of valence bands to include in the sum |
| `acoutic_sum_rule` | bool | True | Enforce acoustic sum rule (zero-sum over atoms) |
| `no_renorm_elph` | bool | False | Skip QP-level el-ph renormalization |
| `trust_kpoints_order` | bool | False | Assume BSE and DFPT k-grids have same ordering |
| `do_vectorized_sums` | bool | True | Use vectorized numpy operations |
| `run_parallel` | bool | False | Enable multiprocessing |
| `num_processes` | int | 1 | Number of parallel processes |
| `read_exciton_pairs_file` | bool | False | Read exciton pairs from `exciton_pairs.dat` |
| `use_second_derivatives_elph_coeffs` | bool | False | Use 2nd-order el-ph coefficients (units: Ry/bohr²) |
| `just_RPA_diag` | bool | False | Skip off-diagonal el-ph; compute only the RPA_diag approximation |
| `use_hermicity_F` | bool | True | Exploit $F_{cvc'v'} = F^*_{c'v'cv}$ to halve the number of computed terms |
| `factor_head` | float | 1.0 | Multiplicative factor applied to the head of BSE matrix elements |
| `dfpt_irreps_list` | list[int] | all | 1-based list of irreducible representations to load from `elph_fine.h5` |
| `log_k_points` | bool | False | Write k-points used in BSE and DFPT calculations to stdout |
| `read_Acvk_pos` | bool | False | Read $A_{cvk}$ from files produced by `summarize_eigenvectors.x` |
| `Acvk_directory` | str | `./` | Directory containing the $A_{cvk}$ files |
| `write_dK_mat` | bool | False | Write $\partial K / \partial Q_\nu$ matrix elements to file |
| `Calculate_Kernel` | bool | False | Compute and add kernel correction $F^{\rm Kernel}$ |
| `hbse_file` | str | `hbse.h5` | BSE Hamiltonian file (required if `Calculate_Kernel True`) |
| `save_forces_h5` | bool | False | Save all forces to `exc_forces.h5` |
| `forces_h5_file` | str | `exc_forces.h5` | Output HDF5 forces file |
| `finite_q_phonon` | bool | False | Compute finite-momentum exciton-phonon matrix elements |
| `eigenvectors_A_file` | str | `eigenvectors_A.h5` | Exciton A file (finite-q mode) |
| `eigenvectors_B_file` | str | `eigenvectors_B.h5` | Exciton B file (finite-q mode) |

---

## Output Files

For each exciton pair $(i, j)$ the code produces two text files and optionally one HDF5 file.

### Per-pair text files

**`exc_forces_{i}_{j}_ph.dat`** — Forces in the phonon-mode basis:

```
# mode   freq(cm-1)   RPA_diag   RPA   RPA_diag_plus_Kernel
  1      123.4        -0.0012    -0.0014   -0.0013
  ...
```

- `RPA_diag`: eq. (1) of arXiv:2502.05144 — only diagonal el-ph ($c=c'$, $v=v'$)
- `RPA`: eq. (3) — full off-diagonal el-ph
- `RPA_diag_plus_Kernel`: `RPA_diag` + kernel correction (only if `Calculate_Kernel True`)
- Units: eV/Å (forces) or eV/Å² (if `use_second_derivatives_elph_coeffs True`)

**`exc_forces_{i}_{j}_cart.dat`** — Forces in the Cartesian atomic basis:

```
# Atom  dir    RPA_diag    RPA_diag_offdiag    RPA_diag_plus_Kernel
  1     x      ...
  1     y      ...
  ...
```

- Shape: `(Nat, 3)`, same three approximation columns
- Units: eV/Å

### Optional HDF5 output (`exc_forces.h5`)

Enabled by `save_forces_h5 True`. Contains all forces for all pairs plus system metadata:

| Dataset/Group | Shape | Units | Description |
|---------------|-------|-------|-------------|
| `forces/ph/RPA_diag` | `(Npairs, Nmodes)` | eV/Å | Phonon-mode forces, RPA_diag |
| `forces/ph/RPA` | `(Npairs, Nmodes)` | eV/Å | Phonon-mode forces, full RPA |
| `forces/ph/RPA_diag_plus_Kernel` | `(Npairs, Nmodes)` | eV/Å | With kernel correction |
| `forces/cart/RPA_diag` | `(Npairs, Nat, 3)` | eV/Å | Cartesian forces, RPA_diag |
| `forces/cart/RPA` | `(Npairs, Nat, 3)` | eV/Å | Cartesian forces, full RPA |
| `exciton_pairs` | `(Npairs, 2)` | — | 1-based pair indices $(i,j)$ |
| `system/kpoints_bse` | `(Nk, 3)` | crystal | BSE k-points |
| `system/phonon_frequencies` | `(Nmodes,)` | cm⁻¹ | Phonon frequencies |
| `system/exciton_energies` | `(Nexc,)` | eV | Exciton eigenvalues |
| `energies/Eqp_cond` | `(Nk, Nc)` | eV | QP conduction energies |
| `energies/Eqp_val` | `(Nk, Nv)` | eV | QP valence energies |
| `config/` | attrs | — | Full `forces.inp` configuration |

---

## Workflow

Set the repository path once and reuse it:

```bash
ESF_DIR=/path/to/excited_state_forces
```

### 1. Workflow for Excited-State Forces ($q=0$)

The prerequisite steps (DFT, DFPT, GW, BSE) produce the raw data files. The steps below handle file assembly and the force calculation itself.

```
DFT + DFPT (Quantum ESPRESSO)
    ↓  ph.x calculation with electron_phonon='simple' and ldisp=.true.
    ↓  produces: scf.in, _ph0/system.phsave/, elph.iq.ipert.xml files

GW + BSE (BerkeleyGW)
    ↓  produces: eqp1.dat, eigenvectors.h5
```

**Step 1: Assemble coarse-grid el-ph into HDF5**

```bash
# Run from the directory containing scf.in and _ph0/
python $ESF_DIR/elph_interpolation/assemble_elph_h5.py scf.in elph.h5
```
Produces `elph.h5` (coarse k-grid el-ph in Cartesian + mode basis).

**Step 2: Interpolate el-ph to the fine BSE k-grid**

```bash
python $ESF_DIR/elph_interpolation/interpolate_elph_bgw.py \
    --elph_coarse elph.h5 \
    --dtmat dtmat \
    --Nval <number_of_valence_bands>
```
Produces `elph_fine.h5` (fine-grid el-ph, ready for `excited_forces.py`).

**Step 3: Create `forces.inp`**

```
iexc                1
eqp_file            eqp1.dat
exciton_file        eigenvectors.h5
elph_fine_h5_file   elph_fine.h5
```

Optionally enable multiprocessing and output options:
```
run_parallel        True
num_processes       8
save_forces_h5      True
```

**Step 4: Run the force calculation**

```bash
python $ESF_DIR/main/excited_forces.py
```

Produces `exc_forces_1_1_ph.dat` and `exc_forces_1_1_cart.dat`.

**Step 5 (optional): Convert to phonon-mode basis**

```bash
python $ESF_DIR/post_processing/cart2ph_eigvec.py
```

---

### 2. Workflow for Finite-Momentum Exciton-Phonon Matrix Elements

This mode computes $\langle A(\mathbf{Q}_A) | \partial H / \partial Q_\nu(\mathbf{q}) | B(\mathbf{Q}_B) \rangle$ where $\mathbf{q} = \mathbf{Q}_B - \mathbf{Q}_A \neq 0$.

**Prerequisites:** Two separate BSE calculations at different center-of-mass momenta $\mathbf{Q}_A$ and $\mathbf{Q}_B$ (set via `exciton_Q_shift` in the BerkeleyGW `kernel.inp` and `absorption.inp`).

```
BSE at Q_A (BerkeleyGW)        BSE at Q_B (BerkeleyGW)
    ↓                               ↓
eigenvectors_A.h5               eigenvectors_B.h5
```

**Step 1: Assemble and interpolate el-ph** (same as $q=0$ workflow, Steps 1–2).

The el-ph file must include the phonon at $\mathbf{q} = \mathbf{Q}_B - \mathbf{Q}_A$. When assembling, make sure the DFPT calculation included this q-point.

**Step 2: Create `forces.inp` with finite-q settings**

```
finite_q_phonon       True
eigenvectors_A_file   eigenvectors_A.h5
eigenvectors_B_file   eigenvectors_B.h5
elph_fine_h5_file     elph_fine.h5
eqp_file              eqp1.dat
read_exciton_pairs_file  True
```

Create `exciton_pairs.dat` listing which pairs $\langle A|dH|B \rangle$ to compute:
```
1 1
1 2
2 1
```

**Step 3: Run**

```bash
python $ESF_DIR/main/excited_forces.py
```

The code automatically:
1. Reads $\mathbf{Q}_A$ from `eigenvectors_A.h5` and $\mathbf{Q}_B$ from `eigenvectors_B.h5`
2. Computes $\mathbf{q} = \mathbf{Q}_B - \mathbf{Q}_A$ and locates it in `elph_fine.h5`
3. Applies the Q-shift to valence-band el-ph matrix elements
4. Computes the exciton-phonon matrix elements

Produces `exc_forces_{i}_{j}_ph.dat` and `exc_forces_{i}_{j}_cart.dat` for each pair.

---

## Module Files

| File | Description |
|------|-------------|
| `excited_forces.py` | Main script — reads inputs, orchestrates all steps, writes output |
| `excited_forces_m.py` | Core functions: force calculation, k-point matching, el-ph renormalization |
| `excited_forces_classes.py` | Data structure classes (`Parameters_MF`, `Parameters_BSE`) |
| `excited_forces_config.py` | Configuration parser for `forces.inp` |
| `bgw_interface_m.py` | Reads BerkeleyGW HDF5 files (`eigenvectors.h5`, `hbse.h5`, `eqp1.dat`) |
| `qe_interface_m.py` | Reads Quantum ESPRESSO DFPT output |

---

## Notes

- **Sign convention**: The code reports forces as $F = -\langle A | \partial H / \partial Q_\nu | A \rangle$. If you need the raw exciton-phonon matrix element $\langle A | \partial H / \partial Q_\nu | B \rangle$, multiply the output by $-1$.
- **Units**: Forces are in eV/Å. The internal DFPT el-ph coefficients are in Ry/bohr and converted at output.
- **Band ordering**: Valence bands follow the BerkeleyGW convention — index 0 is the HOMO, index 1 is HOMO-1, etc.
