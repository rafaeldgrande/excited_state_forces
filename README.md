# Excited State Forces

## Overview

This code calculates excited state forces and exciton-phonon matrix elements using a many-body Green's function formalism (GW/BSE + DFPT). It combines exciton coefficients from the Bethe-Salpeter Equation (BSE) with electron-phonon coupling from DFPT to compute forces and Raman spectra in excited electronic states.

**For detailed theory and benchmarks, see:** https://arxiv.org/abs/2502.05144

---

## Theory

The excited-state force on phonon mode $\nu$ for exciton $|A\rangle$ is:

$$F_\nu^{\rm RPA} = -\sum_{\mathbf{k}cvc'v'} A^*_{\mathbf{k}cv}\, A_{\mathbf{k}c'v'} \left( g^{\nu}_{\mathbf{k}cc'}\,\delta_{vv'} - g^{\nu}_{\mathbf{k}vv'}\,\delta_{cc'} \right)$$

where $A_{\mathbf{k}cv}$ are BSE exciton coefficients and $g^{\nu}_{\mathbf{k}ij}$ are GW-level electron-phonon matrix elements.

The el-ph renormalization from DFT to QP level uses the approximation:

$$g^{\nu,\rm QP}_{\mathbf{k}ij} = g^{\nu,\rm DFT}_{\mathbf{k}ij} \times \frac{\varepsilon^{\rm QP}_{\mathbf{k}i} - \varepsilon^{\rm QP}_{\mathbf{k}j}}{\varepsilon^{\rm DFT}_{\mathbf{k}i} - \varepsilon^{\rm DFT}_{\mathbf{k}j}}$$

---

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `common/` | Shared constants and utility functions (`constants.py`, `utils.py`) |
| `elph/` | Electron-phonon assembly, interpolation, and 2nd-order coefficients |
| `main/` | Core force calculation script and BSE/QE interface modules |
| `post_processing/` | Cartesian-to-phonon-basis conversion and force visualization |
| `resonant_raman/` | Susceptibility tensors and resonant Raman intensity maps |
| `examples/` | Reference examples for CO (molecule) and LiF (bulk crystal) |

---

## Complete Workflow

Set the repository path once:

```bash
ESF_DIR=/path/to/excited_state_forces
```

### Prerequisite: DFT + DFPT + GW + BSE

Run with Quantum ESPRESSO (`pw.x`, `ph.x`) and BerkeleyGW (`epsilon`, `sigma`, `kernel`, `absorption`). This produces:
- `_ph0/<prefix>.phsave/` — DFPT el-ph XML files
- `scf.in` — ground-state SCF input (used to read cell and k-grid)
- `eqp1.dat` — GW quasiparticle energies
- `eigenvectors.h5` — BSE exciton eigenvectors
- `dtmat` — coarse-to-fine transformation matrices (from `absorption.x`)

See [`examples/README.md`](examples/README.md) for complete input files and SLURM scripts for CO and LiF.

---

### Step 1: Assemble coarse-grid el-ph

```bash
# Run from the DFPT directory (containing scf.in and _ph0/)
python $ESF_DIR/elph/assemble_elph_h5.py
# → elph.h5
```

### Step 2: Interpolate to the fine BSE k-grid

```bash
python $ESF_DIR/elph/interpolate_elph_bgw.py \
    --elph_coarse elph.h5 \
    --dtmat dtmat \
    --Nval <number_of_valence_bands>
# → elph_fine.h5
```

### Step 3: Create `forces.inp`

```
iexc                1
eqp_file            eqp1.dat
exciton_file        eigenvectors.h5
elph_fine_h5_file   elph_fine.h5
```

See [`main/README.md`](main/README.md) for the full parameter reference.

### Step 4: Run the force calculation

```bash
python $ESF_DIR/main/excited_forces.py
# → exc_forces_1_1_ph.dat, exc_forces_1_1_cart.dat
```

---

## Resonant Raman

The `resonant_raman/` module computes 1st- and 2nd-order resonant Raman spectra from the exciton-phonon couplings. See [`resonant_raman/README.md`](resonant_raman/README.md) for the full theory and script documentation.

### 1st Order Workflow

Run from a `1st_der_exc_ph/` directory (after completing Steps 1–4 above):

```
python $ESF_DIR/main/excited_forces.py                                       → exc_forces_*_cart.dat
python $ESF_DIR/post_processing/cart2ph_eigvec.py --read_exciton_pairs_file  → forces in phonon basis
python $ESF_DIR/resonant_raman/assemble_exciton_phonon_coeffs.py             → exciton_phonon_couplings.h5
python $ESF_DIR/resonant_raman/susceptibility_tensors_first_order.py         → susceptibility_tensors_first_order.h5
python $ESF_DIR/resonant_raman/resonant_raman.py --flavor 0                  → Raman maps
```

### 2nd Order Workflow

Run from a `2nd_der_exc_ph/` directory. First compute 2nd-order el-ph coefficients from the same `elph_fine.h5`:

```bash
# Step 1: Compute 2nd-order el-ph coefficients via perturbation theory
python $ESF_DIR/elph/elph_coeffs_second_derivative.py \
    --elph_fine ../1st_der_exc_ph/elph_fine.h5 \
    --eqp eqp1.dat \
    --Nval <number_of_valence_bands> \
    --out 2nd_order_elph_fine.h5
```

In `forces.inp`, point to the 2nd-order file:
```
elph_fine_h5_file              2nd_order_elph_fine.h5
use_second_derivatives_elph_coeffs  True
```

Then continue the same pipeline:
```
python $ESF_DIR/main/excited_forces.py
python $ESF_DIR/post_processing/cart2ph_eigvec.py --read_exciton_pairs_file
python $ESF_DIR/resonant_raman/assemble_exciton_phonon_coeffs.py
python $ESF_DIR/resonant_raman/susceptibility_tensors_second_order.py
python $ESF_DIR/resonant_raman/resonant_raman.py \
    --first-order-file ../1st_der_exc_ph/susceptibility_tensors_first_order.h5 \
    --second-order-file susceptibility_tensors_second_order.h5 \
    --flavor 3
```

---

## Module Reference

### `elph/`

Scripts for electron-phonon matrix elements. See [`elph/README.md`](elph/README.md).

| Script | Description |
|--------|-------------|
| `assemble_elph_h5.py` | Reads QE DFPT XML files, rotates to Cartesian basis → `elph.h5` |
| `interpolate_elph_bgw.py` | Interpolates coarse → fine k-grid via BGW `dtmat` → `elph_fine.h5` |
| `elph_coeffs_second_derivative.py` | 2nd-order el-ph via perturbation theory → `2nd_order_elph_fine.h5` |
| `bgw_binary_io.py` | Low-level reader for BerkeleyGW binary files (`dtmat`, `vmtxel`) |
| `modify_WFN_header.py` | Replaces `/mf_header` in a `WFN.h5` file |

### `main/`

Core force calculation. See [`main/README.md`](main/README.md).

| Script | Description |
|--------|-------------|
| `excited_forces.py` | Main script — reads inputs, orchestrates all steps, writes output |
| `excited_forces_m.py` | Core functions: force calculation, k-point matching, el-ph renormalization |
| `excited_forces_classes.py` | Data structure classes (`Parameters_MF`, `Parameters_BSE`) |
| `excited_forces_config.py` | Configuration parser for `forces.inp` |
| `bgw_interface_m.py` | Reads BerkeleyGW HDF5 files (`eigenvectors.h5`, `hbse.h5`, `eqp1.dat`) |
| `qe_interface_m.py` | Reads Quantum ESPRESSO DFPT output |

### `post_processing/`

| Script | Description |
|--------|-------------|
| `cart2ph_eigvec.py` | Converts Cartesian forces to phonon-mode basis |
| `visualize_forces.py` | Force visualization utilities |
| `first_order_pert_on_eigvals_dip_moments.py` | First-order perturbation theory on exciton eigenvalues and dipole moments |

### `resonant_raman/`

See [`resonant_raman/README.md`](resonant_raman/README.md).

| Script | Description |
|--------|-------------|
| `assemble_exciton_phonon_coeffs.py` | Assembles per-pair exciton-phonon couplings into `exciton_phonon_couplings.h5` |
| `susceptibility_tensors_first_order.py` | 1st-order polarizability derivatives vs. excitation energy |
| `susceptibility_tensors_second_order.py` | 2nd-order susceptibility tensors (triple + double resonance) |
| `resonant_raman.py` | Raman intensity maps; 6 flavors of 1st/2nd order contributions |
| `plot_raman_spectra.py` | Raman spectra at fixed excitation energies |
| `plot_susceptibility_tensors.py` | Raw susceptibility tensor components vs. excitation energy |
| `interactive_vis_resonant_map.py` | Self-contained interactive HTML Raman map viewer |
| `analisys_exc_ph_offdiag_coeffs_vs_energy_diff.py` | Diagnostic: off-diagonal coupling convergence vs. energy cutoff |

### `common/`

| Module | Description |
|--------|-------------|
| `constants.py` | Physical constants (`Ry2eV`, `bohr2A`) and tolerances |
| `utils.py` | Shared utility functions |

---

## External Dependencies

- **Quantum ESPRESSO** (`pw.x`, `ph.x`, `dynmat.x`) — DFT ground state and DFPT
- **BerkeleyGW** (`epsilon`, `sigma`, `kernel`, `absorption`) — GW and BSE

Python packages: `numpy`, `scipy`, `h5py`, `ase`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{delgrande2025,
      title={Revisiting ab-initio excited state forces from many-body Green's function formalism: approximations and benchmark}, 
      author={Rafael R. Del Grande and David A. Strubbe},
      year={2025},
      eprint={2502.05144},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.05144}, 
}
```
