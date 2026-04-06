# resonant_raman
Codes to calculate resonant raman based on excited state forces (exciton-phonon coefficients)

## Workflow

Step 1: Calculate excited state forces
Step 2: Convert forces from cartesian to phonon basis using the `cart2ph_eigvec.py` code from the `excited_state_forces` repository
Step 3: Assemble all exciton-phonon coefficients into a single `.h5` file using `assemble_exciton_phonon_coeffs.py`
Step 4: Run `susceptibility_tensors.py` to calculate the susceptibility tensors
Step 5: Still working on it

---

## Scripts

### `assemble_exciton_phonon_coeffs.py`

Reads per-pair exciton-phonon coupling files (in the phonon basis) and assembles them into a single HDF5 file `exciton_phonon_couplings.h5`.

**Inputs:**
- `exciton_pairs.dat` — list of exciton index pairs `(i, j)`, one per line, indicating which pairs to load
- `forces_ph.out_i_j` — excited state forces in the phonon basis for each pair `(i, j)`, as produced by `cart2ph_eigvec.py` and `excited_state_forces` code. See more details in [here](https://github.com/rafaeldgrande/excited_state_forces)

**Output:**
- `exciton_phonon_couplings.h5` — HDF5 file with datasets:
  - `rpa_diag` — diagonal exciton-phonon couplings, shape `(Nmodes, Nexciton, Nexciton)`
  - `rpa_offdiag` — off-diagonal exciton-phonon couplings, shape `(Nmodes, Nexciton, Nexciton)`

**Example use:**
```bash
# exciton_pairs.dat contains lines like:
# 0 0
# 0 1
# 1 1
python assemble_exciton_phonon_coeffs.py
# Output: exciton_phonon_couplings.h5
```

---

### `susceptibility_tensors.py`

Calculates the susceptibility tensors as a function of excitation energy. Computes contributions from both diagonal (2-band) and off-diagonal (3-band) exciton-phonon coupling terms. It saves the results in `susceptibility_tensors.h5` for later use in Raman spectrum calculations. Its shape is `(3, 3, Nmodes, Nexcitation)` for both diagonal and off-diagonal contributions.

**Inputs:**
- `exciton_phonon_couplings.h5` — produced by `assemble_exciton_phonon_coeffs.py`
- `eigenvalues_b1.dat, eigenvalues_b2.dat, eigenvalues_b3.dat` — exciton eigenvalues and dipole matrix elements. Files from BerkeleyGW code. 

**Key parameters (set at top of script):**
- `Emax`, `dE` — excitation energy range and step (eV)
- `gamma` — broadening parameter (eV)

**Output:**
- HDF5 file `susceptibility_tensors.h5` containing the calculated susceptibility tensors:
  - `alpha_tensor_d2` — 2-band contributions, shape `(3, 3, Nmodes, Nexcitation)`
  - `alpha_tensor_d3` — 3-band contributions, shape `(3, 3, Nmodes, Nexcitation)`

**Example use:**
```bash
python susceptibility_tensors.py
# Output: susceptibility_tensors.h5
```

---

### `analisys_exc_ph_offdiag_coeffs_vs_energy_diff.py`

Analysis script that plots the magnitude of off-diagonal exciton-phonon couplings normalized by the exciton energy difference, `|<A|dH/dQ|B>| / ΔΩ`, as a function of `ΔΩ`. Useful for determining the energy cutoff beyond which off-diagonal coupling terms become negligible.

**Inputs:**
- `exciton_phonon_couplings.h5` — produced by `assemble_exciton_phonon_coeffs.py`
- `eigenvalues_b1.dat` — exciton eigenvalues; first column is the energy in eV

**Output:**
- `exciton_phonon_offdiag_vs_energy_diff.png` — scatter plot of `|<A|dH/dQ|B>| / ΔΩ` vs. `ΔΩ` for all modes and exciton pairs

**Example use:**
```bash
python analisys_exc_ph_offdiag_coeffs_vs_energy_diff.py
# Output: exciton_phonon_offdiag_vs_energy_diff.png
```


