# resonant_raman

Codes to calculate resonant Raman spectra based on excited state forces (exciton-phonon coefficients). Supports both 1st and 2nd order resonant Raman.

## Theory

### Notation

| Symbol | Meaning |
|--------|---------|
| $\Omega$ | Laser (excitation) energy |
| $A, B, C$ | Exciton states |
| $\Omega_S$ | Exciton energy |
| $\omega_\nu$ | Phonon frequency of mode $\nu$ |
| $\gamma$ | Lorentzian broadening |
| $\langle 0 \| r_\alpha \| S \rangle$ | Optical transition dipole along direction $\alpha$ |
| $\langle S \| \partial H / \partial Q_\nu \| S' \rangle$ | Exciton-phonon coupling matrix element |
| $\langle S \| \partial^2 H / \partial Q_\nu^2 \| S' \rangle$ | 2nd-order exciton-phonon coupling |

### 1st Order Susceptibility Tensor

The polarizability derivative $\alpha^{\alpha\beta}_\nu(\Omega)$ has two contributions depending on whether the exciton-phonon coupling is diagonal (d2, 2-band) or off-diagonal (d3, 3-band) in exciton space.

**d2 term** (diagonal):

$$\alpha^{\alpha\beta,\text{d2}}_\nu(\Omega) = -\sum_{A} \frac{\langle 0 | r_\alpha | A \rangle \langle A | \partial H / \partial Q_\nu | A \rangle \langle A | r_\beta | 0 \rangle}{(\Omega - \Omega_A + i\gamma)(\Omega - \omega_\nu - \Omega_A + i\gamma)}$$

**d3 term** (off-diagonal, all $A, B$ pairs):

$$\alpha^{\alpha\beta,\text{d3}}_\nu(\Omega) = -\sum_{A, B} \frac{\langle 0 | r_\alpha | A \rangle \langle A | \partial H / \partial Q_\nu | B \rangle \langle B | r_\beta | 0 \rangle}{(\Omega - \Omega_A + i\gamma)(\Omega - \omega_\nu - \Omega_B + i\gamma)}$$

### 2nd Order Susceptibility Tensor

**Triple resonance** (two 1st-order el-ph vertices, modes $\nu$ and $\nu'$):

$$M^{\alpha\beta}_{\nu\nu'}(\Omega) = -\sum_{A,B,C} \frac{\langle 0 | r_\alpha | A \rangle \langle A | \partial H / \partial Q_\nu | B \rangle \langle B | \partial H / \partial Q_{\nu'} | C \rangle \langle C | r_\beta | 0 \rangle}{(\Omega - \Omega_A + i\gamma)(\Omega - \omega_\nu - \Omega_B + i\gamma)(\Omega - \omega_\nu - \omega_{\nu'} - \Omega_C + i\gamma)}$$

**Double resonance** (one 2nd-order el-ph vertex, same mode $\nu$ emitted twice):

$$M^{\alpha\beta,(2)}_{\nu}(\Omega) = -\sum_{A,B} \frac{\langle 0 | r_\alpha | A \rangle \langle A | \partial^2 H / \partial Q_\nu^2 | B \rangle \langle B | r_\beta | 0 \rangle}{(\Omega - \Omega_A + i\gamma)(\Omega - 2\omega_\nu - \Omega_B + i\gamma)}$$

The double-resonance term contributes to the diagonal $\nu = \nu'$ element of $M^{\alpha\beta}_{\nu\nu'}$.

### Raman Intensity

The intensity map as a function of laser energy $\Omega$ and Raman shift $\omega$ is assembled by placing each mode contribution at its phonon frequency with Lorentzian broadening $L(\omega) = \gamma_L^2 / (\omega^2 + \gamma_L^2)$.

**1st order:**

$$I^{(1)}_{\alpha\beta}(\Omega, \omega) = \sum_\nu \left| w_\nu \ \alpha^{\alpha\beta}_\nu(\Omega) \right|^2 L(\omega - \omega_\nu)$$

**2nd order:**

$$I^{(2)}_{\alpha\beta}(\Omega, \omega) = \sum_{\nu,\nu'} \left| w_\nu \ w_{\nu'} \ M^{\alpha\beta}_{\nu\nu'}(\Omega) \right|^2 L(\omega - \omega_\nu - \omega_{\nu'})$$

The phonon weight factor $w_\nu$ includes Bose-Einstein statistics and zero-point motion:

$$w_\nu = \sqrt{\frac{(n_\nu + 1)\hbar}{2\omega_\nu}}, \qquad n_\nu = \frac{1}{e^{\hbar\omega_\nu / k_B T} - 1}$$

The unpolarized Raman invariant used for the powder-averaged intensity is $45\vert\bar{\alpha}\vert^2 + 7\gamma^2 + 5\delta^2$, where $\bar{\alpha}$ is the isotropic part of the tensor.

### Exciton-phonon coupling

The first derivative is given by

$$\langle A | \frac{\partial H^{\rm{BSE}}}{\partial Q_\nu} | B \rangle = \sum_{vc\mathbf{k}} A^{*}_{vc\mathbf{k}} B_{vc\mathbf{k}} (g^{\nu}_{\mathbf{k}cc'} \delta_{vv'} - g^{\nu}_{\mathbf{k}vv'}\delta_{cc'})$$

and the second derivative is given by

$$\langle A | \frac{\partial^2 H^{\rm{BSE}}}{\partial Q_\nu^2} | B \rangle = \sum_{vc\mathbf{k}} A^{*}_{vc\mathbf{k}} B_{vc\mathbf{k}} (g^{(2)\nu}_{\mathbf{k}cc'} \delta_{vv'} - g^{(2)\nu}_{\mathbf{k}vv'}\delta_{cc'})$$

where $g^{\nu}_{\mathbf{k}ij} = \langle \mathbf{k}i \vert \frac{\partial H^{\rm{QP}}}{\partial Q_\nu} \vert \mathbf{k}j \rangle$ is the electron-phonon coefficient at GW level, and $g^{(2)\nu}_{\mathbf{k}ij}$ is the second derivative of the quasiparticle Hamiltonian with respect to phonon mode $\nu$. The second derivative is computed by

$$
g^{(2)\nu}_{\mathbf{k}ij} = 
\sum_{n} -\frac{g^{\nu}_{\mathbf{k}in} g^{\nu}_{\mathbf{k}nj}}{\epsilon_{\mathbf{k}i} - \epsilon_{\mathbf{k}n}} - \frac{g^{\nu}_{\mathbf{k}in} g^{\nu}_{\mathbf{k}nj}}{\epsilon_{\mathbf{k}j} - \epsilon_{\mathbf{k}n}}
$$

---

## Workflow

Set the path to the repository once and reuse it throughout:

```bash
ESF_DIR=/path/to/excited_state_forces
```

### 1st Order Resonant Raman

Run from the `1st_der_exc_ph/` directory:

```bash
# Step 1: Calculate excited state forces (exciton-phonon coupling at 1st order)
python $ESF_DIR/main/excited_forces.py

# Step 2: Convert forces from Cartesian to phonon displacement basis
python $ESF_DIR/post_processing/cart2ph_eigvec.py --read_exciton_pairs_file

# Step 3: Assemble exciton-phonon matrix elements into a single HDF5 file
python $ESF_DIR/resonant_raman/assemble_exciton_phonon_coeffs.py

# Step 4: Calculate susceptibility tensors at 1st order
python $ESF_DIR/resonant_raman/susceptibility_tensors_first_order.py

# Step 5: Calculate 1st order resonant Raman intensities (flavor 0 = d2 only)
python $ESF_DIR/resonant_raman/resonant_raman.py --flavor 0
```

### 2nd Order Resonant Raman

Run from the `2nd_der_exc_ph/` directory. Requires that `elph_coeffs.h5` was saved during the 1st order `excited_forces.py` run (set `save_elph_coeffs True` in `forces.inp`).

```bash
# Step 1: Calculate 2nd-order electron-phonon coefficients via perturbation theory
python $ESF_DIR/resonant_raman/elph_coeffs_second_derivative.py --nval 5

# Step 2: Run excited state forces with 2nd-order el-ph coefficients
# Add these two lines to forces.inp:
#   elph_coeffs_file_to_be_loaded 2nd_derivative_elph_coeffs.h5
#   use_second_derivatives_elph_coeffs True
python $ESF_DIR/main/excited_forces.py

# Step 3: Convert forces from Cartesian to phonon displacement basis
python $ESF_DIR/post_processing/cart2ph_eigvec.py --read_exciton_pairs_file

# Step 4: Assemble exciton-phonon matrix elements
python $ESF_DIR/resonant_raman/assemble_exciton_phonon_coeffs.py

# Step 5: Calculate susceptibility tensors at 2nd order
python $ESF_DIR/resonant_raman/susceptibility_tensors_second_order.py

# Step 6: Calculate 2nd order resonant Raman intensities
# Use --flavor to select the combination of 1st/2nd order contributions
python $ESF_DIR/resonant_raman/resonant_raman.py \
    --first-order-file susceptibility_tensors_first_order.h5 \
    --second-order-file susceptibility_tensors_second_order.h5 \
    --flavor 3
```

---

## Raman Flavor Index

The `--flavor` argument to `resonant_raman.py` selects which contributions to include:

| Flavor | Description |
|--------|-------------|
| 0 | First-order d2 only |
| 1 | First-order d3 only |
| 2 | Second-order triple resonance only |
| 3 | Second-order triple + double resonance |
| 4 | Second-order triple resonance + first-order d3 |
| 5 | Second-order triple + double resonance + first-order d3 |

---

## Scripts

### `assemble_exciton_phonon_coeffs.py`

Reads per-pair exciton-phonon coupling files (in the phonon basis) and assembles them into a single HDF5 file.

**Inputs:**
- `exciton_pairs.dat` — list of exciton index pairs `(i, j)`, one per line
- `forces_ph.out_i_j` — excited state forces in the phonon basis for each pair `(i, j)`, produced by `cart2ph_eigvec.py`

**Output:**
- `exciton_phonon_couplings.h5` — HDF5 file with datasets:
  - `rpa_diag` — diagonal exciton-phonon couplings, shape `(Nmodes, Nexciton, Nexciton)`
  - `rpa_offdiag` — off-diagonal exciton-phonon couplings, shape `(Nmodes, Nexciton, Nexciton)`

```bash
python assemble_exciton_phonon_coeffs.py
```

---

### `susceptibility_tensors_first_order.py`

Calculates the 1st-order susceptibility tensors α(α,β) as a function of excitation energy. Computes both diagonal (2-band, d2) and off-diagonal (3-band, d3) exciton-phonon coupling contributions.

**Inputs:**
- `exciton_phonon_couplings.h5` — produced by `assemble_exciton_phonon_coeffs.py`
- `eigenvalues_b1.dat`, `eigenvalues_b2.dat`, `eigenvalues_b3.dat` — exciton eigenvalues and dipole matrix elements from BerkeleyGW

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--exc_ph_file` | `exciton_phonon_couplings.h5` | Input exciton-phonon file |
| `--dip_mom_file_b1/b2/b3` | `eigenvalues_b1/2/3.dat` | Dipole moment files |
| `--dE` | `0.001` | Excitation energy grid step (eV) |
| `--gamma` | `0.01` | Broadening parameter (eV) |
| `--vectorized_flavor` | `2` | Vectorization level (0=none, 1=exciton, 2=exciton+modes) |
| `--freqs_file` | `freqs.dat` | Phonon frequencies file (cm⁻¹) |

**Output:**
- `susceptibility_tensors_first_order.h5` — shape `(3, 3, Nmodes, Nexcitation)` for d2 and d3 contributions

```bash
python susceptibility_tensors_first_order.py --dE 0.005 --gamma 0.05
```

---

### `susceptibility_tensors_second_order.py`

Calculates the 2nd-order susceptibility tensors, including triple-resonance and double-resonance contributions. Uses multiprocessing for the double-resonance term.

**Inputs:**
- `1st_order_exciton_phonon_couplings.h5` — 1st-order exciton-phonon couplings
- `2nd_order_exciton_phonon_couplings.h5` — 2nd-order exciton-phonon couplings (from `assemble_exciton_phonon_coeffs.py` run in `2nd_der_exc_ph/`)
- `eigenvalues_b1.dat`, `eigenvalues_b2.dat`, `eigenvalues_b3.dat` — dipole matrix elements

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--first_order_exc_ph_file` | `1st_order_exciton_phonon_couplings.h5` | 1st-order couplings |
| `--second_order_exc_ph_file` | `2nd_order_exciton_phonon_couplings.h5` | 2nd-order couplings |
| `--dE` | `0.001` | Excitation energy step (eV) |
| `--gamma` | `0.01` | Broadening (eV) |
| `--nworkers` | system default | Number of parallel workers for double-resonance |

**Output:**
- `susceptibility_tensors_second_order.h5`

```bash
python susceptibility_tensors_second_order.py --nworkers 8
```

---

### `elph_coeffs_second_derivative.py`

Computes 2nd-order electron-phonon coupling coefficients via second-order perturbation theory, starting from the 1st-order coefficients saved in `elph_coeffs.h5`.

**Inputs:**
- `elph_coeffs.h5` — 1st-order el-ph coefficients (requires `save_elph_coeffs True` in `forces.inp`)
- `eqp.dat` — quasiparticle energies

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--elph` | `elph_coeffs.h5` | Input 1st-order el-ph file |
| `--eqp` | `eqp.dat` | Quasiparticle energies file |
| `--nval` | `1` | Number of valence bands to include in the sum |

**Output:**
- `2nd_derivative_elph_coeffs.h5`

```bash
python elph_coeffs_second_derivative.py --nval 5
```

---

### `resonant_raman.py`

Computes resonant Raman intensity maps (Raman shift vs. excitation energy) from susceptibility tensors. Supports different combinations of 1st and 2nd order contributions via `--flavor`.

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--first-order-file` | `susceptibility_tensors_first_order.h5` | 1st-order susceptibility file |
| `--second-order-file` | `susceptibility_tensors_second_order.h5` | 2nd-order susceptibility file |
| `--freqs-file` | `freqs.dat` | Phonon frequencies (cm⁻¹) |
| `--flavor` | `0` | Contribution flavor (see table above) |
| `--temperature` | `300` | Temperature in Kelvin |
| `--nfreq-ph` | `500` | Number of phonon frequency points for spectrum |
| `--output` | `resonant_raman_data.h5` | Output HDF5 file |
| `--plot-map-log-scale` | off | Use log scale for the Raman map |

**Output:**
- `resonant_raman_data_flavor{N}.h5` — Raman map data
- `raman_map_{pol}_flavor_{N}.png` — polarization-resolved Raman maps
- `raman_map_unpolarized_flavor_{N}.png` — unpolarized Raman map

```bash
python resonant_raman.py --flavor 0
python resonant_raman.py --flavor 3 --first-order-file ../1st_der_exc_ph/susceptibility_tensors_first_order.h5
```

---

### `plot_raman_spectra.py`

Plots Raman spectra (Raman shift vs. intensity) at one or more fixed excitation energies. Reads raw susceptibility tensors directly to allow arbitrary phonon broadening, independent of the broadening used in `resonant_raman.py`.

**Key arguments:**
| Argument | Description |
|----------|-------------|
| `--Eexc` | One or more excitation energies (eV) to plot |
| `--first-order-file` | 1st-order susceptibility tensor file |
| `--second-order-file` | 2nd-order susceptibility tensor file |
| `--flavor` | Contribution flavor |

```bash
python plot_raman_spectra.py --Eexc 3.0 3.5 4.0 --flavor 0
```

---

### `plot_susceptibility_tensors.py`

Plots the raw susceptibility tensor components α(α,β) vs. excitation energy for each phonon mode.

- **1st-order**: one figure per phonon mode, 3×3 subplots for each (α,β) pair
- **2nd-order**: one figure per (imode, jmode) pair, with titles showing the sum of phonon frequencies

```bash
python plot_susceptibility_tensors.py --flavor 0
```

---

### `interactive_vis_resonant_map.py`

Generates a self-contained interactive HTML viewer for resonant Raman maps. Reads `resonant_raman_data_flavor{0..5}.h5` and embeds all data into a single HTML file backed by Plotly.js.

- **Left panel**: 2D Raman map — click anywhere to set the excitation energy
- **Right panel**: Raman spectrum at the selected excitation energy
- **Controls**: flavor dropdown, polarization dropdown, excitation energy input, linear/log toggle

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `.` | Directory containing the HDF5 data files |
| `--output` | `resonant_raman_viewer.html` | Output HTML file |
| `--max-eexc-points` | all | Downsample excitation energy axis |
| `--max-ph-points` | all | Downsample phonon frequency axis |

```bash
python interactive_vis_resonant_map.py
python interactive_vis_resonant_map.py --data-dir /path/to/run --output viewer.html
```

---

### `analisys_exc_ph_offdiag_coeffs_vs_energy_diff.py`

Diagnostic script that plots `|<A|dH/dQ|B>| / ΔΩ` vs. exciton energy difference `ΔΩ` for all modes and exciton pairs. Useful for choosing an energy cutoff beyond which off-diagonal coupling terms are negligible.

**Inputs:**
- `exciton_phonon_couplings.h5`
- `eigenvalues_b1.dat`

**Output:**
- `exciton_phonon_offdiag_vs_energy_diff.png`

```bash
python analisys_exc_ph_offdiag_coeffs_vs_energy_diff.py
```


