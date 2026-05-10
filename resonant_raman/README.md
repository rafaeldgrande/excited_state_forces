# resonant_raman

Codes to calculate resonant Raman spectra based on excited state forces (exciton-phonon coefficients). Supports both 1st and 2nd order resonant Raman.

---

## Theory

### Notation

| Symbol | Meaning |
|--------|---------|
| $\Omega$ | Laser (excitation) energy |
| $A, B, C$ | Exciton states |
| $\Omega_S$ | Exciton energy |
| $\omega_\nu$ | Phonon frequency of mode $\nu$ |
| $\gamma$ | Lorentzian broadening |
| $\langle 0 \| r_\alpha \| A \rangle$ | Optical transition dipole along direction $\alpha$ for exciton $A$ |
| $\langle A \| \partial H / \partial Q_\nu \| B \rangle$ | Exciton-phonon coupling matrix element |
| $\langle A \| \partial^2 H / \partial Q_\nu^2 \| B \rangle$ | 2nd-order exciton-phonon coupling |

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

$$\left\langle A \left| \frac{\partial H^{\rm{BSE}}}{\partial Q_\nu} \right| B \right\rangle = \sum_{\mathbf{k}cvc'v'} A^{*}_{\mathbf{k}cv} B_{\mathbf{k}c'v'} (g^{\nu}_{\mathbf{k}cc'} \delta_{vv'} - g^{\nu}_{\mathbf{k}vv'}\delta_{cc'})$$

and the second derivative is given by

$$\left\langle A \left| \frac{\partial^2 H^{\rm{BSE}}}{\partial Q_\nu^2} \right| B \right\rangle = \sum_{\mathbf{k}cvc'v'} A^{*}_{\mathbf{k}cv} B_{\mathbf{k}c'v'} (g^{(2)\nu}_{\mathbf{k}cc'} \delta_{vv'} - g^{(2)\nu}_{\mathbf{k}vv'}\delta_{cc'})$$

where $g^{\nu}_{\mathbf{k}ij}$ is the electron-phonon coefficient at GW level and $g^{(2)\nu}_{\mathbf{k}ij}$ is the second-order el-ph coefficient (see [`elph/README.md`](../elph/README.md)).

---

## Workflow

Set the repository path once and reuse it:

```bash
ESF_DIR=/path/to/excited_state_forces
```

### Prerequisites

Both workflows assume the el-ph preparation steps from [`elph/README.md`](../elph/README.md) have already been completed:

```
elph/assemble_elph_h5.py   → elph.h5
elph/interpolate_elph_bgw.py → elph_fine.h5
```

`forces.inp` must include:
```
elph_fine_h5_file   elph_fine.h5
save_forces_h5      True          # required — writes exc_forces.h5
read_exciton_pairs_file  True     # required — reads exciton_pairs.dat
```

---

### 1st Order Resonant Raman

Create `exciton_pairs.dat` listing all pairs $(i,j)$ to compute, then run from the `1st_der_exc_ph/` directory:

```bash
# Step 1: Compute exciton-phonon matrix elements for all pairs
python $ESF_DIR/main/excited_forces.py
# → exc_forces.h5  (contains forces/ph/RPA and system/phonon_frequencies)

# Step 2 (optional): Merge multiple exc_forces.h5 runs into one file
python $ESF_DIR/main/assemble_exciton_phonon_coeffs.py \
    --input exc_forces_batch1.h5 exc_forces_batch2.h5 \
    --output exciton_phonon_couplings.h5

# Step 3: Calculate susceptibility tensors
# (use exc_forces.h5 directly, or exciton_phonon_couplings.h5 if assembled)
python $ESF_DIR/resonant_raman/susceptibility_tensors_first_order.py \
    --exc_ph_file exc_forces.h5
# → susceptibility_tensors_first_order.h5

# Step 4: Calculate 1st order resonant Raman intensities
python $ESF_DIR/resonant_raman/resonant_raman.py --flavor 0
# → raman_map_*.png
```

---

### 2nd Order Resonant Raman

Run from the `2nd_der_exc_ph/` directory. Requires the 1st-order `elph_fine.h5` from the prior workflow.

```bash
# Step 1: Compute 2nd-order el-ph coefficients via perturbation theory
python $ESF_DIR/elph/elph_coeffs_second_derivative.py \
    --elph_fine ../1st_der_exc_ph/elph_fine.h5 \
    --eqp eqp1.dat \
    --Nval <Nval> \
    --out 2nd_order_elph_fine.h5
# → 2nd_order_elph_fine.h5

# Step 2: Compute 2nd-order exciton-phonon matrix elements
# forces.inp must have:
#   elph_fine_h5_file              2nd_order_elph_fine.h5
#   use_second_derivatives_elph_coeffs  True
#   save_forces_h5  True
#   read_exciton_pairs_file  True
python $ESF_DIR/main/excited_forces.py
# → exc_forces.h5

# Step 3 (optional): Merge multiple runs
python $ESF_DIR/main/assemble_exciton_phonon_coeffs.py \
    --input exc_forces_batch1.h5 exc_forces_batch2.h5 \
    --output 2nd_order_exciton_phonon_couplings.h5

# Step 4: Calculate 2nd-order susceptibility tensors
python $ESF_DIR/resonant_raman/susceptibility_tensors_second_order.py \
    --first_order_exc_ph_file  ../1st_der_exc_ph/exc_forces.h5 \
    --second_order_exc_ph_file exc_forces.h5
# → susceptibility_tensors_second_order.h5

# Step 5: Calculate 2nd order resonant Raman intensities
python $ESF_DIR/resonant_raman/resonant_raman.py \
    --first-order-file  ../1st_der_exc_ph/susceptibility_tensors_first_order.h5 \
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

### `assemble_exciton_phonon_coeffs.py` (in `main/`)

Merges one or more `exc_forces.h5` files produced by `excited_forces.py` (with `save_forces_h5 True`) into a single consolidated file. Useful when exciton pairs were computed in separate batches.

Duplicate pairs are detected and only the first occurrence is kept. The output has the same schema as `exc_forces.h5` and can be passed directly to the susceptibility tensor scripts.

**Input:**
- One or more `exc_forces.h5` files (`--input`)

**Output `exciton_phonon_couplings.h5`** — same schema as `exc_forces.h5`:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `exciton_pairs` | `(Npairs, 2)` | 1-based pair indices $(i, j)$ |
| `forces/ph/RPA_diag` | `(Npairs, Nmodes)` | Forces $F_\nu = -\langle i \| \partial H/\partial Q_\nu \| j \rangle$, RPA_diag |
| `forces/ph/RPA` | `(Npairs, Nmodes)` | Same, full RPA |
| `forces/ph/RPA_diag_plus_Kernel` | `(Npairs, Nmodes)` | Same, with kernel correction |
| `system/phonon_frequencies` | `(Nmodes,)` | Phonon frequencies in cm⁻¹ |

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | — | One or more `exc_forces.h5` files |
| `--output`, `-o` | `exciton_phonon_couplings.h5` | Output file |

```bash
python $ESF_DIR/main/assemble_exciton_phonon_coeffs.py \
    --input batch1/exc_forces.h5 batch2/exc_forces.h5 \
    --output exciton_phonon_couplings.h5
```

---

### `susceptibility_tensors_first_order.py`

Calculates 1st-order susceptibility tensors $\alpha^{\alpha\beta}_\nu(\Omega)$ as a function of excitation energy. Computes both d2 (diagonal) and d3 (off-diagonal) exciton-phonon coupling contributions.

Reads the input h5 file (from `excited_forces.py` or `assemble_exciton_phonon_coeffs.py`) and builds the full $(N_{\rm modes}, N_{\rm exc}, N_{\rm exc})$ exciton-phonon matrix: pairs not present in the file are set to zero, and the Hermitian relation $\langle A|dH|B\rangle = \langle B|dH|A\rangle^*$ is used to fill the transpose. Phonon frequencies are read from the h5 file automatically; `--freqs_file` is a fallback for files that predate this feature.

**Inputs:**
- `exc_forces.h5` or `exciton_phonon_couplings.h5` — exciton-phonon couplings (from `excited_forces.py` or `assemble_exciton_phonon_coeffs.py`)
- `eigenvalues_b1.dat`, `eigenvalues_b2.dat`, `eigenvalues_b3.dat` — exciton eigenvalues and dipole matrix elements from BerkeleyGW

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--exc_ph_file` | `exciton_phonon_couplings.h5` | Input exciton-phonon file |
| `--dip_mom_file_b1/b2/b3` | `eigenvalues_b1/2/3.dat` | Dipole moment files |
| `--dE` | `0.001` | Excitation energy grid step (eV) |
| `--gamma` | `0.01` | Broadening parameter (eV) |
| `--vectorized_flavor` | `2` | Vectorization level (0=none, 1=exciton, 2=exciton+modes) |
| `--freqs_file` | — | Phonon frequencies file in cm⁻¹ (optional; read from h5 if available) |
| `--limit_Nexc` | — | Truncate to this many excitons (for testing) |

**Output:**
- `susceptibility_tensors_first_order.h5` — datasets `alpha_tensor_d2` and `alpha_tensor_d3`, shape `(3, 3, Nmodes, Nfreq)`

```bash
python susceptibility_tensors_first_order.py \
    --exc_ph_file exc_forces.h5 \
    --dE 0.005 --gamma 0.05
```

---

### `susceptibility_tensors_second_order.py`

Calculates 2nd-order susceptibility tensors, including triple-resonance (two 1st-order el-ph vertices) and double-resonance (one 2nd-order el-ph vertex) contributions. Uses multiprocessing for the double-resonance term.

Reads both the 1st-order and 2nd-order exciton-phonon files and builds their full exciton-phonon matrices using the same logic as `susceptibility_tensors_first_order.py`. If the two matrices have different $N_{\rm exc}$, both are truncated to the smaller one. Phonon frequencies are read preferentially from the 1st-order file.

**Inputs:**
- 1st-order `exc_forces.h5` (or assembled) — from the 1st-order `excited_forces.py` run
- 2nd-order `exc_forces.h5` (or assembled) — from the 2nd-order `excited_forces.py` run (with `use_second_derivatives_elph_coeffs True`)
- `eigenvalues_b1.dat`, `eigenvalues_b2.dat`, `eigenvalues_b3.dat` — dipole matrix elements

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--first_order_exc_ph_file` | `1st_order_exciton_phonon_couplings.h5` | 1st-order exciton-phonon file |
| `--second_order_exc_ph_file` | `2nd_order_exciton_phonon_couplings.h5` | 2nd-order exciton-phonon file |
| `--dE` | `0.001` | Excitation energy step (eV) |
| `--gamma` | `0.01` | Broadening (eV) |
| `--vectorized_flavor` | `2` | Vectorization level (0=none, 1=exciton, 2=modes+excitons) |
| `--nworkers` | — | Parallel workers for double-resonance (flavor 1 only; `-1` = all CPUs) |
| `--freqs_file` | — | Phonon frequencies file in cm⁻¹ (optional; read from h5 if available) |

**Output:**
- `susceptibility_tensors_second_order.h5` — datasets `alpha_tensor_triple_resonance` `(3, 3, Nmodes, Nmodes, Nfreq)` and `alpha_tensor_double_resonance` `(3, 3, Nmodes, Nfreq)`

```bash
python susceptibility_tensors_second_order.py \
    --first_order_exc_ph_file  ../1st_der_exc_ph/exc_forces.h5 \
    --second_order_exc_ph_file exc_forces.h5 \
    --nworkers 8
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
python resonant_raman.py --flavor 3 \
    --first-order-file ../1st_der_exc_ph/susceptibility_tensors_first_order.h5
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

Plots the raw susceptibility tensor components $\alpha^{\alpha\beta}$ vs. excitation energy for each phonon mode.

- **1st-order**: one figure per phonon mode, 3×3 subplots for each $(\alpha,\beta)$ pair
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

Diagnostic script that plots $|\langle A|\partial H/\partial Q|B\rangle| / \Delta\Omega$ vs. exciton energy difference $\Delta\Omega$ for all modes and exciton pairs. Useful for choosing an energy cutoff beyond which off-diagonal coupling terms are negligible.

**Inputs:**
- `exciton_phonon_couplings.h5` (or `exc_forces.h5`)
- `eigenvalues_b1.dat`

**Output:**
- `exciton_phonon_offdiag_vs_energy_diff.png`

```bash
python analisys_exc_ph_offdiag_coeffs_vs_energy_diff.py
```

---

## Notes

- **Sign convention**: `excited_forces.py` writes forces $F_\nu = -\langle A|\partial H/\partial Q_\nu|B\rangle$ (with the minus sign). The susceptibility scripts negate internally to recover the exciton-phonon matrix elements. The h5 datasets in both `exc_forces.h5` and assembled files follow the forces convention.
- **Missing pairs**: Exciton-phonon matrix elements for pairs not present in the h5 file are set to zero in the full matrix. Use `exciton_pairs.dat` to control which pairs are computed by `excited_forces.py`.
- **Hermitian symmetry**: If pair $(i,j)$ is computed but not $(j,i)$, the susceptibility scripts fill $\langle j|\partial H|i\rangle = \langle i|\partial H|j\rangle^*$ automatically.
- **2nd-order el-ph**: The `elph_coeffs_second_derivative.py` script that computes $g^{(2)}$ is located in `elph/` — see [`elph/README.md`](../elph/README.md).
