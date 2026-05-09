# elph

Scripts for assembling, interpolating, and processing electron-phonon (el-ph) matrix elements from Quantum ESPRESSO DFPT calculations. These scripts prepare the `elph_fine.h5` file consumed by `main/excited_forces.py`.

---

## Workflow Overview

```
QE DFPT output                    BerkeleyGW output
(scf.in, _ph0/, matdyn.modes)     (dtmat, WFN_fi.h5)
        │                                  │
        ▼                                  │
assemble_elph_h5.py                        │
  → elph.h5  (coarse k-grid,              │
     Cartesian + mode basis)              │
        │                                  │
        └──────────────────────────────────┤
                                           ▼
                              interpolate_elph_bgw.py
                                → elph_fine.h5  (fine k-grid)
                                           │
                                           ▼  (optional)
                         elph_coeffs_second_derivative.py
                                → 2nd_order_elph_fine.h5
                                           │
                                           ▼
                                main/excited_forces.py
```

---

## Scripts

### `assemble_elph_h5.py`

Reads QE DFPT el-ph XML files, rotates from the symmetry-adapted pattern basis to the Cartesian atomic-displacement basis, and assembles everything into a single `elph.h5` file.

**Prerequisites:**
- A completed `ph.x` run with `electron_phonon='simple'` and `ldisp=.true.` (or `nosym=.true.`)
- `_ph0/<prefix>.phsave/elph.iq.ipert.xml` for every (q, perturbation) pair
- `_ph0/<prefix>.phsave/patterns.iq.xml` for every q-point
- `_ph0/<prefix>.phsave/control_ph.xml`
- `scf.in` (used to read the cell, k-points, and number of bands via ASE)
- `matdyn.modes` (optional — produced by `matdyn.x`; enables phonon-mode projection)

**Usage:**

```bash
# Run from the DFPT calculation directory
python /path/to/elph/assemble_elph_h5.py

# Disable acoustic sum rule (default: ASR applied)
python /path/to/elph/assemble_elph_h5.py --no-ASR
```

The script uses `os.getcwd()` to locate `scf.in` and `_ph0/`. Run it from the directory containing those files.

**Output `elph.h5` layout:**

| Dataset | Shape | Units | Description |
|---------|-------|-------|-------------|
| `g` | `(Nq, Npert, Nk, Nbnds, Nbnds)` | Ry/bohr | El-ph in Cartesian atomic-displacement basis; `alpha = 3*iatom + {x,y,z}` |
| `g_mode` | `(Nq, Nmodes, Nk, Nbnds, Nbnds)` | Ry/bohr | El-ph projected onto phonon real-space displacement eigenvectors from `matdyn.modes` (present only if `matdyn.modes` found) |
| `kpoints_dft_crystal` | `(Nk, 3)` | fractional | NSCF k-points from `scf.in` |
| `kpoints_dft_cart` | `(Nk, 3)` | 2π/a | Cartesian k-points |
| `qpoints_cart` | `(Nq, 3)` | 2π/a | DFPT q-points from `control_ph.xml` |
| `qpoints_crystal` | `(Nq, 3)` | fractional | DFPT q-points in fractional coords |
| `phonon_modes/qpoints` | `(Nq_md, 3)` | 2π/a | q-points from `matdyn.modes` |
| `phonon_modes/frequencies` | `(Nq_md, Nmodes)` | cm⁻¹ | Phonon frequencies |
| `phonon_modes/eigenvectors` | `(Nq_md, Nmodes, Nat, 3)` | dimensionless | Real-space phonon displacement eigenvectors, unit norm |

The acoustic sum rule (ASR) enforces $\sum_{\rm atoms} g_{iq,\, 3\cdot\text{atom}+d,\, ik,nm} = 0$ for each Cartesian direction $d$. It is applied by default.

---

### `interpolate_elph_bgw.py`

Interpolates the coarse-grid el-ph from `elph.h5` to the fine BSE k-grid using the BerkeleyGW coarse-to-fine transformation matrices stored in `dtmat`.

**Interpolation formula:**

$$\langle n, \mathbf{k}_{\rm fi}+\mathbf{q} \mid \delta V(\mathbf{q}) \mid m, \mathbf{k}_{\rm fi} \rangle = \sum_{ab} \langle n, \mathbf{k}_{\rm fi}+\mathbf{q} \mid a, \mathbf{k}_{\rm co}+\mathbf{q} \rangle\, g_{ab}(\mathbf{k}_{\rm co}, \mathbf{q})\, \langle b, \mathbf{k}_{\rm co} \mid m, \mathbf{k}_{\rm fi} \rangle$$

where $\mathbf{k}_{\rm co}$ is the nearest coarse k-point to $\mathbf{k}_{\rm fi}$ and the overlaps $\langle n, \mathbf{k}_{\rm fi} \mid a, \mathbf{k}_{\rm co} \rangle$ come from `dtmat`.

**Valence-band ordering:** BerkeleyGW convention — index 0 = HOMO, index 1 = HOMO-1, etc.

**Usage:**

```bash
python interpolate_elph_bgw.py \
    --elph_coarse elph.h5 \
    --dtmat dtmat \
    --Nval <number_of_valence_bands_in_DFPT> \
    --out elph_fine.h5

# For finite-q: provide fine k-points (reads from WFN_fi.h5 automatically if present)
python interpolate_elph_bgw.py \
    --elph_coarse elph.h5 \
    --dtmat dtmat \
    --Nval 13 \
    --wfn-fi WFN_fi.h5

# Real-flavor dtmat (default is complex)
python interpolate_elph_bgw.py --elph_coarse elph.h5 --dtmat dtmat --Nval 13 --real
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--elph_coarse` | — | Path to `elph.h5` from `assemble_elph_h5.py` |
| `--dtmat` | — | Path to BerkeleyGW `dtmat` binary |
| `--Nval` | — | Number of occupied bands in DFPT (QE `nbnd` convention) |
| `--out` | `elph_fine.h5` | Output filename |
| `--wfn-fi` | auto-discovered | Path to `WFN_fi.h5` (required for finite-q) |
| `--real` | — | Use real-flavor dtmat |

**Output `elph_fine.h5` layout:**

| Dataset | Shape | Units | Description |
|---------|-------|-------|-------------|
| `elph_fine_cond_mode` | `(Nq, Nmodes, Nk_fi, Nc, Nc)` | Ry/bohr | Conduction el-ph, phonon-mode basis; `ic=0` → LUMO |
| `elph_fine_val_mode` | `(Nq, Nmodes, Nk_fi, Nv, Nv)` | Ry/bohr | Valence el-ph, phonon-mode basis; `iv=0` → HOMO |
| `elph_fine_cond_cart` | `(Nq, Npert, Nk_fi, Nc, Nc)` | Ry/bohr | Conduction el-ph, Cartesian basis |
| `elph_fine_val_cart` | `(Nq, Npert, Nk_fi, Nv, Nv)` | Ry/bohr | Valence el-ph, Cartesian basis |
| `Kpoints_in_elph_file` | `(Nk_fi, 3)` | fractional | Fine-grid k-points |
| `qpoints_crystal` | `(Nq, 3)` | fractional | q-points (copied from `elph.h5`) |
| `qpoints_cart` | `(Nq, 3)` | 2π/a | q-points in Cartesian (copied from `elph.h5`) |
| `phonon_modes/` | group | — | Phonon frequencies and eigenvectors (copied from `elph.h5`) |

---

### `elph_coeffs_second_derivative.py`

Computes second-order electron-phonon coupling coefficients via second-order perturbation theory. The output file has the same format as `elph_fine.h5` and can be used directly as `elph_fine_h5_file` in `forces.inp` with `use_second_derivatives_elph_coeffs True`.

**Theory:**

$$g^{(2)}_{\alpha, \mathbf{k}, nm} = -\sum_l g_{\alpha \mathbf{k} nl}\, g_{\alpha \mathbf{k} lm} \left( \frac{1}{\varepsilon_{n\mathbf{k}} - \varepsilon_{l\mathbf{k}}} + \frac{1}{\varepsilon_{m\mathbf{k}} - \varepsilon_{l\mathbf{k}}} \right)$$

where $\alpha$ is a Cartesian atomic-displacement index, and $\varepsilon$ are quasiparticle energies (from `eqp1.dat`, converted to Ry internally). The input el-ph is taken directly from `elph_fine_cond_cart` / `elph_fine_val_cart` — no displacement-pattern rotation needed since those datasets are already in the Cartesian basis.

Units of the result: Ry/bohr².

The computation is vectorized: for each q-point and k-point,

$$g^{(2)} = \underbrace{-(g \odot \Lambda) \cdot g}_{\text{term 1}} + \underbrace{g \cdot (g \odot \Lambda)}_{\text{term 2}}, \quad \Lambda_{nm} = \frac{1}{\varepsilon_n - \varepsilon_m}$$

where $\odot$ is element-wise multiplication and $\cdot$ is matrix multiplication over band indices.

After computing $g^{(2)}$ in the Cartesian basis, it is projected to the phonon-mode basis using the phonon eigenvectors from `phonon_modes/eigenvectors`.

**Usage:**

```bash
python elph_coeffs_second_derivative.py \
    --elph_fine elph_fine.h5 \
    --eqp eqp1.dat \
    --Nval <number_of_valence_bands_in_DFPT> \
    --out 2nd_order_elph_fine.h5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--elph_fine` | `elph_fine.h5` | Input from `interpolate_elph_bgw.py` |
| `--eqp` | `eqp1.dat` | Fine-grid QP energy file (from BerkeleyGW `absorption`) |
| `--Nval` | — | Number of valence bands in DFPT |
| `--out` | `2nd_order_elph_fine.h5` | Output filename |

Then in `forces.inp`:
```
elph_fine_h5_file              2nd_order_elph_fine.h5
use_second_derivatives_elph_coeffs  True
```

---

### `bgw_binary_io.py`

Low-level reader for BerkeleyGW unformatted Fortran binary files. Provides:

- `read_dtmat(filename, complex_flavor=True)` — reads the `dtmat` file produced by `absorption.x`, returning coarse-to-fine transformation matrices (`dcn`, `dvn`), k-point arrays, and interpolation coefficients.
- `read_vmtxel(filename, complex_flavor=True)` — reads optical matrix elements from `vmtxel`.
- `dtmat_to_hdf5(in_path, out_path)` / `vmtxel_to_hdf5(in_path, out_path)` — dump to self-describing HDF5.

Used internally by `interpolate_elph_bgw.py`.

---

### `modify_WFN_header.py`

Utility that replaces the `/mf_header` group in a `WFN.h5` file with the header from another file. Useful when two WFN files are slightly incompatible (e.g., different k-grids with matching geometry) and BerkeleyGW refuses to read them together.

```bash
python modify_WFN_header.py source_header.h5 base_file.h5 --output WFN_mod.h5
```

---

## Step-by-Step Usage

### Standard workflow (q=0 forces)

```bash
ESF=/path/to/excited_state_forces

# 1. Assemble coarse el-ph (run from DFPT directory)
python $ESF/elph/assemble_elph_h5.py

# 2. Interpolate to fine grid (run from BSE/forces directory)
python $ESF/elph/interpolate_elph_bgw.py \
    --elph_coarse /path/to/dfpt/elph.h5 \
    --dtmat dtmat \
    --Nval 13

# 3. Compute forces
python $ESF/main/excited_forces.py
```

### Second-order el-ph (optional)

```bash
python $ESF/elph/elph_coeffs_second_derivative.py \
    --elph_fine elph_fine.h5 \
    --eqp eqp1.dat \
    --Nval 13 \
    --out 2nd_order_elph_fine.h5
```

Then set in `forces.inp`:
```
elph_fine_h5_file              2nd_order_elph_fine.h5
use_second_derivatives_elph_coeffs  True
```

---

## Notes

- **Nval**: this is the total number of valence bands included in the DFPT calculation (`nbnd` in `scf.in` up to and including the HOMO). It determines which rows/columns of `g` belong to the conduction vs. valence sector.
- **Band ordering** in all output files follows the BerkeleyGW convention: valence index 0 = HOMO, 1 = HOMO-1, …; conduction index 0 = LUMO, 1 = LUMO+1, …
- **Cartesian vs. mode basis**: `elph_fine.h5` contains both. The Cartesian basis (`_cart` datasets) is needed for forces in the atomic basis; the mode basis (`_mode` datasets) is needed for forces resolved by phonon mode and frequency. Both are used by `excited_forces.py`.
- **Acoustic sum rule**: applied by default in `assemble_elph_h5.py`. Disable with `--no-ASR` if you want the raw uncorrected couplings.
- **Units**: el-ph matrix elements throughout are in Ry/bohr (first order) or Ry/bohr² (second order). Energies in `eqp1.dat` are in eV and are converted to Ry internally where needed.
