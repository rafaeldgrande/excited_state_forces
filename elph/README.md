# elph

Scripts for assembling, interpolating, and processing electron-phonon (el-ph) matrix elements from Quantum ESPRESSO DFPT calculations. These scripts prepare the `elph_interpolated_kgrid.h5` file consumed by `main/excited_forces.py`.

---

## Workflow Overview

```
QE DFPT output              BerkeleyGW output           BerkeleyGW output
(_ph0/, matdyn.modes,      (WFN_co.h5, or scf.in)      (dtmat, WFN_fi.h5,
 scf.in fallback)                                        eqp.dat)
        │                            │                            │
        ▼                            ▼                            │
                     elph_xml_to_h5.py                             │
        assembly stage → elph_orig_kgrid.h5 (coarse k-grid)            │
                    │                                              │
                    └──────── interpolation stage ◄─────────────────┘
                                → elph_interpolated_kgrid.h5  (fine k-grid)
                                           │
                                           ▼  (optional)
                         elph_coeffs_second_derivative.py
                                → 2nd_order_elph_interpolated_kgrid.h5
                                           │
                                           ▼
                                main/excited_forces.py
```

`elph_xml_to_h5.py` runs both stages in one process by default (no disk
round-trip for the coarse data in between), but either stage can also be run
independently: `--skip-interpolation` stops after writing `elph_orig_kgrid.h5`;
omitting `--elph_dir` resumes interpolation-only from a previously written
`elph_orig_kgrid.h5`.

---

## Scripts

### `elph_xml_to_h5.py`

**Assembly stage** — reads QE DFPT el-ph XML files, rotates from the
symmetry-adapted pattern basis to the Cartesian atomic-displacement basis
(and, if `matdyn.modes` is available, to the phonon-mode basis), and writes
the coarse-grid `elph_orig_kgrid.h5`.

**Interpolation stage** — interpolates the coarse-grid el-ph to the fine BSE
k-grid using the BerkeleyGW coarse-to-fine transformation matrices stored in
`dtmat`, and writes `elph_interpolated_kgrid.h5`.

$$\langle n, \mathbf{k}_{\rm fi}+\mathbf{q} \mid \delta V(\mathbf{q}) \mid m, \mathbf{k}_{\rm fi} \rangle = \sum_{ab} \langle n, \mathbf{k}_{\rm fi}+\mathbf{q} \mid a, \mathbf{k}_{\rm co}+\mathbf{q} \rangle\, g_{ab}(\mathbf{k}_{\rm co}, \mathbf{q})\, \langle b, \mathbf{k}_{\rm co} \mid m, \mathbf{k}_{\rm fi} \rangle$$

where $\mathbf{k}_{\rm co}$ is the nearest coarse k-point to $\mathbf{k}_{\rm fi}$ and the overlaps $\langle n, \mathbf{k}_{\rm fi} \mid a, \mathbf{k}_{\rm co} \rangle$ come from `dtmat`.

**Valence-band ordering:** BerkeleyGW convention throughout — index 0 = HOMO, index 1 = HOMO-1, etc. (conduction: index 0 = LUMO, 1 = LUMO+1, ...).

**Prerequisites (assembly, i.e. when `--elph_dir` is given):**
- A completed `ph.x` run with `electron_phonon='simple'` and `ldisp=.true.` (or `nosym=.true.`)
- `_ph0/<prefix>.phsave/elph.iq.ipert.xml` for every (q, perturbation) pair
- `_ph0/<prefix>.phsave/patterns.iq.xml` for every q-point
- `_ph0/<prefix>.phsave/control_ph.xml`
- Cell, k-points, `nbnd`, and Nval (highest occupied band index): from
  `WFN_co.h5` (`--wfn_origin`, preferred — authoritative, no guessing) or from
  `scf.in`/`bands.in` (`--qe_input`, fallback — Nval is auto-derived from
  `scf.out`'s "number of electrons" line, or from pseudopotential `Z_valence`
  if no `scf.out` is found). `--Nval` always overrides either source manually.
- `matdyn.modes` (optional — produced by `matdyn.x`; enables phonon-mode projection)

**Prerequisites (interpolation, i.e. unless `--skip-interpolation`):**
- `dtmat`, produced by BerkeleyGW's `absorption.<flavour>.x`. Not needed at all
  if el-ph was computed directly on the fine grid (DFPT run on the fine k/q-grid
  itself) — use `--skip-interpolation` instead, see below.
- `WFN_fi.h5` (`--wfn_to_interpolate`) for finite-q interpolation (auto-discovered next to `--dtmat` if not given)
- `eqp.dat` (`--eqp`, optional) — fine-grid QP energies from `inteqp.x`, for QP rescaling

**`--skip-interpolation`** (el-ph computed directly on the fine grid, e.g. DFPT
run on the same k/q-grid the BSE calculation uses — no coarse-to-fine
transformation needed, so `--dtmat`/`--wfn_to_interpolate` are irrelevant): if
`--eqp` points to a valid fine-grid `eqp.dat`, the assembled el-ph is still
band-matched to its Nc/Nv window and QP-rescaled — exactly like the
interpolation stage would do — and written to `--elph_fine`, ready to use
directly as `elph_fine_h5_file` in `forces.inp`. Without `--eqp`, only the raw,
unmatched `--elph_coarse` is written.

**Usage:**

```bash
# Full pipeline: assemble XML -> elph_orig_kgrid.h5, interpolate -> elph_interpolated_kgrid.h5
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 \
    --dtmat dtmat --wfn_to_interpolate WFN_fi.h5 --eqp eqp.dat

# El-ph computed directly on the fine grid (DFPT run on the fine k/q-grid
# itself): no dtmat needed. --eqp band-matches + QP-rescales the result and
# writes elph_interpolated_kgrid.h5, ready to use as elph_fine_h5_file.
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 \
    --skip-interpolation --eqp eqp.dat

# Same, but without --eqp: only the raw, unmatched elph_orig_kgrid.h5 is written
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 \
    --skip-interpolation

# Resume: interpolate from a previously-written elph_orig_kgrid.h5
python elph_xml_to_h5.py --elph_coarse elph_orig_kgrid.h5 --wfn_origin WFN_co.h5 \
    --dtmat dtmat --wfn_to_interpolate WFN_fi.h5

# No WFN_co.h5 available: fall back to scf.in / scf.out / pseudopotentials
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --qe_input scf.in \
    --dtmat dtmat --wfn_to_interpolate WFN_fi.h5

# Manual Nval override (takes precedence over WFN_co.h5 / scf.in either way)
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 --Nval 13

# Disable acoustic sum rule (default: ASR applied, assembly stage only)
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_origin WFN_co.h5 --no-ASR
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--elph_dir` | `None` | phsave dir; if given, the assembly stage runs. If omitted, `--elph_coarse` must point to an existing coarse file (resume: interpolation-only) |
| `--modes_file` | `matdyn.modes` | phonon eigenvectors/frequencies from `matdyn.x` (optional) |
| `--no-ASR` | ASR on | Disable the acoustic sum rule (assembly stage only) |
| `--wfn_origin` | `None` | Path to `WFN_co.h5`. If given and found, cell, k-points, `nbnd`, and Nval are read directly from its `mf_header` |
| `--qe_input` | `scf.in` | QE pw.x input file used for cell/k-points/`nbnd`/Nval when `--wfn_origin` is not given or not found |
| `--Nval` | `None` | Manual override for Nval (highest occupied band index, QE `nbnd` convention). Takes precedence over `--wfn_origin`/`--qe_input` either way |
| `--elph_coarse` | `elph_orig_kgrid.h5` | Output path when `--elph_dir` is given; required input path (must exist) when `--elph_dir` is omitted |
| `--elph_fine` | `elph_interpolated_kgrid.h5` | Output HDF5 filename for the fine-grid el-ph file (interpolated, or band-matched/QP-rescaled directly, see `--skip-interpolation`) |
| `--skip-interpolation` | off | El-ph computed directly on the fine grid; `--dtmat`/`--wfn_to_interpolate` ignored. If `--eqp` is found, band-matches + QP-rescales and writes `--elph_fine`; otherwise only the raw `--elph_coarse` is written |
| `--dtmat` | `dtmat` | Path to BerkeleyGW `dtmat` binary |
| `--wfn_to_interpolate` | `WFN_fi.h5` | Path to `WFN_fi.h5` (required for finite-q interpolation; auto-discovered next to `--dtmat` if not found) |
| `--real` | off | Use real-flavor `dtmat` (default: complex) |
| `--eqp` | `eqp.dat` | Path to fine-grid `eqp.dat` (output of `inteqp.x`). When found, saves QP rescaling matrices and Eqp/Edft energies into `--elph_fine` (interpolation stage, or `--skip-interpolation` band-matching) |

**Output HDF5 layout** (`elph_orig_kgrid.h5` and `elph_interpolated_kgrid.h5` share the exact
same dataset schema — only the numeric coefficients and k-points differ,
coarse vs. fine grid):

| Dataset | Shape | Units | Description |
|---------|-------|-------|-------------|
| `elph_cond_mode` | `(Nq, Nmodes, Nk, Nc, Nc)` | Ry/bohr | Conduction el-ph, phonon-mode basis; `ic=0` → LUMO (present only if `matdyn.modes` found) |
| `elph_val_mode` | `(Nq, Nmodes, Nk, Nv, Nv)` | Ry/bohr | Valence el-ph, phonon-mode basis; `iv=0` → HOMO |
| `elph_cond_cart` | `(Nq, Npert, Nk, Nc, Nc)` | Ry/bohr | Conduction el-ph, Cartesian basis; `alpha = 3*iatom + {x,y,z}` |
| `elph_val_cart` | `(Nq, Npert, Nk, Nv, Nv)` | Ry/bohr | Valence el-ph, Cartesian basis |
| `Kpoints_in_elph_file` | `(Nk, 3)` | fractional | k-points (coarse or fine grid, depending on the file) |
| `qpoints_crystal` | `(Nq, 3)` | fractional | DFPT q-points |
| `qpoints_cart` | `(Nq, 3)` | 2π/a | DFPT q-points in Cartesian |
| `phonon_modes/qpoints` | `(Nq_md, 3)` | 2π/a | q-points from `matdyn.modes` |
| `phonon_modes/frequencies` | `(Nq_md, Nmodes)` | cm⁻¹ | Phonon frequencies |
| `phonon_modes/eigenvectors` | `(Nq_md, Nmodes, Nat, 3)` | dimensionless | Real-space phonon displacement eigenvectors, unit norm |
| `crystal/atomic_numbers` | `(Nat,)` | — | Atomic numbers, from `WFN_co.h5` or `scf.in` |
| `crystal/atomic_positions` | `(Nat, 3)` | Å | Cartesian atomic positions |
| `crystal/lattice_vectors` | `(3, 3)` | Å | Real-space lattice vectors, rows = a1,a2,a3 |
| `QP_rescaling_matrix_cond` | `(Nk, Nc, Nc)` | dimensionless | QP renorm. ratio $(E^{\rm QP}_n - E^{\rm QP}_m)/(E^{\rm DFT}_n - E^{\rm DFT}_m)$ for conduction bands; fine grid only, present when `--eqp` found |
| `QP_rescaling_matrix_val` | `(Nk, Nv, Nv)` | dimensionless | QP renorm. ratio for valence bands; fine grid only |
| `Eqp_cond` / `Eqp_val` | `(Nk, Nc)` / `(Nk, Nv)` | eV | QP energies from `eqp.dat`; fine grid only |
| `Edft_cond` / `Edft_val` | `(Nk, Nc)` / `(Nk, Nv)` | eV | DFT energies from `eqp.dat`; fine grid only |

Root attrs include `grid` (`'coarse'`/`'fine'`), `Nval`, and provenance
(source file paths). The acoustic sum rule (ASR) enforces
$\sum_{\rm atoms} g_{iq,\, 3\cdot\text{atom}+d,\, ik,nm} = 0$ for each
Cartesian direction $d$; it is applied by default during assembly.

In addition to the HDF5 files, plain-text q-point report files are written
to the working directory: `qpoints_cart_co.dat`/`qpoints_crystal_co.dat`
(written once `--elph_coarse` is available, whether freshly assembled or
resumed) and `qpoints_cart_fi.dat`/`qpoints_crystal_fi.dat` (written after
the interpolation stage completes). q-points don't change between the coarse
and fine grids — only k-points do — so the `_co`/`_fi` pairs carry the same
values, one written per stage.

---

### `elph_coeffs_second_derivative.py`

Computes second-order electron-phonon coupling coefficients via second-order perturbation theory. The output file has the same format as `elph_interpolated_kgrid.h5` and can be used directly as `elph_fine_h5_file` in `forces.inp` with `use_second_derivatives_elph_coeffs True`.

**Theory:**

$$g^{(2)}_{\alpha, \mathbf{k}, nm} = -\sum_l g_{\alpha \mathbf{k} nl}\, g_{\alpha \mathbf{k} lm} \left( \frac{1}{\varepsilon_{n\mathbf{k}} - \varepsilon_{l\mathbf{k}}} + \frac{1}{\varepsilon_{m\mathbf{k}} - \varepsilon_{l\mathbf{k}}} \right)$$

where $\alpha$ is a Cartesian atomic-displacement index, and $\varepsilon$ are quasiparticle energies (from `eqp1.dat`, converted to Ry internally). The input el-ph is taken directly from `elph_cond_cart` / `elph_val_cart` — no displacement-pattern rotation needed since those datasets are already in the Cartesian basis.

Units of the result: Ry/bohr².

The computation is vectorized: for each q-point and k-point,

$$g^{(2)} = \underbrace{-(g \odot \Lambda) \cdot g}_{\text{term 1}} + \underbrace{g \cdot (g \odot \Lambda)}_{\text{term 2}}, \quad \Lambda_{nm} = \frac{1}{\varepsilon_n - \varepsilon_m}$$

where $\odot$ is element-wise multiplication and $\cdot$ is matrix multiplication over band indices.

After computing $g^{(2)}$ in the Cartesian basis, it is projected to the phonon-mode basis using the phonon eigenvectors from `phonon_modes/eigenvectors`.

**Usage:**

```bash
python elph_coeffs_second_derivative.py \
    --elph_fine elph_interpolated_kgrid.h5 \
    --eqp eqp1.dat \
    --Nval <number_of_valence_bands_in_DFPT> \
    --out 2nd_order_elph_interpolated_kgrid.h5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--elph_fine` | `elph_interpolated_kgrid.h5` | Input from `elph_xml_to_h5.py` |
| `--eqp` | `eqp1.dat` | Fine-grid QP energy file (from BerkeleyGW `absorption`) |
| `--Nval` | — | Number of valence bands in DFPT |
| `--out` | `2nd_order_elph_interpolated_kgrid.h5` | Output filename |

Then in `forces.inp`:
```
elph_fine_h5_file              2nd_order_elph_interpolated_kgrid.h5
use_second_derivatives_elph_coeffs  True
```

---

### `bgw_binary_io.py`

Low-level reader for BerkeleyGW unformatted Fortran binary files. Provides:

- `read_dtmat(filename, complex_flavor=True)` — reads the `dtmat` file produced by `absorption.x`, returning coarse-to-fine transformation matrices (`dcn`, `dvn`), k-point arrays, and interpolation coefficients.
- `read_vmtxel(filename, complex_flavor=True)` — reads optical matrix elements from `vmtxel`.
- `dtmat_to_hdf5(in_path, out_path)` / `vmtxel_to_hdf5(in_path, out_path)` — dump to self-describing HDF5.

Used internally by `elph_xml_to_h5.py` (interpolation stage).

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

# 1-2. Assemble coarse el-ph and interpolate to the fine grid in one run
python $ESF/elph/elph_xml_to_h5.py \
    --elph_dir /path/to/dfpt/_ph0/mos2.phsave \
    --wfn_origin WFN_co.h5 \
    --dtmat dtmat \
    --wfn_to_interpolate WFN_fi.h5

# 3. Compute forces
python $ESF/main/excited_forces.py
```

### Second-order el-ph (optional)

```bash
python $ESF/elph/elph_coeffs_second_derivative.py \
    --elph_fine elph_interpolated_kgrid.h5 \
    --eqp eqp1.dat \
    --Nval 13 \
    --out 2nd_order_elph_interpolated_kgrid.h5
```

Then set in `forces.inp`:
```
elph_fine_h5_file              2nd_order_elph_interpolated_kgrid.h5
use_second_derivatives_elph_coeffs  True
```

---

## Notes

- **Nval**: the total number of valence bands included in the DFPT calculation (`nbnd` in `scf.in`/`WFN_co.h5` up to and including the HOMO). It determines which rows/columns of the raw QE-ordered el-ph belong to the conduction vs. valence sector. Preferred source: `WFN_co.h5`'s `mf_header/kpoints/ifmax` (authoritative — no guessing). Fallback: `scf.out`'s "number of electrons" line, or pseudopotential `Z_valence` if no `scf.out` is found. `--Nval` always overrides.
- **Band ordering** in all output files follows the BerkeleyGW convention: valence index 0 = HOMO, 1 = HOMO-1, …; conduction index 0 = LUMO, 1 = LUMO+1, …
- **Cartesian vs. mode basis**: both `elph_orig_kgrid.h5` and `elph_interpolated_kgrid.h5` contain both. The Cartesian basis (`_cart` datasets) is needed for forces in the atomic basis; the mode basis (`_mode` datasets) is needed for forces resolved by phonon mode and frequency. Both are used by `excited_forces.py`.
- **Acoustic sum rule**: applied by default during the assembly stage. Disable with `--no-ASR` if you want the raw uncorrected couplings.
- **Units**: el-ph matrix elements throughout are in Ry/bohr (first order) or Ry/bohr² (second order). Energies in `eqp1.dat` are in eV and are converted to Ry internally where needed.
- **Coarse band count mismatch (dtmat vs. coarse el-ph)**: `dtmat`'s coarse conduction/valence band counts come from `number_cond_bands_coarse`/`number_val_bands_coarse` in `absorption.inp`, which are usually a *subset* of the bands available in `WFN_co.h5`/the coarse el-ph (e.g. `WFN_co.h5` may have far more total bands than are actually used for BSE). This is expected and handled automatically: `elph_xml_to_h5.py` truncates to the bands closest to the band edge (lowest conduction, highest valence) if more are available than `dtmat` needs, or zero-pads if fewer are available — either way it prints a `NOTE:`/warning naming the counts involved.
