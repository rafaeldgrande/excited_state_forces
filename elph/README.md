# elph

Scripts for assembling, interpolating, and processing electron-phonon (el-ph) matrix elements from Quantum ESPRESSO DFPT calculations. These scripts prepare the `elph.h5` file consumed by `main/excited_forces.py`.

---

## Workflow Overview

`elph_xml_to_h5.py` has two modes:

**Default mode** — el-ph was computed via DFPT directly on the same grid the
BSE calculation uses. No interpolation needed.

```
QE DFPT output              BerkeleyGW output
(_ph0/, matdyn.modes,      (WFN.h5, or scf.in)
 scf.in fallback)          (eqp.dat)
        │                            │
        ▼                            ▼
              elph_xml_to_h5.py (default: no --interpolate_elph_coeffs)
                    assembly + band-match/QP-rescale
                                → elph.h5  (fine k-grid)
                                           │
                                           ▼  (optional)
                         elph_coeffs_second_derivative.py
                                → 2nd_order_elph.h5
                                           │
                                           ▼
                                main/excited_forces.py
```

**`--interpolate_elph_coeffs` mode** — el-ph was computed via DFPT on a
*coarser* grid, and needs BerkeleyGW's dtmat-based coarse-to-fine
interpolation to the fine grid the BSE calculation actually uses:

```
QE DFPT output              BerkeleyGW output           BerkeleyGW output
(_ph0/, matdyn.modes,      (WFN_co.h5, or scf.in)      (dtmat, WFN_fi.h5,
 scf.in fallback)                                        eqp.dat)
        │                            │                            │
        ▼                            ▼                            │
                     elph_xml_to_h5.py --interpolate_elph_coeffs    │
        assembly stage → elph_coarse.h5 (coarse k-grid)            │
                    │                                              │
                    └──────── interpolation stage ◄─────────────────┘
                                → elph.h5  (fine k-grid)
```

Either mode can be resumed: omitting `--elph_dir` (with
`--interpolate_elph_coeffs` and an existing `--elph_coarse`) resumes
interpolation-only from a previously-assembled file.

---

## Scripts

### `elph_xml_to_h5.py`

**Assembly stage** — reads QE DFPT el-ph XML files, rotates from the
symmetry-adapted pattern basis to the Cartesian atomic-displacement basis
(and, if `matdyn.modes` is available, to the phonon-mode basis).

**Default mode (no `--interpolate_elph_coeffs`)** — el-ph is assumed to have
been computed directly on the grid the BSE calculation uses. The assembled
data is band-matched to `--eqp`'s Nc/Nv window and QP-rescaled, then written
straight to `--elph_out` (default `elph.h5`), ready to use as
`elph_fine_h5_file` in `forces.inp`. No `dtmat`/coarse-to-fine transformation
involved at all.

**`--interpolate_elph_coeffs` mode** — el-ph was computed via DFPT on a
coarser grid. The assembled data is written to `--elph_coarse` first, then
interpolated to the fine BSE k-grid using the BerkeleyGW coarse-to-fine
transformation matrices stored in `dtmat`, band-matched/QP-rescaled using
`--eqp`, and written to `--elph_out`:

$$\langle n, \mathbf{k}_{\rm fi}+\mathbf{q} \mid \delta V(\mathbf{q}) \mid m, \mathbf{k}_{\rm fi} \rangle = \sum_{ab} \langle n, \mathbf{k}_{\rm fi}+\mathbf{q} \mid a, \mathbf{k}_{\rm co}+\mathbf{q} \rangle\, g_{ab}(\mathbf{k}_{\rm co}, \mathbf{q})\, \langle b, \mathbf{k}_{\rm co} \mid m, \mathbf{k}_{\rm fi} \rangle$$

where $\mathbf{k}_{\rm co}$ is the nearest coarse k-point to $\mathbf{k}_{\rm fi}$ and the overlaps $\langle n, \mathbf{k}_{\rm fi} \mid a, \mathbf{k}_{\rm co} \rangle$ come from `dtmat`.

**Valence-band ordering:** BerkeleyGW convention throughout — index 0 = HOMO, index 1 = HOMO-1, etc. (conduction: index 0 = LUMO, 1 = LUMO+1, ...).

**Prerequisites (assembly, i.e. when `--elph_dir` is given):**
- A completed `ph.x` run with `electron_phonon='simple'` and `ldisp=.true.` (or `nosym=.true.`)
- `_ph0/<prefix>.phsave/elph.iq.ipert.xml` for every (q, perturbation) pair
- `_ph0/<prefix>.phsave/patterns.iq.xml` for every q-point
- `_ph0/<prefix>.phsave/control_ph.xml`
- Cell, k-points, `nbnd`, and Nval (highest occupied band index): from
  `WFN.h5` of the grid DFPT was run on (`--wfn_dfpt`, preferred — authoritative,
  no guessing) or from `scf.in`/`bands.in` (`--qe_input`, fallback — Nval is
  auto-derived from `scf.out`'s "number of electrons" line, or from
  pseudopotential `Z_valence` if no `scf.out` is found). `--Nval` always
  overrides either source manually.
- `matdyn.modes` (optional — produced by `matdyn.x`; enables phonon-mode projection)
- `eqp.dat` (`--eqp`) — fine-grid QP energies from `inteqp.x`, for band-matching + QP rescaling. Read by default; without it, `--elph_out` is still written but unmatched/un-rescaled.

**Prerequisites (`--interpolate_elph_coeffs` only):**
- `dtmat`, produced by BerkeleyGW's `absorption.<flavour>.x`
- `WFN_fi.h5` (`--wfn_absorption_fine`) — the fine grid actually used by the BSE calculation, for finite-q interpolation (auto-discovered next to `--dtmat` if not given)

**Usage:**

```bash
# DEFAULT: el-ph computed directly on the fine grid the BSE calculation uses.
# --eqp band-matches + QP-rescales the result and writes elph.h5.
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_dfpt WFN_fi.h5 --eqp eqp.dat

# Same, but without --eqp: elph.h5 is still written, just unmatched/un-rescaled
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_dfpt WFN_fi.h5

# OPT-IN: el-ph computed on a coarser grid, needs coarse-to-fine interpolation
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_dfpt WFN_co.h5 \
    --interpolate_elph_coeffs --dtmat dtmat --wfn_absorption_fine WFN_fi.h5 --eqp eqp.dat

# Resume: interpolate from a previously-written elph_coarse.h5
python elph_xml_to_h5.py --elph_coarse elph_coarse.h5 --wfn_dfpt WFN_co.h5 \
    --interpolate_elph_coeffs --dtmat dtmat --wfn_absorption_fine WFN_fi.h5

# No WFN.h5 available: fall back to scf.in / scf.out / pseudopotentials
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --qe_input scf.in --eqp eqp.dat

# Manual Nval override (takes precedence over WFN.h5 / scf.in either way)
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_dfpt WFN_fi.h5 --Nval 13

# Disable acoustic sum rule (default: ASR applied, assembly stage only)
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_dfpt WFN_fi.h5 --no-ASR

# Also write elph_not_filtered.h5 (ALL available DFPT bands, no Nc/Nv windowing) --
# feed this into elph_coeffs_second_derivative.py for a better-converged
# intermediate-state sum. Default mode only (no --interpolate_elph_coeffs).
python elph_xml_to_h5.py --elph_dir _ph0/mos2.phsave --wfn_dfpt WFN_fi.h5 --eqp eqp.dat \
    --save_not_filtered
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--elph_dir` | `None` | phsave dir; if given, the assembly stage runs. If omitted, `--elph_coarse` must point to an existing assembled file and `--interpolate_elph_coeffs` must be set (resume: interpolation-only) |
| `--modes_file` | `matdyn.modes` | phonon eigenvectors/frequencies from `matdyn.x` (optional) |
| `--no-ASR` | ASR on | Disable the acoustic sum rule (assembly stage only) |
| `--wfn_dfpt` | `None` | Path to the `WFN.h5` of the grid DFPT was actually run on. Default mode: this IS the fine grid. `--interpolate_elph_coeffs` mode: this is the COARSE grid |
| `--qe_input` | `scf.in` | QE pw.x input file used for cell/k-points/`nbnd`/Nval when `--wfn_dfpt` is not given or not found |
| `--Nval` | `None` | Manual override for Nval (highest occupied band index, QE `nbnd` convention). Takes precedence over `--wfn_dfpt`/`--qe_input` either way |
| `--elph_out` | `elph.h5` | Final output HDF5 filename, ready to use as `elph_fine_h5_file` in `forces.inp` |
| `--interpolate_elph_coeffs` | off | El-ph was computed on a coarser grid and needs coarse-to-fine interpolation; requires `--dtmat` and `--wfn_absorption_fine`. Default (off): el-ph assumed already on the fine grid, no interpolation |
| `--elph_coarse` | `elph_coarse.h5` | Output path for the assembled (pre-interpolation) file; only used with `--interpolate_elph_coeffs` |
| `--dtmat` | `dtmat` | Path to BerkeleyGW `dtmat` binary; only used with `--interpolate_elph_coeffs` |
| `--wfn_absorption_fine` | `WFN_fi.h5` | Path to the fine-grid `WFN.h5` actually used by the BSE/absorption calculation; only used with `--interpolate_elph_coeffs` (auto-discovered next to `--dtmat` if not found) |
| `--real` | off | Use real-flavor `dtmat` (default: complex) |
| `--eqp` | `eqp.dat` | Path to the fine-grid `eqp.dat` (output of `inteqp.x`). When found, band-matches the el-ph to its Nc/Nv window and saves QP rescaling matrices and Eqp/Edft energies into `--elph_out`. Read by default in both modes |
| `--save_not_filtered` | off | Also write `--elph_not_filtered` with ALL available cond/val bands from DFPT (no Nc/Nv windowing). Default (non-interpolated) mode only |
| `--elph_not_filtered` | `elph_not_filtered.h5` | Output path for the unfiltered file; only used with `--save_not_filtered` |

**Output HDF5 layout** (`--elph_coarse` and `--elph_out` share the exact
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
| `crystal/atomic_numbers` | `(Nat,)` | — | Atomic numbers, from `--wfn_dfpt`'s `WFN.h5` or `scf.in` |
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
to the working directory: `qpoints_cart_dfpt.dat`/`qpoints_crystal_dfpt.dat`
(written once `--elph_coarse` is available, whether freshly assembled or
resumed) and `qpoints_cart_interpolated.dat`/`qpoints_crystal_interpolated.dat`
(written after the interpolation stage completes). q-points don't change
between the coarse and fine grids — only k-points do — so the `_dfpt`/
`_interpolated` pairs carry the same values, one written per stage.

---

### `elph_coeffs_second_derivative.py`

Computes second-order electron-phonon coupling coefficients via second-order perturbation theory. The output file has the same format as `elph.h5` and can be used directly as `elph_fine_h5_file` in `forces.inp` with `use_second_derivatives_elph_coeffs True`.

**Theory:**

$$g^{(2)}_{\alpha, \mathbf{k}, nm} = -\sum_l g_{\alpha \mathbf{k} nl}\, g_{\alpha \mathbf{k} lm} \left( \frac{1}{\varepsilon_{n\mathbf{k}} - \varepsilon_{l\mathbf{k}}} + \frac{1}{\varepsilon_{m\mathbf{k}} - \varepsilon_{l\mathbf{k}}} \right)$$

where $\alpha$ is a Cartesian atomic-displacement index, and $\varepsilon$ are quasiparticle energies (from `eqp1.dat`, converted to Ry internally). The input el-ph is taken directly from `elph_cond_cart` / `elph_val_cart` — no displacement-pattern rotation needed since those datasets are already in the Cartesian basis.

Units of the result: Ry/bohr².

The computation is vectorized: for each q-point and k-point,

$$g^{(2)} = \underbrace{-(g \odot \Lambda) \cdot g}_{\text{term 1}} + \underbrace{g \cdot (g \odot \Lambda)}_{\text{term 2}}, \quad \Lambda_{nm} = \frac{1}{\varepsilon_n - \varepsilon_m}$$

where $\odot$ is element-wise multiplication and $\cdot$ is matrix multiplication over band indices.

After computing $g^{(2)}$ in the Cartesian basis, it is projected to the phonon-mode basis using the phonon eigenvectors from `phonon_modes/eigenvectors`.

The intermediate-state sum over $l$ runs over whatever band range `--elph_fine`
provides. With plain `elph.h5` that's just the BSE `Nc`/`Nv` window — usually
too narrow for a converged sum, since DFPT normally computes far more bands
than the BSE calculation uses. For a better-converged sum, generate
`elph_not_filtered.h5` via `elph_xml_to_h5.py --save_not_filtered` (all
DFPT-available bands, cond/val split but not windowed) and pass that as
`--elph_fine` instead — the script sums $l$ over the full available range and
truncates the *output* $g^{(2)}$ back down to `eqp.dat`'s window before
saving, so `--out` stays a drop-in replacement for `elph.h5` in `forces.inp`
either way. Bands outside `eqp.dat`'s window have no GW energy (GW isn't run
there), so `--wfn_dfpt` is required in that case to supply DFT eigenvalues as
the energy denominator for those extra intermediate states (in-window bands
still use `eqp.dat`'s QP energies).

**QP renormalization of the 1st-derivative el-ph (`--renormalize_elph_with_Eqp`):**
optionally rescales $g$ itself by the QP/DFT ratio
$(E^{\rm QP}_{kn} - E^{\rm QP}_{km})/(E^{\rm DFT}_{kn} - E^{\rm DFT}_{km})$
before the sum above — this must happen *before* the sum, not after, since
$g^{(2)}$ is quadratic in $g$. Band pairs with no real QP data (out-of-window
bands when using `elph_not_filtered.h5`, or any near-degenerate DFT pair) get
ratio 1.0, i.e. the raw DFT el-ph is used unchanged for those. Uses the same
ratio formula as `elph_xml_to_h5.py`'s `QP_rescaling_matrix_cond`/`_val`
(`build_qp_rescaling_ratio`, shared between both scripts) — but note that
`elph_xml_to_h5.py` only ever *saves* that ratio matrix for downstream use, it
never applies it to the el-ph itself; this flag is what actually applies it,
and only here, before the second-order sum.

**Usage:**

```bash
# Plain (elph.h5): intermediate sum stays limited to the BSE Nc/Nv window
python elph_coeffs_second_derivative.py \
    --elph_fine elph.h5 \
    --eqp eqp1.dat \
    --out 2nd_order_elph.h5

# Better-converged: full DFPT band range as intermediate states, truncated
# back to eqp1.dat's window on output, plus QP-renormalized el-ph going into
# the sum
python elph_coeffs_second_derivative.py \
    --elph_fine elph_not_filtered.h5 \
    --eqp eqp1.dat \
    --wfn_dfpt WFN_fi.h5 \
    --renormalize_elph_with_Eqp \
    --out 2nd_order_elph.h5
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--elph_fine` | `elph.h5` | Input from `elph_xml_to_h5.py` (`elph.h5` or `elph_not_filtered.h5`) |
| `--eqp` | `eqp1.dat` | Fine-grid QP energy file (from BerkeleyGW `absorption`) |
| `--Nval` | `None` | Number of valence bands in DFPT. If omitted, read from `--elph_fine`'s stored `Nval` attribute |
| `--wfn_dfpt` | `None` | `WFN.h5` providing DFT eigenvalues for bands outside `eqp.dat`'s Nc/Nv window. Only required when `--elph_fine` has more bands than `eqp.dat` covers (e.g. `elph_not_filtered.h5`) |
| `--renormalize_elph_with_Eqp` | off | Rescale the 1st-derivative el-ph by the QP/DFT ratio before the second-order sum (see above) |
| `--out` | `2nd_order_elph.h5` | Output filename |

Then in `forces.inp`:
```
elph_fine_h5_file              2nd_order_elph.h5
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

# 1. Assemble el-ph directly on the fine grid used by the BSE calculation
#    (default mode: no coarse-to-fine interpolation)
python $ESF/elph/elph_xml_to_h5.py \
    --elph_dir /path/to/dfpt/_ph0/mos2.phsave \
    --wfn_dfpt WFN_fi.h5 \
    --eqp eqp.dat

# 2. Compute forces
python $ESF/main/excited_forces.py
```

If el-ph was instead computed via DFPT on a coarser grid, add
`--interpolate_elph_coeffs --dtmat dtmat --wfn_absorption_fine WFN_fi.h5`
(with `--wfn_dfpt` now pointing at the coarse grid, `WFN_co.h5`) to step 1.

### Second-order el-ph (optional)

```bash
# Add --save_not_filtered to step 1 for a better-converged intermediate sum
# (see elph_coeffs_second_derivative.py section above), then:
python $ESF/elph/elph_coeffs_second_derivative.py \
    --elph_fine elph_not_filtered.h5 \
    --eqp eqp1.dat \
    --wfn_dfpt WFN_fi.h5 \
    --out 2nd_order_elph.h5
```

Then set in `forces.inp`:
```
elph_fine_h5_file              2nd_order_elph.h5
use_second_derivatives_elph_coeffs  True
```

---

## Notes

- **Nval**: the total number of valence bands included in the DFPT calculation (`nbnd` in `scf.in`/`WFN.h5` up to and including the HOMO). It determines which rows/columns of the raw QE-ordered el-ph belong to the conduction vs. valence sector. Preferred source: `--wfn_dfpt`'s `mf_header/kpoints/ifmax` (authoritative — no guessing). Fallback: `scf.out`'s "number of electrons" line, or pseudopotential `Z_valence` if no `scf.out` is found. `--Nval` always overrides.
- **Band ordering** in all output files follows the BerkeleyGW convention: valence index 0 = HOMO, 1 = HOMO-1, …; conduction index 0 = LUMO, 1 = LUMO+1, …
- **Cartesian vs. mode basis**: both `--elph_coarse` and `--elph_out` contain both. The Cartesian basis (`_cart` datasets) is needed for forces in the atomic basis; the mode basis (`_mode` datasets) is needed for forces resolved by phonon mode and frequency. Both are used by `excited_forces.py`.
- **Acoustic sum rule**: applied by default during the assembly stage. Disable with `--no-ASR` if you want the raw uncorrected couplings.
- **Units**: el-ph matrix elements throughout are in Ry/bohr (first order) or Ry/bohr² (second order). Energies in `eqp1.dat` are in eV and are converted to Ry internally where needed.
- **Band-window mismatch (Nc/Nv vs. the BSE calculation)**: whether via `dtmat` (in `--interpolate_elph_coeffs` mode, where the coarse conduction/valence band counts come from `number_cond_bands_coarse`/`number_val_bands_coarse` in `absorption.inp`) or via `--eqp`'s own band window (default mode), the assembled el-ph usually has *more* bands available than the BSE calculation actually uses (e.g. the DFPT run may have far more total bands than are used for BSE). This is expected and handled automatically: `elph_xml_to_h5.py` truncates to the bands closest to the band edge (lowest conduction, highest valence) if more are available than needed, or zero-pads if fewer are available — either way it prints a `NOTE:`/warning naming the counts involved.
- **Not-filtered el-ph and the second-order sum**: `elph.h5`'s `elph_cond_cart`/`elph_val_cart` are always windowed to the BSE `Nc`/`Nv` (via `--eqp`) — fine for first-order forces, but the intermediate-state sum in `elph_coeffs_second_derivative.py` is then artificially confined to that narrow window. `--save_not_filtered` gives that script the full DFPT band range to sum over instead. Note this still keeps the conduction and valence manifolds separate (the intermediate state `l` never crosses between them) — the cross-manifold (cond-val) coupling terms of the raw QE el-ph matrix are discarded during assembly and are not currently reconstructable from either output file.
