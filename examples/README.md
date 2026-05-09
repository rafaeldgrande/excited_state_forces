# examples

Two reference examples covering the complete DFT → GW → BSE → excited-state forces workflow. Both use Quantum ESPRESSO (QE) for DFT/DFPT and BerkeleyGW (BGW) for GW and BSE.

---

## Systems

| | CO | LiF |
|---|---|---|
| Type | Isolated molecule in a box | Bulk crystal (rock-salt) |
| Atoms | C, O (2 atoms) | Li, F (2 atoms, primitive FCC cell) |
| k-grid | Γ only (1 k-point) | 4×4×4 = 64 k-points, `nosym=.true.` |
| Lattice | Cubic, $a = 10$ Å | FCC, $a = 4.059$ Å |
| Cutoff | 100 Ry (WFN) | 60 Ry |
| Val bands (BSE) | 5 | 5 |
| Cond bands (BSE) | 13 | 7 |
| Truncation | Box truncation | Full dielectric + box |
| DFPT phonon | q = Γ only | q = Γ only |

---

## Directory Structure

### CO

```
CO/
├── C.upf, O.upf              Pseudopotentials
├── create_links.bash          Sets up symlinks between steps
├── job_gwbse.sub              SLURM script running steps 1–7 in sequence
│
├── 1-scf/
│   └── scf.in                Ground-state SCF (Γ, nbnd=19, ecutwfc=60)
├── 2-wfn_gw/
│   ├── bands.in              Non-self-consistent bands (nbnd=200, for GW)
│   └── pw2bgw.in             Export WFN + RHO + vxc.dat for BGW
├── 3-wfn_bse/
│   ├── scf.in                SCF for BSE k-grid (nbnd=20, ecutwfc=100)
│   ├── pw2bgw.in             Export WFN for BSE
│   ├── ph.in                 DFPT phonon at Γ (electron_phonon='simple')
│   └── dynmat.in             Dynamical matrix → phonon frequencies/eigenvectors
├── 4-epsilon/
│   └── epsilon.inp           Dielectric matrix (box truncation, 1 q-point)
├── 5-sigma/
│   └── sigma.inp             GW self-energy (bands 1–19, exact static CH)
├── 6-kernel/
│   └── kernel.inp            BSE kernel (5 val, 13 cond)
├── 7-absorption/
│   └── absorption.inp        BSE eigenvectors (write_eigenvectors 20)
└── 8-excited_state_forces/
    ├── forces.inp             Excited-state forces configuration
    └── link_files.bash        Symlinks eigenvectors.h5, eqp.dat, CO.phsave/
```

### LiF

```
LiF/
├── Li.upf, F.upf             Pseudopotentials
├── create_links.bash          Sets up symlinks between steps
├── job_gwbse.sub              SLURM script running steps 1–8 in sequence
│
├── 1-scf_fi/
│   ├── scf.in                SCF on fine 4×4×4 k-grid (nbnd=15, nosym=.true.)
│   ├── ph.in                 DFPT phonon at Γ (electron_phonon='simple')
│   └── pw2bgw.in             Export fine-grid WFN for BSE
├── 2-wfn/
│   ├── bands.in              Fine-grid WFN with many bands (nbnd=100, for GW)
│   └── pw2bgw.in             Export WFN_fi + RHO + vxc.dat
├── 3-wfnq/
│   ├── bands.in              Shifted-q WFN (small q offset for dielectric)
│   └── pw2bgw.in             Export WFNq
├── 4-wfn_co/
│   ├── bands.in              Coarse-grid WFN (4×4×4, nbnd=20, for BSE kernel)
│   └── pw2bgw.in             Export WFN_co
├── 5-epsilon/
│   └── epsilon.inp           Dielectric matrix (64 q-points, full screening)
├── 6-sigma/
│   └── sigma.inp             GW self-energy
├── 7-kernel/
│   └── kernel.inp            BSE kernel (5 val, 7 cond)
├── 8-absorption/
│   └── absorption.inp        BSE eigenvectors (write_eigenvectors 20)
└── 9-excited_state_forces/
    ├── forces.inp             Excited-state forces configuration
    └── link_files.bash        Symlinks eigenvectors.h5, eqp.dat, LiF.phsave/
```

---

## Running the Examples

### Step 1: Set up symlinks

```bash
cd CO/   # or LiF/
bash create_links.bash
```

This links charge densities and WFN files from earlier steps into the directories that need them.

### Steps 2–8: DFT, GW, BSE (on a cluster)

The `job_gwbse.sub` SLURM script runs all QE and BGW steps sequentially. Adapt the partition, node count, and module names for your cluster:

```bash
# Edit job_gwbse.sub to match your cluster, then:
sbatch job_gwbse.sub
```

The script runs (in order):

| Step | Code | Action |
|------|------|--------|
| 1 | `pw.x` | Ground-state SCF |
| 2 | `pw.x` + `pw2bgw.x` | Many-band WFN for GW |
| 3 | `pw.x` + `pw2bgw.x` + `ph.x` + `dynmat.x` | BSE WFN + DFPT phonon |
| 4 | `epsilon.cplx.x` | Dielectric matrix |
| 5 | `sigma.cplx.x` | GW self-energy → `eqp1.dat` |
| 6 | `kernel.cplx.x` | BSE kernel → `bsemat.h5` |
| 7/8 | `absorption.cplx.x` | BSE → `eigenvectors.h5`, `eqp.dat` |

### Step 8: Assemble and interpolate el-ph

After the DFPT phonon run completes, assemble the coarse-grid el-ph and interpolate it to the fine BSE k-grid:

```bash
ESF=/path/to/excited_state_forces

# CO example (run from 3-wfn_bse/, which contains scf.in and _ph0/)
cd CO/3-wfn_bse/
python $ESF/elph/assemble_elph_h5.py
# → produces elph.h5

# Interpolate to fine grid (run from the forces directory)
cd ../8-excited_state_forces/
bash link_files.bash   # symlink eigenvectors.h5 and eqp.dat

python $ESF/elph/interpolate_elph_bgw.py \
    --elph_coarse ../3-wfn_bse/elph.h5 \
    --dtmat dtmat \
    --Nval 5
# → produces elph_fine.h5
```

> **Note:** `dtmat` is produced by `absorption.cplx.x` and lives in `7-absorption/` (CO) or `8-absorption/` (LiF). Add a symlink: `ln -sf ../7-absorption/dtmat .`

### Step 9: Compute excited-state forces

Update `forces.inp` to use the new el-ph file (see note below), then run:

```bash
python $ESF/main/excited_forces.py
```

---

## `forces.inp` — Current vs. Updated Format

> **The `forces.inp` files in these examples use the old `el_ph_dir` parameter, which is no longer supported.** The code now reads a pre-interpolated `elph_fine.h5` file produced by `elph/interpolate_elph_bgw.py`.

Replace the old format:
```
# OLD (no longer works)
el_ph_dir   CO.phsave/
```

With the new format:
```
# NEW
elph_fine_h5_file   elph_fine.h5
```

A minimal `forces.inp` for these examples:

```
iexc                1
eqp_file            eqp.dat
exciton_file        eigenvectors.h5
elph_fine_h5_file   elph_fine.h5
```

See [`main/README.md`](../main/README.md) for the full list of `forces.inp` parameters.

---

## Key Calculation Parameters

### CO

| Parameter | Value | Note |
|-----------|-------|------|
| Cell | Cubic, 10 Å | Box truncation |
| DFPT Ecut | 60 Ry (GS SCF), 100 Ry (BSE SCF) | Higher cutoff for BSE WFN |
| `nbnd` (GW) | 200 | Many empty bands for self-energy sum |
| `nbnd` (BSE SCF) | 20 | 5 val + 13 cond + 2 extra |
| Nval (DFPT) | 7 | 5 core + 2 C 2s → adjust to C 1s+2s+2p, O 1s+2s+2p |
| BSE val / cond | 5 / 13 | |
| phonon q | Γ only | Single q-point |
| `tr2_ph` | 1×10⁻¹⁸ | Tight DFPT convergence |

### LiF

| Parameter | Value | Note |
|-----------|-------|------|
| Cell | FCC primitive, $a = 4.059$ Å | `ibrav=0`, explicit CELL_PARAMETERS |
| `nosym` | `.true.` | Required for BGW k-grid compatibility |
| `nbnd` (GW) | 100 | Fine-grid bands |
| `nbnd` (coarse) | 20 | Coarse-grid WFN for BSE |
| Nval (DFPT) | 5 | Li 1s+2s + F 1s+2s+2p (approximate; verify) |
| BSE val / cond | 5 / 7 | |
| phonon q | Γ only | Single q-point |
| ε q-grid | 64 q-points (4×4×4) | Full dielectric matrix |

---

## Notes

- **`nosym = .true.`** is required in the QE inputs for periodic systems when the k-grid must be fully unfolded for BGW. CO does not need this since it uses Γ only.
- **`cell_box_truncation`** in BGW inputs applies a box Coulomb truncation to suppress spurious image interactions for the CO molecule. For bulk LiF this is not needed and not used in `epsilon.inp` / `sigma.inp`.
- **Phonon convergence**: both examples use `tr2_ph=1d-18`, which is tighter than the QE default. This is recommended when the el-ph matrix elements are used for excited-state properties.
- **`dynmat.x`** (CO only): applies the acoustic sum rule to the dynamical matrix and produces eigenvectors. Its output is not directly consumed by the Python scripts — `assemble_elph_h5.py` reads `matdyn.modes` (from `matdyn.x`) if present, or computes without phonon-mode projection otherwise.
- **Pseudopotentials** (`*.upf`) are stored at the top level of each example directory and referenced via `pseudo_dir = '../'` in the QE inputs.
