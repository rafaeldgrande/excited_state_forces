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

### Polarized Raman

`resonant_raman.py --polarized` computes angle-resolved polarized intensities $I_\parallel(\theta,\Omega,\omega)$ and $I_\perp(\theta,\Omega,\omega)$ for any of the 9 flavors, for an incident linear polarization $\hat{e}_i(\theta)$ rotating in the scattering plane.

**Master equation.** For incident polarization $\hat{e}_i$ and scattered (analyzer) polarization $\hat{e}_s$:

$$I(\hat{e}_i,\hat{e}_s;\Omega,\omega) = \sum_\nu \left| w_\nu \, M_\nu(\hat{e}_i,\hat{e}_s;\Omega) \right|^2 L(\omega-\omega_\nu), \qquad M_\nu(\hat{e}_i,\hat{e}_s;\Omega) = \sum_{\alpha\beta} e_s^{\alpha *}\, \alpha^{\alpha\beta}_\nu(\Omega)\, e_i^{\beta}$$

with the same $w_\nu$ and $L(\omega)=\gamma_L^2/(\omega^2+\gamma_L^2)$ as above. For second order, the analogous mode-pair contraction:

$$M_{\nu\nu'}(\hat{e}_i,\hat{e}_s;\Omega) = \sum_{\alpha\beta} e_s^{\alpha *}\, M^{\alpha\beta}_{\nu\nu'}(\Omega)\, e_i^{\beta}, \qquad I^{(2)} = \sum_{\nu\nu'} \left| w_\nu w_{\nu'}\, M_{\nu\nu'} \right|^2 L(\omega-\omega_\nu-\omega_{\nu'})$$

**Critical ordering constraint — contract, then square, then sum over modes.** The Cartesian path above (`raman_maps`) squares each $\alpha^{\alpha\beta}_\nu$ component *individually*, discarding the relative phases between e.g. $\alpha^{xx}_\nu$ and $\alpha^{xy}_\nu$ — precisely the phases that produce angular structure in the resonant regime. **$I_\parallel(\theta)$ cannot be reconstructed from `raman_maps` or `resonant_raman_data_flavor{N}.h5`** — the polarized path reads the complex tensors directly (`resonant_raman/polarization.py`'s `contract_first_order`/`contract_second_order`) and contracts before squaring. The mode sum itself stays incoherent, as above (a sum over distinct final states), and first- and second-order contributions are still summed incoherently, matching the existing (Cartesian) behavior — see the Notes below.

**Frame construction.** Given a scattering-plane normal $\hat{n}$ (`--n-hat`, default $\hat{z}$), build a right-handed orthonormal triad $\{\hat{e}_1,\hat{e}_2,\hat{n}\}$:

$$\hat{e}_1 = \frac{\hat{r} - (\hat{r}\cdot\hat{n})\,\hat{n}}{\lVert \hat{r} - (\hat{r}\cdot\hat{n})\,\hat{n}\rVert}, \qquad \hat{e}_2 = \hat{n}\times\hat{e}_1$$

where $\hat{r}$ (`--theta-ref`) fixes the $\theta=0$ direction — default $\hat{x}$, falling back to $\hat{y}$ if $|\hat{r}\cdot\hat{n}| > 0.9$. Polarization vectors:

$$\hat{e}_i(\theta) = \cos\theta\,\hat{e}_1 + \sin\theta\,\hat{e}_2, \qquad \hat{e}_s^{\parallel}(\theta) = \hat{e}_i(\theta), \qquad \hat{e}_s^{\perp}(\theta) = \hat{e}_i(\theta+\tfrac{\pi}{2})$$

The $\theta$ grid (`--dtheta`, default $2\pi/100$ rad) is half-open, `theta = np.arange(Ntheta) * dtheta` — **not** `linspace(0, 2*pi, N)`, whose duplicated endpoint would corrupt any $\theta$-average; the curve is closed only at plot time.

> **Caveat**: this is a purely geometric projection of the Raman tensor. It does not include optical propagation effects — refraction at the surface, birefringence, anisotropic absorption — which distort measured polar plots in low-symmetry or strongly absorbing crystals. Comparisons to experiment on such systems need an effective-tensor correction that is out of scope here.

**Isotropic average (randomly-oriented samples, e.g. molecules).** $\theta$ is not meaningful for a powder/gas-phase sample; the observable is the depolarization ratio. Splitting the unpolarized invariant above:

$$I_\parallel^{\rm iso} = 45\vert\bar\alpha\vert^2 + 4\gamma^2, \qquad I_\perp^{\rm iso} = 3\gamma^2 + 5\delta^2, \qquad \rho = \frac{I_\perp^{\rm iso}}{I_\parallel^{\rm iso}}$$

(`common/utils.py`'s `isotropic_parallel`, `isotropic_perpendicular`, `depolarization_ratio` — their sum reproduces the existing `unpolarized_invariant`). $\rho \to 0$ for totally symmetric modes (e.g. benzene's $a_{1g}$ ring-breathing mode), $\rho \to 3/4$ in the fully depolarized limit.

### Helicity-resolved Raman ($\sigma_+$, $\sigma_-$)

`resonant_raman.py --polarized --helicity` computes the four circular-polarization intensities $I_{\sigma_+\sigma_+}$, $I_{\sigma_+\sigma_-}$, $I_{\sigma_-\sigma_+}$, $I_{\sigma_-\sigma_-}$, following the formulation in Hung *et al.* (2024), "QERaman: an open-source program for calculating resonance Raman spectra based on excited state forces", *Comput. Phys. Commun.*, arXiv:[2308.05900](https://arxiv.org/abs/2308.05900).

**In-plane block: the general polarization machinery.** The contraction $\hat e_s^\dagger\,\alpha_\nu\,\hat e_i$ is bilinear in the polarization vectors, and every polarization of interest — linear at any $\theta$, circular, elliptical — lies in the $\{\hat e_1,\hat e_2\}$ plane already built by `build_frame`. So the entire polarization dependence is carried by one small $2\times2$ complex block per mode (or mode pair):

$$\tilde\alpha^{jk}_\nu(\Omega) = \hat e_j^\dagger \cdot \boldsymbol\alpha_\nu(\Omega) \cdot \hat e_k, \qquad j,k \in \{1,2\}$$

(`polarization.py`'s `alpha_plane_first_order`/`alpha_plane_second_order`). $I_\parallel(\theta)$, $I_\perp(\theta)$, and the four $\sigma_\pm$ intensities are then all the *same* two-line contraction (`contract_plane`) against a pair of 2-vectors $(\hat c_s,\hat c_i)$ — only the vectors change:

| Configuration | $\hat c_s$ | $\hat c_i$ |
|---|---|---|
| $I_\parallel(\theta)$ | $(\cos\theta,\sin\theta)$ | $(\cos\theta,\sin\theta)$ |
| $I_\perp(\theta)$ | $(-\sin\theta,\cos\theta)$ | $(\cos\theta,\sin\theta)$ |
| $\sigma_+\sigma_+$ | $(1,i)/\sqrt2$ | $(1,i)/\sqrt2$ |
| $\sigma_+\sigma_-$ | $(1,-i)/\sqrt2$ | $(1,i)/\sqrt2$ |
| $\sigma_-\sigma_+$ | $(1,i)/\sqrt2$ | $(1,-i)/\sqrt2$ |
| $\sigma_-\sigma_-$ | $(1,-i)/\sqrt2$ | $(1,-i)/\sqrt2$ |

The Cartesian circular-polarization vectors reduce, for $\hat n=\hat z$, $\hat e_1=\hat x$, to the standard Jones vectors $\mathbf P_{\sigma_+} = (1,i,0)/\sqrt2$, $\mathbf P_{\sigma_-} = (1,-i,0)/\sqrt2$. The `.conj()` on the scattered vector in `contract_first_order`/`contract_second_order`/`contract_plane` is load-bearing here (not cosmetic, as it is for real linear polarizations) — dropping it silently gives wrong circular-polarization results.

**⚠ Convention: `jones` vs. `propagation`.** In backscattering the scattered photon propagates along $-\hat n$. Helicity is $\mathbf S\cdot\hat k$, so a scattered photon with the *same* Jones vector as the incident one (in the fixed $\{\hat e_1,\hat e_2\}$ lab basis) has the same spin projection onto $\hat n$ but the **opposite** helicity relative to its own propagation direction. The two conventions disagree on every label:

| `--helicity-convention` | Scattered basis vector labeled $\sigma_+$ | Meaning |
|---|---|---|
| `jones` (default) | $\hat e_+$ | Fixed lab-frame Jones vector; matches QERaman and most 2D-materials literature |
| `propagation` | $\hat e_-$ | True helicity relative to the outgoing $-\hat n$ direction |

Under `jones`, labels match the paper's convention: for monolayer MoS$_2$ the out-of-plane $A_1'$ mode ($\sim403$ cm$^{-1}$) appears in $\sigma_+\sigma_+$ and the in-plane $E'$ mode ($\sim383$ cm$^{-1}$) appears in $\sigma_+\sigma_-$ — the selection rule follows from the mode symmetry's transformation under the $C_3$ rotation that relates $\sigma_\pm$ to angular momentum $\Delta m=0,\pm1$ (Tatsumi & Saito, *Phys. Rev. B* **97**, 115407 (2018)). Under `propagation` these two swap. See the Notes below for the "helicity-conserving ≠ Jones-conserving" naming trap. The resolved convention string is stored as an h5 attribute on every output file.

**Selection rules (synthetic check, `tests/test_polarization.py`).** For an in-plane block $c=a+ib$:

| Mode | In-plane block $\tilde\alpha$ | $I_{\sigma_+\sigma_+}$ | $I_{\sigma_+\sigma_-}$ |
|---|---|---|---|
| $A_1$-like | $\begin{pmatrix}c&0\\0&c\end{pmatrix}$ | $\vert c\vert^2$ | $0$ |
| $E_{(1)}$-like | $\begin{pmatrix}0&c\\c&0\end{pmatrix}$ | $0$ | $\vert c\vert^2$ |
| $E_{(2)}$-like | $\begin{pmatrix}c&0\\0&-c\end{pmatrix}$ | $0$ | $\vert c\vert^2$ |

The zeros are exact under `jones`; the pattern swaps under `propagation` (both columns exchange, since it only relabels which scattered vector is called $\sigma_+$). The same $E$-like blocks also give the closed-form linear intensities $I_{E_{(1)}}(\theta)=4(a^2+b^2)\cos^2\theta\sin^2\theta$, $I_{E_{(2)}}(\theta)=(a^2+b^2)\cos^22\theta$ — their $\theta$-independent sum $a^2+b^2$ is the closed-form version of the degenerate-mode gauge-invariance requirement above.

**Sum rule (the strongest regression check).** Independent of the tensor, since $\{\hat e_+,\hat e_-\}$ is a unitary rotation of $\{\hat e_1,\hat e_2\}$ (Frobenius-norm invariance) and both sides reduce to the same quadratic-form circular average:

$$I_{\sigma_+\sigma_+} + I_{\sigma_+\sigma_-} + I_{\sigma_-\sigma_+} + I_{\sigma_-\sigma_-} = 2\left\langle I_\parallel(\theta) + I_\perp(\theta)\right\rangle_\theta$$

This holds to machine precision for *any* $\theta$ grid of 3 or more evenly-spaced points (not just asymptotically for a fine grid) — a `linspace` with a duplicated endpoint breaks it, which is exactly why the grid convention above matters.

**When are $I_{\sigma_+\sigma_+}=I_{\sigma_-\sigma_-}$ and $I_{\sigma_+\sigma_-}=I_{\sigma_-\sigma_+}$? Checked directly against this project's data — not in general.** Write the in-plane block $\tilde\alpha=\begin{pmatrix}p&q\\r&s\end{pmatrix}$ and expand the four contractions: $I_{\sigma_+\sigma_+}=I_{\sigma_-\sigma_-}$ requires only $\tilde\alpha$ **symmetric** ($q=r$), even with $p,q,s$ fully complex; $I_{\sigma_+\sigma_-}=I_{\sigma_-\sigma_+}$ (the "Reciprocity" test in `tests/test_polarization.py`, §7.4 of `HELICITY_RAMAN_SPEC.md`) needs the *stricter* $\tilde\alpha$ **symmetric AND real**. Symmetry of $\tilde\alpha$ (hence of the Cartesian tensor, $\alpha^{\alpha\beta}=\alpha^{\beta\alpha}$) traces back to whether the exciton dipole matrix elements $\langle0|r_\alpha|A\rangle$ can be taken real — true for a simple two-level system, but **verified false here**: measured on the real MoS$_2$ data (`RESONANT_RAMAN_2nd_ORDER_q0/`), the maximum relative $|I_{\sigma_+\sigma_+}-I_{\sigma_-\sigma_-}|$ is 13% for flavor 3 (1st order, diagonal-only — the term one might most expect to be symmetric), 26% for flavor 4 (diag+offdiag), 49% for flavor 5 (triple resonance), 94% for flavor 6 (double resonance); $I_{\sigma_+\sigma_-}$ vs.\ $I_{\sigma_-\sigma_+}$ differs by comparable or larger amounts throughout. The exciton dipole moments in this pipeline are genuinely complex (three Cartesian components with independent relative phases — a single per-exciton gauge choice cannot make all three real simultaneously), so **none of these equalities should be assumed for any flavor** without checking. `interactive_vis_resonant_map.py --plot-polar-plots` adds all four $\sigma_a\sigma_b$ combinations to the main polarization dropdown (regridded onto the same downsampled axes as `xx`/`xy`/etc., alongside the dedicated helicity panel) specifically so this can be inspected visually per flavor/mode rather than assumed.

**Degree of circular polarization**, for incident $\sigma_+$:

$$\rho_{\rm circ}(\Omega,\omega) = \frac{I_{\sigma_+\sigma_+} - I_{\sigma_+\sigma_-}}{I_{\sigma_+\sigma_+} + I_{\sigma_+\sigma_-}} \in [-1, 1]$$

with the denominator guarded (set to $0$, not divided-by-zero, wherever no mode has spectral weight).

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
elph/interpolate_elph_bgw.py → elph_interpolated_kgrid.h5
```

`forces.inp` must include:
```
elph_fine_h5_file   elph_interpolated_kgrid.h5
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
python $ESF_DIR/resonant_raman/resonant_raman.py --flavor 3
# → raman_map_*.png
```

---

### 2nd Order Resonant Raman

Run from the `2nd_der_exc_ph/` directory. Requires the 1st-order `elph_interpolated_kgrid.h5` from the prior workflow.

```bash
# Step 1: Compute 2nd-order el-ph coefficients via perturbation theory
python $ESF_DIR/elph/elph_coeffs_second_derivative.py \
    --elph_fine ../1st_der_exc_ph/elph_interpolated_kgrid.h5 \
    --eqp eqp1.dat \
    --Nval <Nval> \
    --out 2nd_order_elph_interpolated_kgrid.h5
# → 2nd_order_elph_interpolated_kgrid.h5

# Step 2: Compute 2nd-order exciton-phonon matrix elements
# forces.inp must have:
#   elph_fine_h5_file              2nd_order_elph_interpolated_kgrid.h5
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
    --flavor 7
```

---

## Raman Flavor Index

The `--flavor` argument to `resonant_raman.py` selects which contributions to include. **Renumbered 2026-08-05** — if you have scripts or notes referencing the old 0-8 scheme, see the mapping table right after this one.

| Flavor | Description | Required files |
|--------|-------------|----------------|
| 0 | IPA first order | `--ipa-first-order-file` |
| 1 | IPA second order | `--ipa-second-order-file` |
| 2 | IPA first + second order | `--ipa-first-order-file` and `--ipa-second-order-file` |
| 3 | First order, diagonal exciton-phonon only | `--first-order-file` |
| 4 | First order, diagonal + off-diagonal exciton-phonon | `--first-order-file` |
| 5 | Second order, triple resonance only | `--second-order-file` (or `--q-points-file`) |
| 6 | Second order, double resonance only | `--second-order-file` ($\Gamma$-only, see caveat below) |
| 7 | Second order, double + triple resonance | `--second-order-file` ($\Gamma$-only, see caveat below) |
| 8 | First order (diag+offdiag) + second order (double+triple) | `--first-order-file` and `--second-order-file` ($\Gamma$-only) |

Flavors 3–8 use BSE exciton-phonon matrix elements from `excited_forces.py`. Flavors 0–2 use the Independent Particle Approximation (IPA) computed directly from the interpolated el-ph coefficients in `elph_interpolated_kgrid.h5`.

**$\Gamma$-only caveat (flavors 6, 7, 8)**: double resonance needs the real
2nd-derivative el-ph coefficients (`elph_coeffs_second_derivative.py`),
which in this pipeline are only ever computed at $\Gamma$ ($q{=}0$) — the
`alpha_tensor_double_resonance` dataset in the finite-$q$
`susceptibility_tensors_second_order_q_{iq}.h5` files is present (for
schema uniformity) but identically zero for every $q$, including $q{=}0$'s
copy in that directory. Running flavors 6-8 with `--q-points-file` will
silently give an all-zero double-resonance contribution rather than erroring
— don't do that; use the $\Gamma$-only `--second-order-file` (i.e. the one
in `RESONANT_RAMAN_2nd_ORDER_q0/`-style directories, not
`RESONANT_RAMAN_2nd_ORDER_all_q/`) for these flavors. Only flavor 5 (triple
resonance only) is meaningful with `--q-points-file`.

### Mapping from the pre-2026-08-05 flavor numbers

| Old | New | | Old | New |
|---|---|---|---|---|
| 0 (`d2` only) | 3 | | 5 (triple+double+`d3`) | 8 |
| 1 (`d3` only) | 4 | | 6 (IPA 1st) | 0 |
| 2 (triple only) | 5 | | 7 (IPA 2nd) | 1 |
| 3 (triple+double) | 7 | | 8 (IPA 1st+2nd) | 2 |
| 4 (triple+`d3`, no double) | *(removed — no new equivalent)* | | | |

Old flavor 4 (triple resonance combined with first-order `d3`, but
*without* double resonance) had no clean physical motivation as a
standalone combination and was dropped rather than carried forward.

---

---

### IPA Workflow (flavors 0–2)

The IPA susceptibility tensors are computed directly from `elph_interpolated_kgrid.h5` (produced by `interpolate_elph_bgw.py` with `--eqp`), bypassing the BSE exciton-phonon step. This is faster but omits excitonic effects.

```bash
# First order (gamma point, default iq=0)
python $ESF_DIR/resonant_raman/susceptibility_tensors_IPA.py
# → susceptibility_tensors_first_order_IPA.h5

# Second order at q-point iq
python $ESF_DIR/resonant_raman/susceptibility_tensors_IPA.py \
    --compute_second_order --skip_first_order_calculation --iq 0
# → susceptibility_tensors_second_order_IPA_q_0.h5

# Rename/symlink for resonant_raman.py (which expects the default filename)
ln -sf susceptibility_tensors_second_order_IPA_q_0.h5 \
        susceptibility_tensors_second_order_IPA.h5

# Compute Raman map (IPA first + second order)
python $ESF_DIR/resonant_raman/resonant_raman.py --flavor 2
```

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
| `--output` | `susceptibility_tensors_second_order.h5` | Output filename |
| `--finite-q` | off | Enable finite-q mode: reads `exc_forces.h5` at a finite q-point (exciton-phonon matrix non-Hermitian); uses Q=q energies for the intermediate exciton state |

**Output:**
- Datasets `alpha_tensor_triple_resonance` `(3, 3, Nmodes, Nmodes, Nfreq)` and `alpha_tensor_double_resonance` `(3, 3, Nmodes, Nfreq)` saved to `--output`.

```bash
# Standard (gamma-only) second order
python susceptibility_tensors_second_order.py \
    --first_order_exc_ph_file  ../1st_der_exc_ph/exc_forces.h5 \
    --second_order_exc_ph_file exc_forces.h5 \
    --nworkers 8

# Finite-q: one output file per q-point
python susceptibility_tensors_second_order.py \
    --first_order_exc_ph_file exc_phonon_q_1.h5 \
    --finite-q \
    --output susceptibility_tensors_second_order_q_1.h5
```

---

### `resonant_raman.py`

Computes resonant Raman intensity maps (Raman shift vs. excitation energy) from susceptibility tensors. Supports different combinations of 1st and 2nd order contributions via `--flavor`.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--first-order-file` | `susceptibility_tensors_first_order.h5` | 1st-order susceptibility file |
| `--second-order-file` | `susceptibility_tensors_second_order.h5` | 2nd-order susceptibility file (single q-point) |
| `--q-points-file` | `None` | File with `qx qy qz weight` rows for BZ-averaged 2nd-order Raman; row `iq` → `susceptibility_tensors_second_order_q_{iq}.h5` |
| `--ipa-first-order-file` | `susceptibility_tensors_first_order_IPA.h5` | IPA 1st-order file (flavors 6, 8) |
| `--ipa-second-order-file` | `susceptibility_tensors_second_order_IPA.h5` | IPA 2nd-order file (flavors 7, 8) |
| `--freqs-file` | `freqs.dat` | Phonon frequencies (cm⁻¹) |
| `--flavor` | `0` | Contribution flavor (see table above) |
| `--temperature` | `300` | Temperature in Kelvin |
| `--nfreq-ph` | `500` | Number of phonon frequency points for spectrum |
| `--output` | `resonant_raman_data.h5` | Output HDF5 file |
| `--plot-map-log-scale` | off | Use log scale for the Raman map |
| `--polarized` | off | Also compute angle-resolved $I_\parallel(\theta)$/$I_\perp(\theta)$ (see "Polarized Raman" under Theory) |
| `--dtheta` | $2\pi/100$ | $\theta$ step in radians |
| `--dtheta-deg` | `None` | $\theta$ step in degrees; overrides `--dtheta` |
| `--n-hat` | `0 0 1` | Scattering-plane normal, 3 floats (Cartesian) |
| `--theta-ref` | `None` (auto) | Reference vector fixing $\theta=0$ |
| `--polar-output` | `resonant_raman_polar_flavor{N}.h5` | Output file for polarized amplitudes |
| `--polar-store-maps` | off | Also store full $(\theta,\Omega,\omega)$ intensity maps (float32; can be large) |
| `--helicity` | off | Also compute the four $\sigma_\pm$ helicity-resolved intensities (see "Helicity-resolved Raman" under Theory); requires `--polarized` |
| `--helicity-convention` | `jones` | `jones` or `propagation` — which scattered Jones vector is labeled $\sigma_+$ (see Theory) |
| `--jones-incident` | `None` | Explicit incident Jones vector `Px Py phi` (phi in radians), overriding the $\sigma_\pm$ presets |
| `--jones-scattered` | `None` | Same, for the scattered polarization |

**Output:**
- `resonant_raman_data_flavor{N}.h5` — Raman map data
- `raman_map_{pol}_flavor_{N}.png` — polarization-resolved Raman maps
- `raman_map_unpolarized_flavor_{N}.png` — unpolarized Raman map
- `resonant_raman_polar_flavor{N}.h5` — polarized amplitudes (with `--polarized`; see the schema in `plotting/plot_polar_raman.py`'s entry below), extended with the $\sigma_\pm$ maps when `--helicity` is also set

```bash
# 1st-order BSE (diagonal only)
python resonant_raman.py --flavor 3

# 2nd-order BSE (single gamma-point, double+triple)
python resonant_raman.py --flavor 7 \
    --first-order-file ../1st_der_exc_ph/susceptibility_tensors_first_order.h5

# 2nd-order BSE (BZ average over finite-q phonons -- triple only, see the
# Gamma-only caveat above for why double resonance can't be BZ-averaged)
python resonant_raman.py --flavor 5 \
    --q-points-file q_points.dat

# IPA first + second order
python resonant_raman.py --flavor 2

# Polarized (angle-resolved), default scattering-plane normal (z-hat)
python resonant_raman.py --flavor 7 --polarized

# Polarized with a custom normal and finer theta grid, storing full maps
python resonant_raman.py --flavor 3 --polarized --n-hat 0 0 1 \
    --dtheta-deg 1 --polar-store-maps

# Polarized + helicity-resolved (sigma_+/sigma_-), jones convention (default)
python resonant_raman.py --flavor 8 --polarized --helicity

# Same, true-helicity convention relative to the outgoing -n_hat direction
python resonant_raman.py --flavor 8 --polarized --helicity \
    --helicity-convention propagation
```

---

### `plotting/plot_raman_spectra.py`

Plots Raman spectra (Raman shift vs. intensity) at one or more fixed excitation energies. Reads raw susceptibility tensors directly to allow arbitrary phonon broadening, independent of the broadening used in `resonant_raman.py`.

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--Eexc` | One or more excitation energies (eV) to plot |
| `--first-order-file` | 1st-order susceptibility tensor file |
| `--second-order-file` | 2nd-order susceptibility tensor file |
| `--flavor` | Contribution flavor |

```bash
python plotting/plot_raman_spectra.py --Eexc 3.0 3.5 4.0 --flavor 3
```

---

### `plotting/plot_susceptibility_tensors.py`

Plots the raw susceptibility tensor components $\alpha^{\alpha\beta}$ vs. excitation energy for each phonon mode.

- **1st-order**: one figure per phonon mode, 3×3 subplots for each $(\alpha,\beta)$ pair
- **2nd-order**: one figure per (imode, jmode) pair, with titles showing the sum of phonon frequencies

```bash
python plotting/plot_susceptibility_tensors.py --flavor 3
```

---

### `plotting/interactive_vis_resonant_map.py`

Generates a self-contained interactive HTML viewer for resonant Raman maps. Reads `resonant_raman_data_flavor{0..8}.h5` and embeds all data into a single HTML file backed by Plotly.js.

- **Left panel**: 2D Raman map — click anywhere to set the excitation energy (single polarization: the first one checked, see below)
- **Middle/right panels**: Raman spectrum / excitation profile at the selected point
- **Controls**: flavor dropdown, polarization multi-select, excitation energy input, $\omega_{\rm ph}$/$\Omega_{\rm exc}$ axis-range inputs, linear/log toggles

**Multi-polarization comparison**: the polarization control is a multi-select (ctrl/cmd-click for more than one). With exactly one checked, the Spectrum and Excitation Profile panels behave as before. With $N>1$ checked, both panels become an $N$-row grid of subplots (one row per polarization), always sharing the x-axis (row $2..N$'s x-axis `matches` row 1's, via Plotly). The **y-axis** is independent per row by default (each polarization auto-scales to its own data — useful since $\sigma_a\sigma_b$ combinations, in particular, can differ by orders of magnitude) — check **"Share y-axis across rows"** to link them too (Plotly `matches` on the y-axes as well), which is what you want when directly comparing absolute intensities, e.g. checking whether $I_{\sigma_+\sigma_+}$ and $I_{\sigma_-\sigma_-}$ actually coincide for a given flavor/mode (see the Notes below — usually they don't, in this dataset). Overlaid flavors (see below) appear in every row that has data for that polarization. The Map panel intentionally stays single-polarization (showing the first checked one) — a full small-multiples heatmap grid was considered and explicitly deferred, see git history / session notes if reviving it later.

**$\omega_{\rm ph}$/$\Omega_{\rm exc}$ axis ranges**: two pairs of min/max number inputs, defaulting to $[0,\,\max]$ of the initial flavor's axis (set once at page load; clear a field to fall back to that axis's true auto range instead) — zoom every panel that shares that axis: Map, Spectrum, and the phonon DOS/JDOS panels for $\omega_{\rm ph}$; Map, Excitation Profile, and the exciton-DOS/absorption panels for $\Omega_{\rm exc}$. Useful for cropping out an uninteresting low-energy region (e.g. $\Omega_{\rm exc}<1.2$ eV) without needing to re-run the script.

**`--plot-polar-plots`**: additionally reads `resonant_raman_polar_flavor{N}.h5` per flavor (`resonant_raman.py --polarized`, and `--helicity` for the $\sigma_\pm$ data) and (a) merges the four $\sigma_a\sigma_b$ combinations into the *same* polarization multi-select as `xx`/`xy`/etc. — full map/spectrum/profile curves, regridded onto the main map's downsampled axes via nearest-neighbor since the polar file's native excitation axis is coarser — and (b) adds two more panels wired to the *same* map click as the spectrum/excitation-profile panels: a polar panel ($I_\parallel(\theta)$, $I_\perp(\theta)$ at the selected point; needs `--polar-store-maps`) and a helicity panel (bar chart of the four $\sigma_\pm$ combinations + $\rho_{\rm circ}$ at the selected point; needs `--helicity`). No separate "guide map" is embedded for either — the existing map panel already covers the same grid.

Flavors missing the polar file, or missing helicity data within it, show a placeholder (or, for the polarization dropdown, are simply skipped when selected) rather than erroring — this supersedes the retired standalone `interactive_vis_polar_raman.py`, whose guide-map + polar-plot layout is now this flag's polar panel plus the shared map panel.

**IPA DOS**: when `--ipa-elph-file` (the same fine-grid el-ph h5 `susceptibility_tensors_IPA.py` reads, with `Eqp_cond`/`Eqp_val`/`Edft_cond`/`Edft_val` datasets) is found, a second curve is added to the Exciton DOS panel: the non-interacting (IPA) joint DOS of all $(\mathbf{k},c,v)$ transition energies, same broadening/axis as the excitonic DOS, for direct BSE-vs-IPA comparison. Uses the *same* energies (`--ipa-energy-levels gw`, default, matching `susceptibility_tensors_IPA.py --flavor_energy_levels`'s own default) actually driving this project's flavor 0–2 results — pass `dft` to compare against the DFT-level version instead. This is $Q{=}0$ (Γ) only, **not** BZ-$Q$-averaged like the excitonic DOS/JDOS curves above it (those average over 7 phonon-$Q$-shift points; the IPA DOS is inherently single-$Q$, built from the electron/hole $\mathbf{k}$-grid rather than the phonon $Q$-grid) — labeled accordingly in the panel legend.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `.` | Directory containing the HDF5 data files |
| `--output` | `resonant_raman_viewer.html` | Output HTML file |
| `--max-eexc-points` | all | Downsample excitation energy axis |
| `--max-ph-points` | all | Downsample phonon frequency axis |
| `--plot-polar-plots` | off | Also embed $I_\parallel(\theta)$/$I_\perp(\theta)$ and (if present) $\sigma_\pm$ data; adds the $\sigma_a\sigma_b$ polarization options plus the polar + helicity panels |
| `--polar-max-eexc-points` | `40` | Downsample the polar/helicity data's excitation-energy axis (independent of `--max-eexc-points`, which only affects the main map) |
| `--polar-max-ph-points` | `40` | Downsample the polar/helicity data's phonon-frequency axis |
| `--polar-max-theta-points` | `48` | Downsample the $\theta$ axis for $I_\parallel(\theta)$/$I_\perp(\theta)$ |
| `--ipa-elph-file` | `../RESONANT_RAMAN_IPA/elph.h5` | Fine-grid el-ph h5 with QP/DFT band energies, for the IPA DOS curve; skipped with a warning if not found or missing those datasets |
| `--ipa-energy-levels` | `gw` | `gw` or `dft` — which band energies to use for the IPA DOS transition sum |

```bash
python plotting/interactive_vis_resonant_map.py
python plotting/interactive_vis_resonant_map.py --data-dir /path/to/run --output viewer.html

# With polar/helicity panels + sigma_a*sigma_b in the polarization dropdown
python plotting/interactive_vis_resonant_map.py --plot-polar-plots

# Include the polar/helicity panels (needs resonant_raman.py run with
# --polarized --polar-store-maps --helicity for each flavor first)
python plotting/interactive_vis_resonant_map.py --plot-polar-plots
```

---

### `plotting/interactive_vis_resonant_map_2D_materials.py`

Interactive BZ q-contribution map for second-order resonant Raman in 2D materials. For each q-point loads `susceptibility_tensors_second_order_q_{iq}.h5`, computes the phonon-weighted Raman intensity, and renders an interactive HTML showing which q-points in the first BZ dominate the signal at each excitation energy.

Reads direct lattice vectors either from a BerkeleyGW `WFN.h5` or from explicit `--a1`/`--a2` arguments, then constructs the reciprocal lattice and Wigner–Seitz BZ boundary via a Voronoi construction.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--q-points-file` | — | `qx qy qz weight` file in crystal coords (one q per row) |
| `--data-dir` | `.` | Directory with `susceptibility_tensors_second_order_q_{iq}.h5` files |
| `--wfn` | `None` | BerkeleyGW `WFN.h5` — reads direct lattice vectors automatically |
| `--a1` | `None` | In-plane lattice vector a1 in Å (x y), alternative to `--wfn` |
| `--a2` | `None` | In-plane lattice vector a2 in Å (x y), alternative to `--wfn` |
| `--temperature` | `300` | Temperature in K for Bose factors |
| `--output` | `bz_raman_map.html` | Output HTML file |

```bash
# From BGW WFN.h5
python plotting/interactive_vis_resonant_map_2D_materials.py \
    --q-points-file q_points.dat --wfn WFN_fi.h5

# From explicit lattice vectors (graphene, a=2.46 Å)
python plotting/interactive_vis_resonant_map_2D_materials.py \
    --q-points-file q_points.dat \
    --a1 2.46 0.0 --a2 1.23 2.132
```

---

### `plotting/plot_polar_raman.py`

Plots angle-resolved polarized Raman intensities from `resonant_raman.py --polarized`'s output (`resonant_raman_polar_flavor{N}.h5`). Six plot types: per-mode polar plots ($I_\parallel(\theta)$, $I_\perp(\theta)$ overlaid, at a fixed excitation energy — works with the default amplitude-only output, since each mode/mode-pair is an exact line, no broadening needed), a $\theta$–$\omega$ map at fixed excitation energy, a $\theta$–$\Omega$ map at fixed Raman shift (these two need `--polar-store-maps` output), and three helicity-resolved plots reading the `--helicity` output — helicity-resolved spectra ($I_{\sigma_+\sigma_+}$ solid / $I_{\sigma_+\sigma_-}$ dashed vs. Raman shift, stacked by excitation energy, reproducing the layout of Figs. 3/5 of the QERaman paper), excitation profiles ($I_{\sigma_+\sigma_\pm}$ vs. $\Omega$ at each Raman-active mode's shift, one panel per mode), and a $\rho_{\rm circ}(\Omega,\omega)$ map with a diverging colormap centred at zero. Every helicity plot is annotated with the `--helicity-convention` used.

**Output schema of `resonant_raman_polar_flavor{N}.h5`** (written by `resonant_raman.py --polarized`):

| Dataset | Shape | dtype |
|---|---|---|
| `theta` | `(Ntheta,)` | float64 |
| `n_hat`, `e1`, `e2` | `(3,)` | float64 |
| `excitation_energies` | `(Nfreq,)` | float64 |
| `excitation_energies_1st` | `(Nfreq_1st,)` | float64 (only if the flavor has a 1st-order term; can differ in length from `excitation_energies`) |
| `phonon_frequencies_cm` | `(Nmodes,)` | float64 |
| `freq_axis_cm` | `(Nfreq_ph,)` | float64 |
| `M_parallel_first_order`, `M_perp_first_order` | `(Ntheta, Nvalid, Nfreq_1st)` | complex128 |
| `M_parallel_second_order`, `M_perp_second_order` | `(Ntheta, Nmodes, Nmodes, Nfreq)` | complex128 (single-q flavors only) |
| `q_weights` | `(Nq,)` | float64 (`--q-points-file` flavors only) |
| `q_M_parallel_second_order`, `q_M_perp_second_order` | `(Nq, Ntheta, Nmodes, Nmodes, Nfreq)` | complex128 (`--q-points-file` flavors only, replaces the single-q datasets above) |
| `raman_map_parallel`, `raman_map_perpendicular` | `(Ntheta, Nfreq, Nfreq_ph)` | float32 (only with `--polar-store-maps`) |
| `raman_map_sigma_pp`, `raman_map_sigma_pm`, `raman_map_sigma_mp`, `raman_map_sigma_mm` | `(Nfreq, Nfreq_ph)` | float32 (only with `--helicity`; no $\theta$ axis, so small — written unconditionally) |
| `degree_circular_polarization` | `(Nfreq, Nfreq_ph)` | float32 ($\rho_{\rm circ}$ for incident $\sigma_+$; only with `--helicity`) |
| attr `helicity_convention` | — | string, `'jones'` or `'propagation'` (only with `--helicity`) |

Amplitudes, not full maps, are stored by default — at the default grid ($100\times2000\times500$) a full map is ~800 MB per polarization in float64, whereas the amplitudes are small and any map/spectrum/polar slice can be regenerated from them (apply the phonon weight $w_\nu$, square, and broaden with $L(\omega-\omega_\nu)$).

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--polar-file` | `resonant_raman_polar_flavor0.h5` | Output of `resonant_raman.py --polarized` |
| `--plot` | `all` | `polar`, `theta-omega`, `theta-Omega`, `helicity-spectra`, `helicity-profile`, `rho-circ`, or `all` (skips plots whose data isn't present) |
| `--Eexc` | `None` | Excitation energy in eV (`polar`, `theta-omega`) |
| `--Eexc-list` | `None` | One or more excitation energies in eV to stack in `helicity-spectra` (falls back to `--Eexc`) |
| `--raman-shift` | `None` | Raman shift in cm$^{-1}$ (`theta-Omega`) |
| `--mode` | `None` (all valid modes) | 1st-order phonon mode index (also selects the mode(s) plotted by `helicity-profile`) |
| `--mode-pair` | `None` | 2nd-order `(imode, jmode)` pair |
| `--output-prefix` | `raman_polar` | Prefix for output PNG filenames |

```bash
# Per-mode polar plots at 2.2 eV (1st order: all valid modes; 2nd order: modes 0,1)
python plotting/plot_polar_raman.py --polar-file resonant_raman_polar_flavor7.h5 \
    --plot polar --Eexc 2.2 --mode-pair 0 1

# theta-omega and theta-Omega maps (needs --polar-store-maps in the resonant_raman.py run)
python plotting/plot_polar_raman.py --polar-file resonant_raman_polar_flavor7.h5 \
    --plot theta-omega --Eexc 2.2
python plotting/plot_polar_raman.py --polar-file resonant_raman_polar_flavor7.h5 \
    --plot theta-Omega --raman-shift 300

# Helicity-resolved spectra stacked over 3 excitation energies, excitation
# profiles at each mode, and the rho_circ map (needs --helicity in the run)
python plotting/plot_polar_raman.py --polar-file resonant_raman_polar_flavor8.h5 \
    --plot helicity-spectra --Eexc-list 1.8 2.0 2.2
python plotting/plot_polar_raman.py --polar-file resonant_raman_polar_flavor8.h5 \
    --plot helicity-profile
python plotting/plot_polar_raman.py --polar-file resonant_raman_polar_flavor8.h5 \
    --plot rho-circ
```

---

### `susceptibility_tensors_IPA.py`

Computes IPA susceptibility tensors for 1st and/or 2nd order resonant Raman using el-ph coefficients from `elph_interpolated_kgrid.h5` directly (bypassing the BSE exciton-phonon step). QP renormalization of el-ph is applied automatically when `elph_interpolated_kgrid.h5` contains `QP_rescaling_matrix_cond/val` datasets (produced by `interpolate_elph_bgw.py --eqp`).

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--elph_fine_file` | `elph_interpolated_kgrid.h5` | Input from `interpolate_elph_bgw.py` (must include `--eqp` datasets for QP renorm) |
| `--dip_mom_noeh_file_b1/b2/b3` | `eigenvalues_b{1,2,3}_noeh.dat` | Dipole moment files (IPA, no electron-hole interaction) |
| `--dE` | `0.001` | Excitation energy grid step (eV) |
| `--gamma` | `0.01` | Broadening (eV) |
| `--no_renorm_elph` | off | Skip QP renormalization of el-ph coefficients |
| `--skip_first_order_calculation` | off | Skip first-order susceptibility (saves time when only second-order is needed) |
| `--compute_second_order` | off | Compute and save the second-order susceptibility tensor |
| `--iq` | `0` | q-point index in `elph_interpolated_kgrid.h5` for the second-order calculation |
| `--vectorized_flavor` | `2` | Vectorization level for first order |
| `--vectorized_flavor_second_order` | `1` | Vectorization level for second order |

**Outputs:**
- First order → `susceptibility_tensors_first_order_IPA.h5` (always, unless `--skip_first_order_calculation`)
- Second order → `susceptibility_tensors_second_order_IPA_q_{iq}.h5` (when `--compute_second_order`)

```bash
# First order only (gamma)
python susceptibility_tensors_IPA.py

# Second order at q-point 0 only
python susceptibility_tensors_IPA.py \
    --compute_second_order --skip_first_order_calculation --iq 0
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
- **Contract-then-square ordering (polarized Raman)**: the Cartesian `raman_maps` path squares each $(\alpha,\beta)$ component individually and therefore cannot be used to reconstruct $I_\parallel(\theta)$/$I_\perp(\theta)$ — the polarized path (`polarization.py`) must contract Cartesian indices with the polarization vectors *before* squaring, to preserve the relative phases between tensor components. See "Polarized Raman" under Theory.
- **Degenerate-mode caveat**: per-mode polar plots (`plotting/plot_polar_raman.py --plot polar`) are only physically meaningful when summed over a degenerate phonon subspace — the split between individual degenerate modes depends on an arbitrary gauge choice (which unitary mixing DFPT happened to return) and is not itself observable.
- **`alpha_tensor_d3` naming**: despite the h5-dataset name (unchanged from `susceptibility_tensors_first_order.py`, which this renumbering doesn't touch), `alpha_tensor_d3` (used for flavor 4, "diagonal + off-diagonal exciton-phonon") is the *full* diagonal+off-diagonal exciton sum, not just the off-diagonal remainder — see the "d2 + d3: full ($N_{\rm exc}\times N_{\rm exc}$) sum" comment in `susceptibility_tensors_first_order.py`. Flavor 4, not flavor 3, is the complete first-order tensor; flavor 3 (`alpha_tensor_d2`, "diagonal exciton-phonon only") is the diagonal-only subset. This naming quirk is exactly why the flavor descriptions were renamed away from "d2"/"d3" jargon in the 2026-08-05 renumbering.
- **Optical-propagation caveat (polarized Raman)**: the polarized intensities are a purely geometric projection of the Raman tensor and do not include refraction, birefringence, or anisotropic absorption at the sample surface — see the caveat under "Polarized Raman" in Theory.
- **The Jones-vs-helicity naming trap**: "helicity-conserving" in most 2D-materials Raman literature (including QERaman) means "same Jones vector in the fixed lab frame" — which is actually the *opposite* of true optical helicity ($\mathbf S\cdot\hat k$) for the backscattered photon, since it propagates along $-\hat n$. `--helicity-convention jones` (default) matches the literature's labels; `--helicity-convention propagation` gives true helicity and swaps every $\sigma_+\sigma_+ \leftrightarrow \sigma_+\sigma_-$ label. Always check which convention a paper you're comparing against actually used — see "Helicity-resolved Raman" under Theory.
- **Excitonic (BSE) vs. IPA comparison for $\sigma_\pm$ selection rules**: flavors 3–8 include excitonic effects via BSE; flavors 0–2 (IPA) omit them. Comparing the same $A_1'$/$E'$ helicity selection rule across both is a physically meaningful check, not just numerical — the QERaman paper itself reports disagreement with experiment at 2.33 eV that they attribute partly to a mode-dependent broadening $\gamma$ and partly to the absence of excitonic effects in their IPA-level calculation. The *relative* magnitudes of $I_{\sigma_+\sigma_+}$ vs. $I_{\sigma_+\sigma_-}$ are $\Omega$-dependent and sensitive to $\gamma$ (`--gamma`/`gamma_lor`) in both cases.
