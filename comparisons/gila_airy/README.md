# Plane-wave reflectance: own S-matrix vs Gila vs Airy/TMM

This comparison follows Gila's tutorial
[07_thin_film](https://moleskysean.github.io/GilaElectromagnetics.jl/stable/tutorials/07_thin_film/).
Gila solves one Bloch-periodic 2×2-cell column of an infinite stack. We compare its |r|² for s
and p against two other computations:

- **own**: S₂₁ of the whole stack from `ModeSolver` + `ScatteringMatrix` (Redheffer). We
  evaluate it at kρ = 2π sinθ and project onto a unit plane wave.
- **TMM**: an independent characteristic-matrix (Airy) reference in the e^{−iωt} convention, so
  loss is Im ε > 0. On the single film it reproduces the tutorial's closed Airy formula to 1e-15.

```bash
julia -t auto --project=comparisons/gila_airy comparisons/gila_airy/airy_comparison.jl
```

The folder has its own environment because `blc` / `setBlc!` exist only in the moleskysean fork
of Gila (1.1.0). The main Manifest still pins PaulVirally 1.0.3.

## Results (Δ = λ/64, angles 0–85° in steps of 2.5°, 2026-10-22)

| case | films (χ, d/λ), bottom → top | max \|own − TMM\| | max \|Gila − TMM\| | R(0°) TMM / own / Gila |
|---|---|---|---|---|
| `film` | (1.25, 1/2) | 3e-15 | 3.3e-3 (s, 80°), 6.1e-3 (p, 85°) | 0.14793 / 0.14793 / 0.14826 |
| `stack` | (1.25, 1/4), (3.0, 3/16), (1+0.1i, 1/8) | 1e-14 | 8.4e-4 (s, 37.5°), 6.3e-4 (p, 0°) | 0.07097 / 0.07097 / 0.07160 |

- The `film` case reproduces the tutorial's numbers (0.14826 vs 0.14793, worst gap ≈ 0.006 at
  grazing incidence).
- Our S-matrix core is exact to machine precision on both cases. Every remaining gap is Gila's
  λ/64 discretization.
- The time convention matters on the lossy stack. With the e^{+iωt} (Macleod) matrices the TMM
  turns the lossy layer into gain, and it was off by 0.11 against both solvers.

Raw values: `results/<case>.tsv`. Figures: `results/<case>.png`.

## Plane-wave field maps above the stack (`plane_wave_slice.jl`)

```bash
julia -t auto --project=comparisons/gila_airy comparisons/gila_airy/plane_wave_slice.jl <stack> <incidence> [cells per λ]
```

A plane wave with a fixed k_t and a random complex s/p polarization falls from above on a
three-film stack. The script compares the field in the top vacuum layer (0 < z < λ/2) on an x/z
slice. In the figures z is vertical and the slice sits above the drawn stack.

- **own:** a `ModalSeed` at the top edge of layer 1 (src = obs = layer 1) and
  `Modal_amp_profile` give the amplitudes at each depth, and `_mode_field_core` gives (Ex, Ey, Ez).
- **Gila:** the tutorial's Bloch 2×2 column (λ/64 by default) in Float64, with Bloch phases
  (−kx·2Δ, −ky·2Δ). It is extended along x with the same phase.

Both solve the same laterally infinite, periodic problem. Gila's tutorial setup with a point
source would instead give a periodic array of sources, which is not our single-source field.

The main metric is the **scattered** field E − E_inc, the stack's response. For evanescent
incidence the incident wave grows with z and dominates the total field.

Stacks, films listed bottom → top as (εt, εz, d):

- `base`: (2.25, 1/4λ), (4.0, 3/16λ), (2.0+0.1i, 1/8λ), isotropic, the stack of `airy_comparison.jl`.
- `hyperbolic`: the same, with the middle film uniaxial, εt = 4.0+0.1i, εz = −3.2+0.3i.

Incidences, with k_t along φ = 31°:

- `oblique`: |k_t| = 0.583 k0, θ = 35.7°.
- `normal`: k_t = 0.
- `resonance`: the strongest reflection pole between k0 and 3k0, i.e. the largest singular value
  of the physical s/p reflection matrix at the stack face. For `base` that is 1.0624 k0, with
  |r| = 21; there is a second pole at 1.66 k0, with |r| = 12.
- `evanescent`: |k_t| = 2.5 k0, which propagates inside the hyperbolic layer.

The reflection matrix matches the isotropic TMM of `airy_comparison.jl` exactly on `base`
(|r_ss|², |r_pp|² at 0°, 35.7° and 60°; cross terms 1e-16).

| stack | incidence | ‖r_sp‖ | total, own vs Gila | scattered, own vs Gila |
|---|---|---|---|---|
| base | oblique | 0.39 | 1.9e-3 | 6.5e-3 |
| base | normal | 0.26 | 2.1e-3 | 8.5e-3 |
| base | resonance (λ/64) | 21 | 1.9e-1 | 2.0e-1 |
| base | resonance (λ/128) | 21 | 4.9e-2 | 4.9e-2 |
| base | resonance (λ/256) | 21 | 1.2e-2 | 1.2e-2 |
| hyperbolic | oblique | 0.37 | 1.9e-3 | 7.3e-3 |
| hyperbolic | normal | 0.24 | 2.0e-3 | 8.4e-3 |
| hyperbolic | evanescent | 0.42 | 3.5e-6 | 1.5e-2 |

- **Away from a pole**, every case agrees to about 1%. That includes normal incidence and the
  hyperbolic layer, for propagating and for high-k_t evanescent waves.
- **At the resonance**, the gap falls by 4× per halving of Gila's cell, 0.20 → 0.049 → 0.012.
  So Gila converges as Δ² onto our field. A sharp pole amplifies the small shift that
  discretization causes, and our solver has no such error.
- Each Gila solve takes about 3 s.

Figures and logs: `results/plane_wave_<stack>_<incidence>[_<cells per λ>].{png,log}`.

## Box source above the stack, Gila with a lateral absorber (`box_source_slice.jl`)

One λ/32 box source with a random complex moment sits 0.234λ above the three-film stack.
Gila uses a finite 64×64 grid in Float64, with the absorber of `RESULTS_gila_anisotropic_comparison.md`
(0.25λ, rel_imag 1.0, power 2) in the films. Our code uses the closed-form box, src = obs = layer 1.
The comparison is the top vacuum layer on the x/z plane through the source.

| region | relative L2, own vs Gila |
|---|---|
| off the source row | 2.6e-2 |
| off the source row, away from the source (interior) | 1.8e-2 |
| source row (source voxel excluded) | 3.0e-2, but 20–30% beyond 0.25λ |

**The "0.35 factor" (investigated 2026-10-22).** It was not a normalization difference:

- An off-by-one in the first version of this script mirrored our rows about the middle of the
  layer, which put our source one row off Gila's.
- The free-space G₀ of the fork and of the old PaulVirally Gila agree to 1e-8 on a voxel source.
  Against an analytic point dipole, our field off the source row fits with a factor of 0.998
  (residual 9e-5), and Gila's with 0.989∠0.5°.

**The source row.** Our field there is the open item of STATUS (no quasi-static subtraction).
Only the in-plane longitudinal component is off: Ex from p_x on this x slice, by 16–30% out to
1λ. Gila is within 1–6% of the dipole on that row. N_phi_near must be 8192 here, because 2048
aliases the source row beyond ~0.5λ. So compare off the source row until that item is done.

**Bloch boundaries instead of the absorber** (`box_source_slice.jl bloch`, 2026-10-22). Gila runs
with Bloch-periodic lateral boundaries (period = the grid, phase 0) and no absorber. The period is
60 cells = 1.875λ: at 64 cells (2λ) the (0, ±2) order lies exactly on the light circle, a
Rayleigh–Wood anomaly, and Gila stops. Off the source row, by lateral distance from the source:

| distance | absorber (64 cells) | Bloch, phase 0 (60 cells) |
|---|---|---|
| 0–0.12λ | 2.6e-2 | 5.0e-2 |
| 0.16–0.31λ | 1.1e-2 | 0.65 |
| 0.34–0.50λ | 1.5e-2 | 0.79 |
| 0.53–0.75λ | 3.3e-2 | 1.0 |
| 0.78–1.00λ | 8.4e-2 (absorber band) | 0.94 |

With Bloch boundaries Gila solves a periodic array of sources over an infinite stack. Its images
are one period away and the χ = 3 film guides their light back, so beyond ~0.15λ the field is a
different one (fringes with nodes near x ≈ ±0.6λ). A like-for-like comparison would need our field
summed over the same reciprocal lattice, k = k_Bloch + 2π(m, n)/L. `CavityTransferCache2D` only
takes a (kρ, φ) product grid, so that needs a node-list variant of the cache.
Figures: `results/box_source_slice_stack_{absorber,bloch}.png`.
