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
