# Gila vs own (closed-form) solver on anisotropic / negative-εzz stacks — results, 2026-09-28/29

Question: at which negative εzz does Gila stop converging, how does the Gila/own-solver error
evolve with εzz, and does the mismatch seen on the biaxial stacks (εxx ≠ εyy) come from the own
solver?

All runs: 3-layer stack (vacuum / 0.5λ material slab / vacuum), dipole [1,1,−1]/√3 at 0.484λ
above the slab (Gila's source cell), box source = Gila's voxel, res 1/32λ, raw comparison (no
calibration). Errors are L2 relative, own vs Gila, on the complex field vector.

## Conclusions

1. **The own solver is not the source of the mismatch.** It passes exact per-(kρ,φ) symmetry
   tests for biaxial media to machine precision, its mode wavenumbers match the analytic ones, and
   its field is converged in all three quadrature knobs on a negative-εzz stack (tables below).
2. **Gila has no symmetry/axis bug either** (real-space reciprocity, mirror and 90° rotation with
   εxx↔εyy all hold to solver tolerance), and **its 1/32 discretization is not the issue**
   (1/32 → 1/64 moves its field by 1–5 %, positive and negative εzz alike).
3. **The dominant cause is the lateral truncation of Gila's slab.** Gila's material ends at the
   grid edge (±0.75λ on the 1.5λ stacks); the own solver's is infinite. Guided / hyperbolic modes
   reflect or scatter there. The error grows towards the slab edges, grows with the in-plane index
   (more confined modes) and is 2.5× larger at 1.5λ than at 3.0λ for the same material. A graded
   lateral absorber in Gila's material (§ Absorber) removes most of it.
4. **It is not specific to εxx ≠ εyy.** Uniaxial 4.6/4.6/+2 (R8) is worse than biaxial
   2.25/4.6/+2 (R3) without the absorber, and after it uniaxial and biaxial negative-εzz stacks
   sit at the same level.
5. **Gila convergence vs εzz.** Iterations roughly double per −0.25 in εzz on the biaxial series.
   Threshold at 1.5λ: ≈ −1.5 without the absorber, between −2.0 and −3.2 with it (the edge
   reflections were part of the convergence problem). On a **3.0λ** domain Gila fails already at
   εzz = −1 (R2 biaxial stalls; R9 uniaxial *diverges* after ~1000 iterations, even with the
   absorber) — in-plane loss is only 0.01–0.08 on these stacks, vs 0.1 on the gyrotropic stack,
   which does converge at 3.0λ with εzz = −1.
6. **Error vs εzz, with the absorber:** no growth with |εzz|. Slab error is flat at ≈ 0.19–0.22
   (interior) from εzz = −0.25 to −3.2, vs ≈ 0.10 for positive εzz; close to the source
   (ρ < 0.25λ) it is 0.11–0.15, i.e. close to the positive stacks. The remaining gap depends on the
   absorber settings, so it is attributed to the small (1.0λ) interior of the 1.5λ domain, not to
   either solver. The definitive 3.0λ + absorber check (R9) could not be done: Gila diverges there.

**Follow-up 2026-09-29/30 — proportionate losses at 3.0λ (R10–R12).** Uniaxial, every component
with Im ε ≈ 0.1·|Re ε| (ε_t = 2.25+0.225i), to test whether Gila's failure depends on where
χzz = εzz − 1 sits (R10: εzz = −1, the flat-interface plasmon condition; R12: −9.3, far from it
and from ENZ). R12: bicgstabl **diverges** like R9 (residual 2.0e4 → min 5.3e2 at ~1000 its →
1.55e4 at the cap, 1h43). R11 (−3.2+0.32i) showed the same turn-around (3.0e2 at 800 its →
1.6e3 at 2400) and was stopped after 9h40 (the machine was swapping); R10 was not run. So
neither proportionate loss nor χ far from −1 helps: the breakdown is tied to the large lateral
domain with a negative-εzz slab, not to specific χ values, and is a Krylov breakdown rather than
slow convergence.

Open: a Gila solve that converges on a 3.0λ negative-εzz stack (e.g. restarted GMRES instead of
bicgstabl, or slightly more in-plane loss) to confirm point 6 on a large interior.

## The εzz sweep (biaxial, εxx = 2.25+0.01i, εyy = 4.6+0.08i, εzz + 0.3i, 1.5λ)

![sweep](figures/summary_biaxial_ezz_sweep_res1-32.png)

"Interior" = outside the 0.25λ absorber band (same pixels with and without the absorber);
"near" = ρ < 0.25λ from the source axis. Absorber: `absrel=0.25,1.0`.

| tag | εzz | Gila (no abs → abs), iterations | slab interior, no abs → abs | bottom interior, no abs → abs | slab near / bottom near, abs |
|---|---|---|---|---|---|
| R3_biaxial_ezz+2p0 | +2.0 | ✓ 36 → ✓ | 0.301 → **0.111** | 0.441 → 0.223 | 0.090 / 0.195 |
| R5_biaxial_ezz-0p25 | −0.25 | ✓ 515 → ✓ 175 | 0.281 → 0.187 | 0.467 → 0.373 | 0.145 / 0.316 |
| R6_biaxial_ezz-0p5 | −0.5 | ✓ 851 → ✓ 413 | 0.276 → 0.206 | 0.409 → 0.333 | 0.147 / 0.258 |
| R7_biaxial_ezz-0p75 | −0.75 | ✓ 1610 → ✓ 662 | 0.283 → 0.214 | 0.346 → 0.274 | 0.133 / 0.176 |
| R06_biaxial_ezz-1p0 | −1.0 | ✓ ~2900 → ✓ 819 | 0.287 → 0.216 | 0.315 → 0.240 | 0.109 / 0.130 |
| R07_biaxial_ezz-1p5 | −1.5 | ✗ cap → **✓ 1666** | 0.280 → 0.208 | 0.287 → 0.204 | 0.140 / 0.115 |
| R08_biaxial_ezz-2p0 | −2.0 | ✗ cap → **✓ 3645** | 0.613 → 0.206 | 0.396 → 0.190 | 0.152 / 0.120 |
| R05_biaxial_ezz-3p2 | −3.2 | ✗ cap → ✗ cap (1e‑3) | 0.643 → 0.219 | 0.413 → 0.181 | 0.151 / 0.127 |

Cap = 20 000 mat-vecs (~5000 iterations), reltol 1e‑5.

## Controls and baselines

| tag | ε (xx, yy, zz) | lateral | Gila | slab full / interior / near, no abs | slab interior / near, abs |
|---|---|---|---|---|---|
| R02_isotropic | 2.25 | 3.0λ | ✓ | 0.170 / 0.136 / 0.064 | — |
| R03_pos_uniaxial | 2.25, 2.25, 4.3 | 3.0λ | ✓ | 0.187 / 0.151 / **0.057** | — |
| R01_gyrotropic | [2.1+0.1i ±3.14i; −1+0.3i] | 3.0λ | ✓ | 0.283 / 0.124 / 0.072 | — |
| R04_neg_uniaxial_ezz-5p2 | 2.25, 2.25, −5.2 | 3.0λ | ✗ | 1.00 (Gila unusable) | — |
| R1_pos_uniaxial | 2.25, 2.25, 4.3 | 1.5λ | ✓ | 0.250 / 0.206 / 0.140 | 0.093 / **0.072** |
| R8_uniaxial_4p6_ezz+2p0 | 4.6, 4.6, 2 | 1.5λ | ✓ | 0.501 / 0.411 / 0.384 | 0.124 / 0.081 |
| R4_uniaxial_ezz-1p0 | 2.25, 2.25, −1 | 1.5λ | ✓ 102 → ✓ 55 | 0.362 / 0.227 / 0.147 | 0.219 / 0.145 |
| R2_biaxial_ezz-1p0 | 2.25, 4.6, −1 | 3.0λ | ✗ stalls (abs too) | 1.00 | 0.78 (unusable) |
| R9_uniaxial_ezz-1p0 | 2.25, 2.25, −1 | 3.0λ | ✗ diverges (abs) | — | 0.98 (unusable) |
| R12_uniaxial_lossy_ezz-9p3 | 2.25+0.225i ×2, −9.3+0.91i | 3.0λ | ✗ diverges | 1.00 (unusable) | — |

R1 vs R03 (same material): shrinking 3.0λ → 1.5λ alone multiplies the near-source slab error by
2.5; the absorber brings it back to the 3.0λ level (0.072 vs 0.057).

## Tests on the own solver (`diagnostics/spectral_tests.jl`, `kz_check.jl`, `own_krho_checks.jl`)

Per-(kρ,φ) spectral Green's tensor, 6 kρ ∈ [0.4, 12]ω × 3 φ, εzz ∈ {+2, −1, −2, −3.2}:

| test | worst violation |
|---|---|
| reciprocity top ↔ top (reflection) | 7e‑15 |
| reciprocity top ↔ bottom (transmission, 4-layer stack) | 2e‑13 |
| mirror x → −x | 2e‑13 |
| 90° rotation with εxx ↔ εyy | 2e‑13 |
| eigen() path (Biaxial, εxx = εyy) vs closed-form Uniaxial | 8e‑14 |
| mode kz² vs analytic (s: εyy, p: εxx, εzz for kt ∥ x) | 1e‑15 |

Quadrature convergence on R4 (uniaxial εzz = −1), change vs defaults (top / slab / bottom):
`kρ_cap` 3000 → 6000: 0.070 / 0.0000 / 0.0000 (top = the known unconverged source row, Newell
subtraction); `rtol` 1e‑3 → 1e‑4: 0 / 0 / 0; `N_phi` 256 → 512: 0 / 0 / 0.

## Tests on Gila (`diagnostics/gila_symmetry_tests.jl`, `gila_refine.jl`, `gila_refine_abs.jl`, `xyswap.jl`)

Real-space full solves at reltol 1e‑9 on R3's material (2.25, 4.6, 2): reciprocity top ↔ top
7e‑9, top ↔ bottom 7e‑8; x-mirror 7e‑9 (slab 2e‑7); 90° rotation with εxx ↔ εyy 6e‑9 (slab 1e‑7).
Swapping Gila's x and y against the own field raises the error from 0.44 to 1.1 — axes agree.

Refinement, 1/32 vs 1/64 (same physical source = 2×2×2 fine cells, averaged back), slab / bottom:

| stack | Gila 1/32 vs 1/64 | own vs Gila 1/32 → 1/64 (slab) |
|---|---|---|
| R1 (no abs) | 0.012 / 0.011 | 0.250 → 0.249 |
| R3 (no abs) | 0.027 / 0.026 | 0.374 → 0.374 |
| R8 (no abs) | 0.053 / 0.038 | 0.501 → 0.502 |
| R3 (abs, interior) | 0.016 / 0.024 | 0.111 → 0.107 |
| R4 (abs, interior) | 0.041 / 0.032 | 0.219 → 0.239 |

(R06 at 1/64 with the absorber was stopped after 1h40 without converging.)

## Absorber

`GilaRun.add_edge_absorber!` (mod_GilaRun.jl), opt-in via `ensure_gila_field(...; edge_absorber)`
or on the command line (`abs=w,m[,p]` / `absrel=w,r[,p]` after the stack name). Only material
voxels (χ ≠ 0) within `w` of a lateral grid edge; ramp `((w − d)/w)^p`, `d` = distance to the
nearest lateral edge; only Im of the diagonal changes. `abs`: `+ i·m·ramp` on every diagonal
entry; `absrel`: `+ i·r·|Re ε_jj|·ramp` per entry. Cached as `gila_<spec_key>_abs<hash>.jld2`;
figures tagged `_abs-w0p25-m3p0` / `_absrel-w0p25-r1p0` (`-p<n>` only if p ≠ 2), cyan dotted lines
at the inner edge, and an extra "lateral interior only" stats table in the log.

Strength scan (near-source slab / bottom error, w = 0.25λ, p = 2), `logs/absorber_scan*.log`,
`logs/absorber_sensitivity.log`: the best absolute strength scales with Re ε (R1: m ≈ 1, R8:
m ≈ 3–10), hence `absrel`; r = 1.0 has the lowest worst case over R1/R3/R8 (r = 0.7 is best for
R1 alone). Too strong or too wide an absorber reflects and raises the error again (R1 at m = 10;
R4 at r = 2 or w = 0.35λ). R4 prefers a gentler one (r = 0.4 or p = 3).

## Files

- Stacks: `mod_StackCompare.jl` (R01–R08 = older stacks, R1–R9 = diagnostic series, `_diagnostic_stack`).
- Figures: `figures/<kind>_<tag>_res1-32[_absrel-…|_abs-…].png`, 8 kinds per run, ε tensor and
  geometry in the info strip; `figures/summary_biaxial_ezz_sweep_res1-32.png`.
- Consolidated metrics (recomputed from the caches): `diagnostics/summarize_results.jl` →
  `diagnostics/results_summary.csv`.
- Logs: `logs/` (per-run comparison logs, `absrel_*`, scans, refinements, own-solver checks).
