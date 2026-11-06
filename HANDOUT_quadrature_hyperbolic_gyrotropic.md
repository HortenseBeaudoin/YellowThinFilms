# Handout: k-space quadrature for low-loss hyperbolic and gyrotropic stacks

2026-11-05. This handout collects what went wrong with the default k-grids on the Funky figures (2026-11-04), the ZrS₂ figure (2026-11-03) and `Additional_tests/` (2026-11-05), what replaced them, and what that suggests for the core. It builds on `HANDOFF_tail_stall_keep_segments.md` (same folder), which fixed one symptom: the tail fallback. Here the target is the cause.

**Summary**
- The default grids (`point_kx_grid`, `box_kρ_grid`, both built on `adaptive_kx_grid`) only know where the **light lines** are. They have no idea where the **polariton poles** are.
- For a dielectric stack that is fine: the light lines are the only real-axis features, and they all sit below `kmax`.
- For hyperbolic, metallic or gyrotropic layers the important structure is a set of narrow poles. They sit far beyond the largest light line, have a width proportional to the loss, and in gyrotropic or in-plane anisotropic stacks also move with φ.
- What worked every time was a hand-placed, fixed Gauss–Legendre rule with three ingredients:
  1. breakpoints graded onto every light line;
  2. a uniform fine band covering all the poles, with the node spacing set by the pole width;
  3. a cut-off set by the geometry.

  Each rule was then checked by halving every panel width.
- The proposals at the end turn that recipe into code.

---

## 1. What the default does

| | `point_kx_grid` (1D) | `box_kρ_grid` (2D) |
|---|---|---|
| Use | point source, azimuthally symmetric stack only | box source, any stack |
| Engine | `adaptive_kx_grid`: per-probe h-adaptive Gauss–Kronrod (QuadGK), with the panels of all probes merged | same, run on each φ group (`phi_groups`) |
| Finite band | `[0, kmax]`, with breakpoints at `branch_points(stack, ω)` | same |
| `kmax` | largest `ω·Re√ε_ii` over the layers with **Re ε_ii > 0** | same |
| Tail | `(kmax, ∞)` via `kρ = kmax + t/(1−t)`, capped at `tail_bound` = `target_decay/h` (25 e-folds), at least 80·kmax, at most `kρ_cap` = 2000 | same |
| Tolerances | `rtol` = 1e-6, `maxevals` = 1e4, order 7 | `rtol` = 1e-4, `maxevals` = 1e4, `N_phi` = 64, `on_stall` = `:error` |

`branch_points` reads the diagonal of ε only, and does not depend on φ. Nothing in this chain *finds* a pole: the docstring of `adaptive_kx_grid` says so explicitly. A pole gets resolved only if Gauss–Kronrod refinement stumbles onto it, and only once loss has moved it off the real axis.

## 2. Why it fails on these materials

### 2a. The poles land in the "tail"

In a hyperbolic or metallic layer, the axis with Re ε < 0 contributes no branch point. `kmax` is therefore set by air or the substrate, around k₀ to 1.5 k₀. The polaritons sit far above it:
- **ZrS₂ on Au, 250 cm⁻¹:** poles at 9–186 × `kmax`.
- **hBN film, 1500 cm⁻¹:** q ≈ 7 k₀.
- **hBN in the hyperlens:** volume modes up to tens of k₀.

The tail substitution is designed for a small, smooth, decaying correction. Here it receives the main part of the integral, compressed in t near 1. Per probe, it then either:
- **stalls at `maxevals`.** Hyperbolic ZrS₂ had 26 of 100 probes stalled at 1e4, and 4 of 100 still stalled at 1e6. Under the old `:fallback` behaviour the field was 23% off whatever `maxevals` was; under `:keep` it is 1e-3, but at 13.7k nodes;
- **or "converges" on a panel whose Kronrod nodes happen to miss a narrow peak.** The code's own comment on `_kronrod_reference_rule` describes this; I did not observe it directly.

### 2b. Pole width scales with the loss, and nothing sizes the panels to it

The half-width of a pole is Im q ∝ γ. The relevant materials are exactly the low-loss ones, which is why the Funky figures used γ = 1 cm⁻¹ for hBN and γ = 1e-3 ω_p for InSb.

| Case | Pole position | Im q (half-width) |
|---|---|---|
| hBN 100 nm on BaF₂, 1500 cm⁻¹, γ = 1 | 6.55 µm⁻¹ | 0.04 µm⁻¹ |
| same, 812 cm⁻¹ (backward wave) | −6.40 µm⁻¹ | 0.14 µm⁻¹ |
| ZrS₂ slab, 250 cm⁻¹ | 0.005–0.1 nm⁻¹ | ~1e-4 nm⁻¹ |
| InSb magnetoplasmon, Ω = 1.09, γ = 1e-3 ω_p | 1.76 k₀ (+ grazing −1.005 k₀) | ≈ 3e-2 k₀ |
| α-MoO₃ bilayer, 930 cm⁻¹, γ = 2–4 | up to tens of µm⁻¹ | ≳ 0.3 µm⁻¹ |

With no loss at all, the poles sit on the integration path. `box_kρ_grid` detects this and warns, suggesting Im ε ≈ 1e-6, which only works for isotropic and uniaxial stacks.

### 2c. Hyperbolic media do not have a "small tail"

For a hyperbolic half-space, r_p(k → ∞) tends to a complex constant: Im r_p ≈ 0.53 for hBN at 1500 cm⁻¹, whatever γ is (Purcell figure). The integrand then decays only through the geometric factor e^{−k·h}, so:
- the cut-off must come from the geometry (`target_decay/h`), never from the light lines;
- the region between the last pole and the cut-off must be sampled densely enough to follow e^{−kh}, which is very different from a thin tail.

The same holds for thick hyperbolic slabs, whose dense ladder of volume modes behaves as a quasi-continuum.

### 2d. Light-line singularities are under-resolved

Every light line, at k = Re√(ε_ii μ_jj)·k₀ for each layer, carries a square-root branch point, and 1/k_z is integrably singular there. Gauss–Legendre panels that merely end at that point converge slowly. In `Test_Mu.jl` the dipole above a magnetic half-space was off by 1e-5 to 1e-2. Grading both quadratures geometrically onto k₀, the substrate's √(εμ)k₀ and the surface-plasmon pole, down to 2⁻⁴⁰, brought it to ≤ 5e-7.

Two related traps:
- **Grazing modes hug the light line:** the InSb grazing wave sits at −1.005 k₀.
- **r_pp normalisation is singular at |k| = ω exactly** (the grazing air mode): `Fig_InSb_Dispersion.jl` keeps its k grid off that point.

### 2e. Gyrotropic and in-plane-anisotropic stacks: the poles move with φ, and there is no point-source path

- The 1D cache refuses such stacks, and `point_kx_grid` is 1D only. The only adaptive 2D builder, `box_kρ_grid`, is designed around box sources and their probes.
- Pole positions depend on φ, and with gyrotropy they are not even symmetric: k(φ) ≠ k(φ + π). The InSb magnetoplasmon exists along +x only.
- `phi_groups` accounts for this by giving every φ its own adaptive run (one group per φ once off-diagonal terms exist), which is costly.
- `branch_points` ignores the off-diagonal terms entirely.
- The 2D cache takes a **tensor-product** grid (one kρ rule shared by all φ). The kρ rule therefore has to resolve the union of every φ's poles anyway.

### 2f. The φ rule is a trapezoid; `N_phi` = 64 is too small for large maps

The lateral factor e^{i kρ ρ cos(φ − ψ)} contains harmonics up to |m| ≈ kρ·ρ. The uniform trapezoid rule is exact only for |m| < N_φ, so N_φ must exceed kρ_max·ρ_max with margin.

| Case | kρ_max·ρ_max | N_φ | Error |
|---|---|---|---|
| InSb, ±300 µm map | ≈ 360 | 160 / 320 / 640 vs 1280 | 50% / 8% / 0.2% |
| MoO₃, ±2 µm map | ≈ 170 | 400 → 800 | 9.3e-7 |

### 2g. Source plane: the direct term never decays

On the source's own plane (Δz = 0), the free-space part of the integrand does not decay in k, and no finite cut resolves it. The way out is `terms = :scattered` plus the analytic free-space dipole field, which is what the InSb and MoO₃ figures did. Near-source rows that the cut cannot resolve are blanked: the hyperlens blanks rows within 12 nm of the source.

## 3. What worked

Every rule below is a fixed, piecewise-uniform Gauss–Legendre rule (order 8 unless stated), built by `gl_panels(a, b, width)` in the scripts. Each was checked by halving every panel width, and in 2D by doubling N_φ.

| Figure | Breakpoints and panel widths | Cut | Nodes | Self-check |
|---|---|---|---|---|
| `Fig_hBN_BackwardWave.jl`, `Fig_hBN_SpinLocked.jl` (1D, γ = 1) | [0, 2ω] ω/40 · **[2ω, 8] 2e-3** (poles) · [8, 40] 2e-2 · [40, 120] 0.1 µm⁻¹ | 120 µm⁻¹ (e^{−120·0.15} ≈ 1e-8) | ≈ 45k | halving: ≤ 3e-6 |
| `Fig_hBN_Hyperlens.jl` (1D, γ = 1) | [0, 2ω] ω/40 · **[2ω, 60] 5e-3** · [60, 150] 2e-2 · [150, 400] 0.1 | 400 µm⁻¹ (rows < 12 nm from the source blanked) | ≈ 150k | halving: 2.5e-3 / 1.5e-3 |
| `Fig_hBN_Purcell.jl` (Sommerfeld side) | s < 1 in s_z, s > 1 in v = √(s² − 1): **both remove the 1/k_z singularity**; log-graded v panels; breakpoint at √ε_z; graded to 1e-9 around the surface-plasmon pole | 40/(2k₀d_min) | — | vs closed-form G: ≤ 1.2e-3 |
| `Fig_InSb_OneWay.jl` (2D, γ = 1e-3 ω_p) | graded onto k₀ from below and above (2⁻¹²) · **[k₀, 2.5k₀] k₀/400** (covers the pole at every φ) · [2.5k₀, 6k₀] k₀/100 · to 1.2 µm⁻¹ k₀/10; **N_φ = 640** | e^{−2kh} ≈ e^{−12} | 6.0M (20 s per cache) | pole panels halved: 3.4e-3 |
| `Fig_MoO3_Twisted.jl` (2D) | graded onto k₀ and √ε_BaF₂·k₀ (2⁻¹⁰) · 0.05 µm⁻¹ to 60; **N_φ = 400** | e^{−2·60·h} = e^{−12} | — | halving: 1.5e-13; N_φ 400 → 800: 9.3e-7 |
| `Fig_ZrS2_Dipole.jl` (1D, nm) | [0, 2ω√ε_top] ω/10 · **[·, 0.2] 5e-4** · [0.2, 1] 5e-3 · [1, 2] 5e-2 nm⁻¹ | 2 nm⁻¹ | ≈ 5.2k | ~1e-3 vs a 49k-node reference (adaptive with `:keep`: 1e-3 at 13.7k nodes) |
| `Additional_tests/Test_Mu.jl` [2] | graded onto k₀, each √(ε_t μ_z)k₀ and √(μ_t ε_z)k₀, and each surface pole (2⁻⁴⁰), then geometric to the cut; order 16 | 50/(2d) | — | vs Sommerfeld: ≤ 5e-7 |

**The recipe, distilled:**
1. **Light lines:** add a breakpoint at every Re√(ε_ii μ_jj)·k₀ of every layer, and grade geometrically onto it from both sides (2⁻¹⁰ to 2⁻⁴⁰), or change variables as in the Purcell figure.
2. **Poles:** locate them first, from Im r_pp peaks along the real axis. The poles in these figures were found by hand, either way.
   - Cover the whole pole region (in 2D, the union over φ) with uniform fine panels.
   - The node spacing that converged ranged from ≈ 0.01 × Im q (hBN, InSb) to ≈ 0.6 × Im q (ZrS₂, 8 GL nodes on a 5e-4 panel against a 1e-4 width). A spacing ≲ Im q/2 is a reasonable default.
3. **Coarsening:** beyond the poles, widen the panels in steps (×10 per decade of k worked), still following e^{−kh}.
4. **Cut-off:** set it from the geometry, k_cut ≈ 18–25/h_min with h_min the smallest source-to-observation vertical distance (2h for the scattered field on the source plane). Never set it from the light lines.
5. **2D only:** use N_φ ≳ 1.5–2 × kρ_cut·ρ_max.
6. **Source plane:** `terms = :scattered` plus the analytic direct field.
7. **Always self-check:** halve every panel width, double N_φ, and report the relative change.

**Cost.** The rules are not cheap: 45k–150k nodes in 1D and 6M in 2D. But they are predictable, and halving them is the convergence proof. The adaptive grid is cheaper when it works, and on these stacks it either stalls or silently misses structure.

## 4. Proposals for the core

These are ordered by value for effort. None is implemented yet.

**P1. Cut-off from geometry, poles inside the finite band.** For any stack with a layer where Re ε_ii < 0 (or simply always), set the finite band to `[0, tail_bound(...)]`. The geometric bound already exists. Then drop the t-substituted tail, or keep it only for what lies beyond the geometric bound. This alone moves the poles out of the "tail" pathology of 2a and 2c.

**P2. A real-axis pole scan: `real_axis_poles(stack, ω, φ; kmax)`.**
1. Evaluate log|Im r_pp|, or the stack's cavity denominator, on a log-spaced kρ grid up to `kmax`. It is cheap: one 4×4 S-matrix per point, as in `Fig_InSb_Dispersion.jl`.
2. Pick the peaks and estimate each width from the half-maximum (≈ Im q).
3. Return `(q, Γ)`.
4. The grid builder then inserts a fixed fine band over `[q_min − nΓ, q_max + nΓ]` with node spacing ≤ Γ/2.

A hyperbolic slab gives a ladder of poles, and a thick slab or half-space gives a quasi-continuum. In that case, cover the whole hyperbolic band uniformly rather than pole by pole. In 2D, scan a handful of φ and take the union, because the cache's kρ rule is shared across φ.

**P3. Complete `branch_points`.**
- Include μ: Re√(ε_ii μ_jj).
- Handle off-diagonal ε and μ, using the eigenvalues of the in-plane part, or simply the diagonal of a few rotated frames.
- Always grade geometrically onto each branch point.
- Optional: substitute k = k₀√(1 − s_z²) and k = k₀√(1 + v²) around air's light line, as in the Purcell figure, to remove the 1/k_z singularity entirely.

**P4. A 2D point-source builder: `point_kρ_grid_2d`.**
- The kρ rule comes from P1–P3, with the union of poles over φ.
- N_φ is set from the geometry: `N_φ = ceil(c · kρ_cut · ρ_max)` with c ≈ 1.5–2, rounded to an even number. An even N keeps the φ + π pairs, which the exact reciprocity checks rely on.
- There is no φ grouping, so the tensor-product cache is used as is.

This fills the gap where gyrotropic and in-plane-anisotropic point sources currently have no default at all.

**P5. Built-in self-check.** Add an option such as `check = true` that rebuilds with halved panels (and 2N_φ in 2D) at a few probe points and reports the relative change in `info`. Cheap reciprocity checks are a second guard, since they are exact on any even-φ grid: G(Δ) = G(−Δ)ᵀ for reciprocal stacks, and Onsager–Casimir with B → −B for gyrotropic ones. A failure there signals a physics error, not a quadrature error.

**P6. Lossless or very low loss: contour deformation (larger change).**
- Integrating on a path pushed off the real axis would make the cost independent of γ. The Berreman and S-matrix layers already accept complex k.
- What would have to change: the caches' Float64 kρ nodes, the lateral kernels (Bessel and exponentials with complex kρ), and the handling of the branch cuts.
- The cheaper alternative, adding artificial loss (`box_kρ_grid` suggests Im ε ≈ 1e-6), changes the propagation length, which is often the very quantity being studied. If used, it must be reported as a model change.

**Keep, but don't rely on:** adaptive Gauss–Kronrod is still useful for dielectric stacks and as a cheap first pass. With P1–P3 seeding its breakpoints, it would no longer have to find poles by brute force.

## 5. Open points

- All of this concerns **point sources**. Box sources (the `box_kρ_grid` probes, the interior slab) were not revisited.
- The pole-spacing rule (≲ Im q/2) is an envelope of the cases above, not a derived bound.
- The γ → 0 limit was not pushed: the lowest losses used were γ = 1 cm⁻¹ for hBN and 1e-3 ω_p for InSb.
- The hand-tuned panel widths (2e-3, 5e-3, k₀/400, …) were chosen per figure from the pole widths in section 2b. P2 is what would make that automatic.
