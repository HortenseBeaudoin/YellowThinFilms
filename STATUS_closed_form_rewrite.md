# Status: closed-form rewrite (`PLAN_closed_form_point_source_rewrite.md`)

Last updated 2026-10-16 (step 3, z skeletons, implemented, validated and benchmarked on branch `z-skeletons`: see "Branch `z-skeletons`"). Phases 0–8 of the plan are implemented and validated, and the legacy implementation has been removed from the project (see "Cleanup"). The comparison against GilaElectromagnetics has been ported to the closed-form box and run (see "Gila comparison"), and followed by the anisotropic study of 2026-09-28/30 (see "Changes since 2026-09-24"). Next: move the project into a git checkout of the GitHub repository (see "Repository state"), then the quasi-static (Newell) subtraction (§4.2) and the speed-up of the 2D box materialization.

## Repository state (2026-10-12)

GitHub: <https://github.com/HortenseBeaudoin/YellowThinFilms> (public, branch `main`). It has a single commit, `b370e8b` "Add files via upload" (2026-09-24, a web upload). The working folder `G_z-z_Box-Source/` is **not** a git repository. Plan: clone the repository into a new folder, copy the working files into it, commit, and work on branches from then on.

Folder vs GitHub, checked file by file on 2026-10-12:

- **Identical (15 files):** `ClosedForm.jl`, `Project.toml`, `Manifest.toml`, `Small_example`, and the modules `mod_{CavityTransfer,ClosedFormAssembly,ClosedFormField,ClosedFormGrid,Dipz_SemiAnalytical,FieldSolver,KxGrid,ScatteringMatrix,SourceProj,Sources,UniformSource}.jl`.
- **Different (1 file):** `mod_ModeSolver.jl`. The folder holds the 2026-10-09 `conj` fix (see below), and GitHub still has the old version. Commit it on its own so it is traceable.
- **Only in the folder, to commit:**
  - Validation suite: `Gz_ClosedForm_{Foundation,Point,Box,Grid}_Validation.jl`, `Gz_ClosedForm_TestHelpers.jl`.
  - Gila comparison: `compare_gila_own_solver.jl`, `mod_StackCompare.jl`, `mod_OwnSolverRun.jl`, `mod_GilaRun.jl`, and `diagnostics/` (scripts, `results_summary.csv`).
  - Notes: `PLAN_closed_form_point_source_rewrite.md`, this file, `RESULTS_gila_anisotropic_comparison.md`.
- **Only in the folder, keep out of git (`.gitignore`):**
  - `field_cache/` (1.2 GB) and `greens_cache/` (348 MB). They can be regenerated, but slowly: about 21 min (own solver) + 5 min (Gila) per stack, and hours when Gila does not converge. Copy them by hand to the new folder if replots are wanted.
  - `figures/` (58 MB), `logs/`, `ShapeCompare_figs/` (archive), `Gz_ClosedForm_SourcePlane.png`, `.DS_Store`.

## Branch `small-speedups` (2026-10-12)

The project now lives in a clone of the GitHub repository (this folder). The validation suite was copied in from `G_z-z_Box-Source/`. The `conj` fix in `mod_ModeSolver.jl` was already on GitHub (the file is identical), so the "Different (1 file)" note below is out of date.

- **Wraparound fix (assembly).** The FFT apply was a circular convolution of `G` sampled on offsets `[−L, L)`. Any source off the grid centre picked up `G` wrapped by 2L near the opposite edge (O(1) errors there; e.g. a source at x = 0.4 with L = 1 was wrong for x < −0.6). `Ĝ` is now built on the 2× zero-padded offset grid (`padded_offsets`, `2Nx×2Ny`), and the apply is an exact linear convolution. `ClosedFormGreen` keeps only `Ĝ` (with `Δx·Δy/(Px·Py)` folded in), not `G`.
- **Apply = FFT · multiply · iFFT only.** `closed_form_field` FFTs each source plane once, accumulates `Ĝ·Ĵ` over all source depths in k-space, and inverse-transforms each observation plane once. FFTs are in-place, threaded and MEASURE-planned. `precompute_ghats!(ghat_cache, caches, grid, z_src_by_layer, z_obs_by_layer; kw...)` builds every `Ĝ` up front, so the apply does no grid work.
- **2D `closed_form_G`.** One fused pass per row, threaded over φ: the observation phase is computed once per (node, mode), and the direct term and local Ez are folded in. The amplitude loops are threaded too. One FINUFFT plan (`setpts` once) is reused by every row. Output is bit-identical to before.
- **1D `closed_form_G`.** The 3×3 response is computed once per distinct ρ, then rotated per pixel (`G = Rot·R(ρ)·Rotᵀ`). The Bessel table is `Nk × Nρ`. Agrees with `closed_form_pixel` to 1e-14.
- **Timings (8 threads).**
  - One source depth × 128 obs rows (box 1/32, 3-layer stack, 48×48 grid): 118 s → 7.1 s (iso, 300k nodes) and 34 s → 2.4 s (biaxial, 84k nodes), now on the 4×-larger padded grid. The apply on 128 rows takes 0.016 s.
  - 1D path, 96×96 × 32 rows: 15.2 s → 0.9 s.
  - Suite: Box 10 → 3 min; Point and Grid ~1.4 min each.
- **`nufft_eps` now defaults to 1e-9** (was 1e-12). It is ~25% faster on the precompute and changes `Ĝ` by ~5e-10, against a quadrature `rtol` of 1e-3. The tests that check the NUFFT against the exact sum (Point [c], Assembly [a]) pin 1e-12.
- **`Gz_ShapeCompare_Validation.jl` rewritten without the legacy path.** It compares point and box on the closed-form code only, with the same stacks, dipoles, 64 px × 60-row volume and scenarios as before. Each scenario writes 3 figures to `ShapeCompare_figs/` (48 PNGs, git-ignored):
  - maps: point | box | relative intensity difference;
  - an |Ez| slice at x = 2;
  - |Ez|² 2 px beside the source.

  Results:
  - Box vs point at |Δz| ≥ 0.5 from the source plane: 6–11% in vacuum, up to 21% in vac-glass-vac. This is expected for a 0.3λ box (the lateral sinc at k₀ is ≈ 0.86 per axis).
  - The point source shows its own-plane stripe, and the box doesn't.
  - Timings: build 1.4–2.2 s (point) / 9–14 s (box); precompute 0.17–0.8 s (point) / 2–3.6 s (box) per source depth for all 60 rows; apply 1 ms.
- **New:** `Gz_ClosedForm_Assembly_Validation.jl`. It checks the padded apply against per-dipole `closed_form_G` at shifted offsets, for off-centre dipoles near the edges at two depths in all layers (1D: 2e-16; 2D box NUFFT: 2e-13), and checks that precompute-then-apply builds nothing.
- `UniformSource._forward_transform`/`_inverse_transform`/`_centering_ramp` are no longer used by the assembly.
- **Next (planned).** A z-structured operator per (source layer, obs layers):
  - same layer: Toeplitz (direct, any material) + Hankel (scattered, iso/uniaxial; verified to 1e-15, broken by biaxial mode mixing at 3%);
  - cross-layer and biaxial scattered: low-rank skeletons in z (measured ranks ~10 of 32 at 1e-6, except on a hyperbolic layer's own side), with dense per-pair as the fallback; *(2026-10-16: how this was measured is not recorded; per pixel frequency the rank is much higher. See "Step 3 pre-study and plan".)*
  - apply via z-FFT / BLAS, accumulated in k-space.

## Branch `z-structured-operator` (2026-10-12, from `small-speedups`)

- **`closed_form_G(...; terms=...)`** keeps one class of terms: `:all` (default, unchanged), `:direct`, `:scattered`, `:toeplitz` (direct + matched-branch reflections f→f, b→b), `:hankel` (cross-branch reflections). It is split at the node level through `amplitude_at(...; seeds=:f|:b)`.
  - `:toeplitz + :hankel = :all` and `:direct + :scattered = :all` hold on any medium.
  - On an isotropic source layer, or a uniaxial one in an azimuthally symmetric stack, `:toeplitz` depends on `z' − zpos` only and `:hankel` on `z' + zpos` only (1e-15).
  - Biaxial: 0.8% / 39% violation, as expected (mode mixing between different `Q`).
- **`mod_GreenOperator.jl`:** `build_layer_operator(cache, grid, z_src, z_obs_by_layer; structure=:auto, kw...)` → `LayerGreenOperator`; `apply_operator(op, P)` with `P` = moments per pixel and source plane (`dipole_moments(op, dips)` builds it); `operator_summary(op)`.
  - Uniform z grids per layer, given by the user. Source planes are reordered internally to increasing cavity-frame `zpos` (layer 1 flips).
  - Blocks:
    - same layer, structured: Toeplitz + Hankel, `Ns+No−1` planes each, FFT'd along z (`Lz ≥ Ns+No−1`). The Hankel part uses the plane-reversed source spectrum, with its output shift folded into the kernel.
    - same layer, biaxial/gyrotropic: Toeplitz direct + dense scattered.
    - cross layer: dense per pair.
  - `:auto` decides from the media, then checks the 4-point identity on a few pixels and falls back to dense with a warning.
  - Apply: 2D FFT of every source plane; z-FFT, pointwise 3×3 and z-iFFT for the Toeplitz/Hankel blocks; plane-pair products for dense blocks; one 2D iFFT per observation plane. All accumulation is in k-space.
- **`Gz_ClosedForm_Operator_Validation.jl`** (~45 s): term split [t]; operator = `closed_form_field` for random moments, all layers, point/box, source layers 1/2, structured / toeplitz_dense / dense [o]: ≤ 1.3e-15; cost [m]. *(Since then: grouped grid [g] and z skeletons [s], see "Branch `z-skeletons`".)*
- **Cost.** Gila-like stack (vac 1.5 / glass 0.5 / vac 2.0), box 1/32 in layer 1, 48×48 px (padded 96×96), 300k nodes:
  - layer 1 → 1 (48×48 planes): structured, 243 MB, 5.6 s (per-depth route ≈ 2.9 GB, ≈ 75 s);
  - layer 1 → 2 (48×16): dense, 972 MB, 21 s;
  - apply, full layer-1 source volume (110k voxels) → layers 1 + 2: 0.08 s;
  - per-depth route: 101 s, 3.9 GB.
- **`box_kρ_grid` memory (2026-10-13).** It kept a full one-kρ cache per visited kρ (≈ 0.45 MB at 256 φ without the rotation shortcut, 10⁴–10⁵ kρ), which ran out of 16 GB on biaxial stacks. It now memoizes the probe values only (probes threaded per kρ). Bit-identical grids; 4.8 → 1.15 GB on the biaxial check.
- **Benchmark (2026-10-13, `Bench_GreenOperator.jl`, informal, 4 threads).** Box source in layer 2 (0.5 thick), outer layers 0.5, all layers observed; Δ = voxel side, 2λ × 2λ window. Max-precision quadrature: `box_kρ_grid` rtol 1e-3, `kρ_cap` 3000, `ρ_far` 1, 256 φ, `nufft_eps` 1e-12; cache + operator built in chunks of 64 φ and summed (exact). Rows in `logs/bench_green_operator.csv`.

  | stack | S: 16²×4 (Δ=1/8) | M: 32²×8 | L: 64²×16 (Δ=1/32) | apply, full volume S / M / L |
  |---|---|---|---|---|
  | iso vac/glass/vac | 26 s, 7 MB | 74 s, 89 MB | 261 s, 1.3 GB | 0.8 / 6.6 / 90 ms |
  | uni vac/uni/glass/vac | 40 s, 8 MB | 187 s, 107 MB | 339 s, 1.6 GB | 1.2 / 7.5 / 122 ms |
  | hyperbolic biaxial, εzz = −3.2+0.01i | 74 s, 8 MB | 153 s, 116 MB | 445 s, 1.8 GB | 0.8 / 6.4 / 129 ms |

  - Precompute = grid + cache + operator build. At L, 85–90% is the operator build, ≈ 100–140 s per dense / toeplitz_dense block vs 30 s for the structured block.
  - Apply cost does not depend on the number of non-zero sources (1, one per plane, or every voxel: same time).
  - εzz = −1 is a bad hyperbolic test against vacuum (ε + 1 = 0, plasmon condition): the far-lateral probe's tail stalls. At −3.2 everything converges.
  - Biaxial εzz = +2+0.3i (Gila R3) is not in the table (it stalled at M); fixed since, see "`box_kρ_grid` far-lateral probe fix" below. The grids of this table also carried that bug (extra kρ, not a wrong field).
  - **Threads** (iso, L): `-t 4 / 8 / 10` → precompute 262 / 266 / 267 s, apply 98 / 62 / 60 ms. `-t auto` gives 4 on the M5 (4 performance + 6 efficiency cores). The build does not scale with `-t`: ≈ 94% of each `closed_form_G` call is `finufft_exec!`, which runs on FINUFFT's own OpenMP threads (all cores by default). FINUFFT options don't help on this transform (585k nodes → 128², 9 per row, 55 ms): `nthreads=4` 110 ms, `nthreads=10` 62 ms, `upsampfac=1.25` 100 ms, eps 1e-9 51 ms. The grid gets slower with more threads (22 → 30 s). Use `-t 8` for applies; the build's levers are fewer transforms (step 3, or batching a source plane's rows into one FINUFFT call) or fewer nodes.
- **Low losses instead of lossless (2026-10-13, `diagnostics/loss_sweep.jl`).** Same δ = Im ε on every component of every layer; box 1/8, benchmark knobs.
  - Iso and uniaxial stacks: `box_kρ_grid` converges down to δ = 1e-6 with no stalls, ≈ 15% more kρ than at 1e-2 and the same time; the narrowest panel scales with δ (the near-real poles are resolved). The field converges linearly to the lossless limit: change per decade of δ 3e-6 (vertical neighbour) to 3e-5 (ρ = 1 in the slab) at δ = 1e-6, so ≈ 3e-5 from the limit. The error grows with lateral distance (guided modes decay laterally at a rate set by δ), so wider windows want smaller δ. At δ = 0 every probe's finite band stalls (half the probes) and the field is wrong at O(1).
  - Hyperbolic layer (εzz = −3.2): low losses do not work. 13.9k kρ / 83 s at δ = 1e-2, 89k kρ / 528 s at 1e-3 (field still moving 20–30% from 1e-2), stalls from 1e-4 (15/44, then 21/44 at 1e-5). The loss is what makes its high-kρ integrand decay.
- **Lossless stacks raise (2026-10-13, branch `loss-biaxial-fixes`).** `box_kρ_grid` warns up front when no layer has any loss (`is_lossless(stack)`), and a stalled finite band now throws, listing the stalled probes (`on_stall=:error`, the default; `on_stall=:warn` returns the unconverged grid as before, used by `diagnostics/loss_sweep.jl`). In `:error` mode `adaptive_kx_grid`'s per-probe stall warnings are silenced (`warn_stall=false`). A stalled tail still falls back to the fixed evanescent band, as documented. Vac/glass/vac at δ = 0 raises (16/16 probes); at δ = 1e-6 it converges (1,335 kρ, 5 s).
- **`box_kρ_grid` far-lateral probe fix (2026-10-13).** The "far lateral" probe sat at `z_obs = z̄ + Δz/2 + δ` with no check against the layer: for the near-bottom-face depth (`z̄ = d − Δz/2 − 1e-3·d`, a default `zbar`) it was outside the source layer (0.562 in a 0.5 layer at Δ = 1/8), where the layer's synthesis grows like `e^{+kρ·overshoot}` (|f(kρ = 300)| ≈ 1e10 vs 1e-10 in the layer). That one probe drove the node count: alone, 14.3k kρ for biax εzz = +2 at 64 φ (vs 765 in the layer), 2.5× extra for iso. Biaxial mode mixing amplified it, hence the growth with N_phi seen only for biaxial. The probe now goes on whichever side of the box stays in the layer (skipped if neither does).
  - Biax εzz = +2+0.3i, ρ_far = 1, Δ = 1/8: 7k / 19k / 65k → 6.0k / 6.1k / 5.7k kρ at 8 / 64 / 256 φ (24 s at 256), same as iso (6.0k / 5.6k / 5.4k). Δ = 1/16, 256 φ: 5.9k kρ in 30 s (stalled before).
  - Accuracy (biax, 64 φ, Δ = 1/8): G on the rtol 1e-3 grid vs the rtol 1e-4 grid, ≤ 6e-7 at the self voxel, vertical neighbour, ρ = 0.5 and 1 in the slab and ρ = 0.5 in the layer below.
  - Grid, Box and Operator validation suites pass.
- **Loss benchmark (2026-10-13, in progress).** `Bench_GreenOperator.jl` now runs every stack (iso, uni, biax εzz = +2, hyp εzz = −3.2) at δ = Im ε ∈ {1e-1, 1e-3, 1e-6} on every component of every layer, S/M/L. It skips (and records) a case whose grid stalls or needs > 40k kρ, and carries the skip to larger sizes, as for a stalled tail. `BENCH_RESUME=1` resumes from `logs/bench_green_operator_loss.csv`. S and M are done; L is still to run.
  - Iso/uni: low loss costs little (5.2k → 6.4k kρ from 1e-1 to 1e-6).
  - Biax εzz = +2: 5.2k / 12.9k / 56.6k kρ at 1e-1 / 1e-3 / 1e-6 (over budget). Its node count also grows with low loss: the φ-dependent guided-mode poles of any biaxial layer, not only hyperbolic ones.
  - Hyp: 5.6k at 1e-1; 89k (over budget) at 1e-3; at 1e-6 the finite bands converge but the tail stalls (32.8k kρ, fixed-band fallback, 2.4 h at S).
  - Biaxial hyperbolic N_phi scan (Δ = 1/8, `ρ_far` = 1): uniaxial hyperbolic is flat in N_phi (8.3k at 1e-2, 17.5–18.7k at 1e-3 for 8–128 φ); biaxial hyperbolic grows with it at 1e-3 (35k / 51k / 76k at 8 / 32 / 128 φ). Fixed by the φ-dependent kρ grid below; these S/M rows used the shared grid and are kept in `logs/bench_green_operator_loss_product.csv`.
  - The first M rows (iso/uni/biax/hyp at 1e-1, iso at 1e-3) ran in the process that had just done the 2.4 h hyp case and look inflated (iso M 204–230 s vs 47 s in a fresh process): rerun before quoting them.
- **φ-dependent kρ grid (2026-10-14).** In a biaxial layer the guided-mode poles, and a hyperbolic layer's high-kρ Fabry–Pérot comb, sit at φ-dependent kρ; the shared kρ grid resolves every φ's poles once they are narrower than their shift between φ nodes, hence the growth with N_phi. One φ needs far fewer: biax-hyp δ = 1e-3, 18–24k (vs 89k shared at 256 φ); biax εzz = +2, 1.0–6.7k at 1e-3 (vs 12.9k) and 1.8–7.4k at 1e-6 (vs 56.6k).
  - `ClosedFormGrid.box_kρ_grid_grouped` splits the uniform φ grid into `phi_groups` (one group for an azimuthally symmetric stack, then identical to `box_kρ_grid`; mirror classes {φ, −φ, π−φ, π+φ} for diagonal ε/μ, merged `classes_per_group` at a time; single φ otherwise) and runs `box_kρ_grid` per group, in parallel, with the lateral probes along the group's φ (`lateral_dir`; with x̂ probes a φ ≈ π/2 group never resolves its `e^{i ky y}` oscillation: 26% error). `GreenOperator.build_layer_operator_grouped` builds a cache + operator per group (split by a node budget) and sums them: exact, G is linear in the quadrature sum (split vs product 1e-15). The cache, field and operator code are unchanged.
  - Accuracy (biax εzz = +2, δ = 1e-3, 64 φ, vs a shared rtol 1e-5 grid): the per-group grid does not over-resolve like the shared one (union of every probe's panels: shared rtol 1e-3 lands at 4e-8), so it runs at rtol 1e-4: source layer 2.3e-6 (1 class/group) / 3e-7 (4), other layers ≤ 7e-8, with 0.54–0.64× the shared rtol-1e-3 grid's nodes.
  - Group size (256 φ, S, δ = 1e-3, rtol 1e-4): biax 6.1k / 6.4k / 8.1k mean kρ at 1 / 4 / 16 classes per group, grid ≈ 220 / 64 / 58 s; hyp 21.8k / 38.2k / 69.3k, grid 803 / 551 / 571 s. The grid cost is per-solve overhead (≈ 1.3 ms per kρ solve for a 4-φ group: ~20 small `closed_form_G` calls), so small groups are slow to build; 4 classes per group is the benchmark default.
  - S, 1 class per group (sequential groups): biax 1e-3 6.1k kρ, 302 s precompute (shared: 12.9k, 132 s: the grid dominates at S); hyp 1e-3 21.8k kρ, 1,092 s (was skipped at 89k); biax 1e-6 6.9k kρ, 349 s (was skipped at 56.6k). One mirror group of hyp at 1e-6 converges with 55.7k kρ in 43 s (the shared grid's tail stalled after 1 h), still over the 40k benchmark budget.
  - `CavityTransfer`'s cache loops use the default (dynamic) `@threads` instead of `:static` (no thread-local buffers), so groups can run in parallel.
  - New `[g]` test in `Gz_ClosedForm_Operator_Validation.jl`; all seven validation suites pass.
- **Loss benchmark, φ-dependent grid (2026-10-14, `logs/bench_green_operator_loss.csv` + `logs/bench_hyp_1e-3.csv`).** Same δ on every component of every layer, 256 φ, kρ_cap 3000, nufft_eps 1e-12; shared grid at rtol 1e-3 for iso/uni, grouped grid at rtol 1e-4 (4 classes per group) for biax/hyp; kρ = mean per φ. S = 16²×4, M = 32²×8, L = 64²×16 voxels.
  - Precompute at L (s): iso 146 / 156 / 172, uni 183 / 197 / 221, biax 315 / 351 / 481, hyp 322 / 2,089 / – at δ = 1e-1 / 1e-3 / 1e-6. kρ at L: iso 5.9–7.0k, uni 6.0–7.3k, biax 6.6 / 7.6 / 10.6k, hyp 6.7 / 46.4k. Memory at L 1.3 (iso) – 1.8 GB.
  - Apply (1 source / one per plane / every voxel): ≈ 0.7–1.3 ms at S, 5–9 ms at M, 75–108 ms at L, for every stack and loss.
  - Build is 70–90% of the precompute at L; the grouped grid's own cost is 70–120 s at L for biax (vs 15–20 s for the shared iso/uni grid), 573 s for hyp at 1e-3.
  - Hyp at δ = 1e-3 needs 38k / 41k / 46k kρ at S / M / L (run with `BENCH_KMAX=60000`; over the default 40k from M).
  - Not converged: hyp at δ = 1e-6. With 4 classes per group 16/17 tails stall (fixed-band fallback, 1,633 s grid at S); one class per group converges (55.7k kρ, see above) but was not benchmarked.
- **Next:** low-rank skeletons in z for the cross-layer blocks and the biaxial scattered block (step 3), planned in "Step 3 pre-study and plan" below. The cross-layer dense block now dominates memory and build time; layer 1 → 3 (48×64) would be ≈ 4 GB at this size.

## Step 3 pre-study and plan (2026-10-16)

Goal: replace the `Ns·No` plane pairs of the non-Toeplitz blocks (`:dense` cross layer, the scattered `D` of `:toeplitz_dense`), the main memory and build cost (≈ 100–140 s and 600 MB per block at L), by a factorisation in z. The apply should stay dominated by the lateral FFTs.

### Why the z-FFT covers only the same layer

The z-FFT needs translation invariance in z (Toeplitz: depends on `z' − z` only) or its mirror (Hankel, `z' + z`, image sources). Both hold only when source and observation sit in the same homogeneous medium: `k_t` is conserved at every interface, `k_z` is not. The direct term is Toeplitz in any medium; the reflections are Toeplitz + Hankel only without mode mixing (isotropic, or uniaxial in an azimuthally symmetric stack). Cross-layer and biaxial-reflected blocks have no such structure. What they do have is separability: per quadrature node, `Ψ_n(z', z) = Φ_n(z')ᵀ·C_n·Ψ_n(z)`, i.e. everything passes through the source layer's edge amplitudes `[sf∘e^{iQf(d−z)}; sb∘e^{−iQb z}]` (the input of `T^o` in `amplitude_at`). The z-rank per node is ≤ 4, and ≤ 2 into a semi-infinite cladding, which carries outgoing waves only (checked: σ₅/σ₁ ≈ 1e-14 on a 1- and 2-node cache).

### Measured: per pixel frequency the rank is not 4 (`diagnostics/zrank_study.jl`)

Box source in layer 2 of a 3-layer stack, cubic voxel Δ = 1/32, 16 planes per layer, 32² px, max-precision quadrature (`kρ_cap` 3000, 256 φ, `nufft_eps` 1e-12). Per lateral pixel frequency `k_m`, `D(k_m)` as a (3·No) × (3·Ns) matrix; ranks relative to the block's largest σ (the 3 components are counted, so full rank is 48).

| block | rank per k_m @1e-6, max (mean) | best rank-4 error (Frobenius) | basis shared by all k_m @1e-6, obs / source side |
|---|---|---|---|
| iso δ = 1e-3, 2 → 1 / 2 → 3 (cross) | 16 (12.3) | 4.2e-2 | 31 / 33 |
| iso, 2 → 2 scattered | 21 (17.0) | 1.7e-1 | 42 / 42 |
| biax εzz = +2, δ = 1e-3, cross | 17 (12.8) | 4.3e-2 | 30 / 36 |
| biax, 2 → 2 scattered | 23 (18.0) | 1.8e-1 | 42 / 42 |
| hyp εzz = −3.2, δ = 1e-1, cross | 25 (18.7) | 1.6e-2 | 30 / **48** |
| hyp, 2 → 2 scattered | **48** (48) | 1.6e-1 | **48 / 48** |

- An edge state of 4 amplitudes per `k_m` is **not** exact: 4e-2 relative error on cross blocks. The rank-4 residual is spread evenly over all planes, not concentrated next to the interface.
- **Cause: the finite lateral window, not aliasing.** At Δ = 1/8: capping kρ below the pixel band (`kρ_cap` = 20 < π/Δ, no node can alias) leaves the rank unchanged; loss (δ = 1e-3 → 1) barely changes it; a wider window helps (e₄ 4.4e-2 → 1.3e-2 from L = 0.5 to 2 at δ = 1e-1). `Ĝ(k_m)` is G truncated to the padded offsets, i.e. its spectrum smeared by a Dirichlet kernel ≈ 2π/(4L) wide (≈ k₀/2 at L = 0.5), which mixes many `k_t` with very different `Q`. That is what makes the operator exact on the window, so it cannot be avoided per `k_m`.
- Compressing per `k_m` (SVD of each `D(k_m)`) would save ≈ 1.5× at 16 planes, at 1e-6, and would still need the full `D` to build. Not worth it.

### The 4(Ns + No) idea, and why it is not the plan

Idea (from the step 3 discussion): on a cross block only the combined plane-wave amplitudes at the source layer's edges matter, so store 4 amplitudes per source plane and per observation plane, `4(Ns + No)` instead of `Ns·No` pairs. It is exact **per quadrature node** (one conserved `k_t`, so one `k_z` per layer: σ₅/σ₁ ≈ 1e-14, above), and fails **per pixel frequency** (rank-4 error 4e-2, table above): `Ĝ(k_m)` is a bundle of nodes ≈ 2π/(4L) wide in `k_t`, each with its own `Q`, and 4 amplitudes cannot describe a bundle of plane waves with different `k_z`. The only exact form keeps the nodes in the apply: NUFFT each source plane from the pixels to the ~10⁶ nodes, contract with the edge amplitudes, expand onto the observation planes, NUFFT back. That is ≈ 10⁶ / (Px·Py) ≈ 250× more points per plane than the lateral FFTs at every apply, against the priority of an FFT-dominated apply, so it is not used. The skeleton below keeps the per-node factorisation and drops the requirement that only 4 numbers survive the binning: `R_s·R_o` pairs instead of `4(Ns + No)`.

### What works: a node-independent skeleton per layer (`diagnostics/zfamily_rank.jl`)

Expand the z-dependence **per node, before the lateral binning**, on a basis shared by every node: `e^{iQ(k_t)z'} ≈ Σ_p U_p(z')·α_p(k_t)`. `U` is not touched by the binning (window, aliases), so the error is only the truncation of the exponential family, set by a tolerance τ. The family is known analytically per layer (its `Q`s and planes), with no stack solve. Its numerical rank grows only logarithmically with the number of planes N (0.5-thick layer, kρ ∈ [0, 3000]):

| family, rank @1e-6 (@1e-9) | N = 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| vacuum cladding, outgoing waves only | 8 (8) | 11 (14) | 12 (17) | 14 (19) | 16 (22) |
| glass layer, both directions | 8 (8) | 14 (16) | 17 (23) | 21 (28) | 24 (33) |
| hyperbolic layer (εzz = −3.2), both directions | full | full | full | full | full |

The shared-basis columns of the table above agree: 31/48 ≈ 3 × 10.3 on the vacuum side, consistent with 11 at N = 16. Expected saving on a cross block, `(R_o·R_s)/(No·Ns)` in memory and build: ≈ 0.6 at 16 planes (1e-6), ≈ 0.07 at 64 planes (≈ 14×), ≈ 0.025 at 128. Step 3 pays off for many planes per layer (thick layers, fine Δ), little at the benchmark's 16. A hyperbolic layer's own side cannot be compressed (high-kρ waves propagate there): keep it dense on that side.

At 16 planes the dense pair-loop apply (`9·Ns·No` per `k_m`) costs about as much as the lateral FFTs, so the apply is not the issue at that size; memory and build are.

### Plan (new branch from `main`, new session)

1. **Skeleton bases per layer and plane grid (interpolative decomposition).** For each layer ℓ in the operator, the family matrix `F_ℓ[j, n]`:
   - observation side: `e^{iKZf·z'_j}`, `e^{iKZb·(z'_j−d)}` for every node and mode of `cache.basis[ℓ]`; outgoing branch only in a semi-infinite layer;
   - source side: `e^{iQf(d−z_i)}`, `e^{−iQb z_i}` from `cache.Qf_src`/`Qb_src`; the box factor `D` and the seeds are per-node constants and don't enter.

   Column-pivoted QR, or an ID on rows, at tolerance τ (default 1e-9, well below the quadrature `rtol`), gives skeleton planes `𝒮_ℓ` (R_ℓ of them) and an interpolation matrix `P_ℓ` (N × R_ℓ) with `F_ℓ ≈ P_ℓ·F_ℓ[𝒮_ℓ, :]` for **every** node. All entries are bounded by 1 (each exponential decays into the layer), so τ is a relative bound per node term. `F` has ~10⁶ columns: sample them, e.g. all kρ at a few φ, or a randomized ID, then check on all nodes.
2. **New block kind `:skeleton`** (cross layer) and `:toeplitz_skeleton` (biaxial same layer: Toeplitz direct + skeleton scattered). Store `D_sk = Ĝ(k_m; 𝒮_o, 𝒮_s)` as `(Px, Py, 3, 3, R_s, R_o)`, plus `P_o` (No × R_o) and `P_s` (Ns × R_s).
   - **Build** with the existing `_dense_planes` on the skeleton pairs only: no new kernel code, R_s·R_o `closed_form_G` rows instead of Ns·No.
   - **Exactness:** source side `Σ_i G(z', z_i)·J_i = Σ_q G(z', z_{s_q})·(P_sᵀ J)_q`; observation side `G(z'_j, ·) = Σ_p P_o[j, p]·G(z'_{o_p}, ·)`. Both hold to τ per node, so also after binning.
3. **Apply:**
   1. `Ĵ_sk = P_sᵀ·Ĵ2` along z: one GEMM, the same matrix for every `k_m`;
   2. the pair loop on `R_s × R_o`;
   3. `Ê = P_o·Ê_sk` along z: one GEMM;
   4. the existing 2D iFFT.

   Batch the small products over all `Px·Py` frequencies.
4. **Automatic fallback:** if `R_ℓ ≥ 0.8·N` on a side (hyperbolic layer), use identity on that side (all planes are skeletons). It falls back to `:dense` when both sides are full. Iso/uniaxial same-layer blocks keep Toeplitz + Hankel.
5. **Validation:**
   - `Gz_ClosedForm_Operator_Validation.jl`: skeleton vs dense for random moments, all layers, point/box, source layers 1/2, iso/biax, to ≈ τ × (‖P‖ growth);
   - the existing [o]/[g] tests keep passing;
   - a hyperbolic case checks the fallback.
6. **Benchmark:** `Bench_GreenOperator.jl` S/M/L as before, plus a thick-layer case (64 planes in a cladding or a 2λ source layer), where the ≈ 10× saving is expected. Report memory, build, apply per block, and R per layer.
7. **Open points:**
   - the choice of τ, and whether `P` (an ID) amplifies errors more than an orthogonal `U`: check ‖P‖;
   - sampling the ~10⁶ family columns without missing the evanescent extremes;
   - layer 1's flipped plane order (`flip`).

   All three are settled in "Branch `z-skeletons`" below.

## Branch `z-skeletons` (2026-10-16, from `main`): step 3

Steps 1–6 of the plan are done: the code is in `mod_GreenOperator.jl`, the benchmark is below. Write-up: `docs/step3_z_skeletons.tex`.

- **Skeletons (`_zskeleton`, `z_skeletons`).** Per side of a block, the family of the layer's own exponentials on every quadrature node and mode:
  - observation side (local `z'`): `e^{iKZf·z'}`, `e^{iKZb(z'−d)}`;
  - source side (cavity-frame `zpos`): `e^{iQf(d−zpos)}`, `e^{−iQb·zpos}`;
  - semi-infinite layers carry one branch only, checked on a cache: `T^o` rows/columns exactly 0 for the upward wave in obs layer 1, the downward one in the last layer, and the `Ψb` input of a source in layer 1.
  
  Built as: streaming QR (TSQR, threaded) of all ~10⁶ columns down to an `N × N` factor (exact, no sampling of the evanescent extremes), column-pivoted QR of it, then the smallest rank whose interpolation error, **checked on every node term**, is ≤ τ (bisection). Entries are ≤ 1 in modulus, so τ is an absolute bound per node term. ≈ 2 s for 750k columns at 64 planes; BLAS is pinned to 1 thread inside (nested BLAS threads were ≈ 10× slower).
- **‖P‖ ≈ 1.8 at 16 planes, 3.2 at 64** (biaxial layer, τ = 1e-6): the interpolation does not amplify errors.
- **`flip`:** the source skeleton is built on the internal plane order (`zs_int`, increasing `zpos`), so layer 1 needs nothing special.
- **Blocks.** `:skeleton` (cross layer) and `:toeplitz_skeleton` (biaxial same layer: Toeplitz direct + skeleton scattered). `D` is evaluated with the existing `_dense_planes` on the skeleton pairs only. A side whose skeleton has ≥ 80% of its planes is kept whole (`Ps`/`Po = nothing`); both whole → the plain `:dense` / `:toeplitz_dense` block. Iso/uniaxial same-layer blocks are unchanged (Toeplitz + Hankel). `structure=:dense` disables the skeletons (the reference).
- **Apply.** Fold the source planes (`Ĵ·Ps`, one GEMM per component, shared by every block from the same source skeleton), pair loop on `R_s × R_o`, interpolate (`·Poᵀ`, one GEMM), then the usual 2D iFFT.
- **Grouped build.** The blocks are summed over φ groups, so all groups must share the skeletons: `z_skeletons_grouped` builds them before the first cache, from each layer's own `_modes_layer` solve at every `(kρ, φ)` node of every group (φ = 0 only on an azimuthally symmetric stack). This is not a stack solve.
- **API.** `build_layer_operator(...; skeleton=true, skeleton_tol=1e-6, skeletons=nothing)`, the same keywords on `build_layer_operator_grouped`. Default τ = 1e-6, 100× below the grouped grid's rtol 1e-4; the field error measured is ≈ 0.1–0.7 τ.

### Skeleton sizes with the per-node criterion (`diagnostics/zskeleton_rank.jl`)

Rank at τ = 1e-4 / 1e-6 / 1e-9, without the 80% rule; families from 6k kρ ∈ [0, 3000] (× 32 φ for biaxial), both modes.

| layer | 0.5λ, N = 8 | 16 | 32 | 64 | 128 | 2λ, N = 64 | 2λ, 128 |
|---|---|---|---|---|---|---|---|
| vacuum cladding (outgoing only) | 8/8/8 | 9/12/15 | 11/14/18 | 12/16/21 | 13/17/24 | 14/18/23 | 16/20/26 |
| glass layer (obs = src) | 8/8/8 | 12/15/16 | 15/19/24 | 18/23/30 | 20/27/36 | 21/25/33 | 23/30/38 |
| biaxial εzz = +2 (obs = src) | 8/8/8 | 12/15/16 | 16/19/24 | 18/24/30 | 21/27/37 | 24/27/34 | 26/32/40 |
| hyperbolic εzz = −3.2, δ = 0.1 | full | full | full | full | full | full | full |

- 1–4 planes more than `zfamily_rank.jl` (whose σ/σ₁ criterion depends on the sampling density), still logarithmic in N.
- A 2λ layer costs only 2–3 planes more than a 0.5λ one at the same N: the propagating part (Nyquist count ∝ thickness·n/λ) is small next to the evanescent one.
- The biaxial family is the glass one ±1 plane.
- Expected size of a vacuum ↔ glass cross block at 1e-6, `(R_o·R_s)/(No·Ns)`:
  - 16 planes: 0.75 (the glass side is kept whole);
  - 32: 0.26;
  - 64: 0.09 (11×);
  - 128: 0.028 (36×).

### Validation (`Gz_ClosedForm_Operator_Validation.jl`, all PASS)

- [o], [g] and [m] now run with `skeleton=false`; they test exactness and are unchanged (≤ 1.3e-15, split 6e-16).
- New [s]: skeleton vs dense blocks for random moments on every plane, Δ = 1/32, 0.75λ claddings (24 planes), 0.5λ middle layer, τ = 1e-6:

| case | blocks (R_s × R_o) | skeleton vs dense |
|---|---|---|
| iso box 2D, source layer 2 | 1: 16×14, 2: structured, 3: 16×14 | 7.6e-8 |
| iso box 2D, source layer 1 (flip) | 1: structured, 2: 14×16, 3: 14×14 | 7.7e-8 |
| biaxial box 2D, layer 2 | 1: 16×13, 2: toeplitz_dense (both sides whole), 3: 16×13 | 4.4e-7 |
| hyperbolic box 2D, layer 2 | source side whole on every block (fallback), 2: toeplitz_dense | 3.4e-7 |
| iso point 1D, layer 2 | 1: 16×13, 3: 16×13 | 6.8e-7 |
| grouped biaxial box, 3 groups, shared skeleton | as above | 4.4e-7 |

The other suites do not use `mod_GreenOperator.jl`.

### Benchmark (2026-10-18, `Bench_GreenOperator.jl`, size L, δ = 1e-3, τ = 1e-6, 8 threads)

Box source in layer 2, Δ = 1/32, 64×64 px, 256 φ, max-precision grid as before. The claddings are observed on their first 0.5λ (16 planes). Logs: `logs/bench_skeleton_{on,off}.{out,csv}`.

| stack | block | skeleton R_s × R_o | MB on / off | build s on / off |
|---|---|---|---|---|
| iso vac/glass/vac (Ns = 16) | 2 → 1, 2 → 3 | 16 × 11 (glass side whole) | 396 / 576 | 31 / 52 |
| biax vac/biax/vac (Ns = 16) | 2 → 1, 2 → 3 | 16 × 12 | 432 / 576 | 58 / 75 |
| | 2 → 2 | both sides whole: `toeplitz_dense` | 648 / 648 | 88 / 85 |
| thick vac/glass 2λ/vac (Ns = 64) | 2 → 1, 2 → 3 | **26 × 11** | **644 / 2304** | **70 / 241** |

| stack | precompute s on / off | total MB on / off | apply ms on / off (all voxels) |
|---|---|---|---|
| iso | 92 / 135 | 936 / 1296 | 56 / 49 |
| biax | 318 / 343 | 1512 / 1800 | 84 / 77 |
| thick | **210 / 550** | **1647 / 4968** | **162 / 205** |

- **Many source planes (thick, 64): 2.6× less precompute, 3.0× less memory, 21% faster apply.** The cross blocks are 3.6× smaller (26·11 / 64·16 = 0.28, as predicted) and 3.4× faster to build.
- **At 16 planes (iso, biax): 1.2–1.5× on the cross blocks.** Only the cladding side compresses (11–12 of 16); the glass/biaxial side stays whole. The biaxial same-layer block is unchanged.
- **Apply at 16 planes is 6–8 ms slower (+12%)**: the fold/interpolation GEMMs and their buffers cost more than the 16 × 11 pair loop saves over 16 × 16. Either keep a side whole unless it saves more (e.g. R ≤ 0.6·N instead of 0.8·N), or remove the allocations in the apply. Not done yet.
- Skeleton construction: < 1 s (iso, thick), ≈ 10 s for biax (2 M (kρ, φ) mode solves per layer), inside the build column.
- The grid and the cache are unchanged. Even with skeletons, the operator build still dominates the thick case's precompute (190 s of 210 s; grid 19 s, cache 1.5 s): the same-layer structured block (52 s) is now as costly as each cross block.

## ω/2π normalization fix (2026-10-18, branch `z-skeletons`)

Found by the theory review (handout `../G_z-z_Box-Source/HANDOUT_omega_over_2pi_normalization.md`): the code returned **(ω/2π) × the physical field**. The source vector carried an extra `−i` and the node weights an extra `iω/(2π)`. The error is invisible at ω = 2π (λ = 1), where every earlier run and validation was done. Confirmed before the fix: code/analytic free-space dipole = 0.5 at ω = π and 0.1592 at ω = 1, on every component, 1D and 2D.

The code now matches the write-up term by term:

- `Source_projection_Berreman_xyz`: `J_Et = kt·Jz/(ϵzz·ω)`, `J_Ht = R2s·Jt − R2s·ϵtz·Jz/ϵzz` (no `−i`).
- 2D weight `kρ·wρ·wφ/(4π²)` (was `iω·kρ·wρ·wφ/(8π³)`), 1D weight `i·wt/(2π)` (was `−wt·ω/(4π²)`).
- `ez_local_density`: `Ez_local = −p_z/(ϵzz·Δz)` (was `i·p_z/(ϵzz·Δz)`), from the same Et-jump relation.
- The box suite's copies of `Js`, `Ez_local` and the bulk-solve `J_Ht` sign follow.

Results at ω = 2π are unchanged (all suites pass with the same errors), so existing λ = 1 caches, figures and the Gila comparison stand. New check [f] in the point suite: near-vacuum stack vs the analytic free-space dipole at ω = 2π, π, 1, 3.7, 1D and 2D, worst 8e-12.

## Changes since 2026-09-24

- **2026-09-28/30: Gila anisotropic study.** Written up in `RESULTS_gila_anisotropic_comparison.md`, with metrics in `diagnostics/results_summary.csv` (regenerate with `diagnostics/summarize_results.jl`). Conclusions:
  - The own solver is correct for biaxial media (exact symmetry, kz and quadrature checks).
  - The Gila mismatch is dominated by the lateral truncation of Gila's slab. The opt-in Gila edge absorber (`absrel=0.25,1.0`) fixes most of it.
  - Gila's convergence degrades with negative ε_zz and strongly with lateral domain size. bicgstabl diverges at 3.0λ on R9 and also on R12 (proportionate losses, ε_zz = −9.3+0.91i), so the failure is tied to domain size, not to the χ values.
  - Figures use short `R<n>` run tags. Parameters are shown in an info strip on each figure.
- **2026-10-09: `conj` fix in `_berreman_core` (`mod_ModeSolver.jl`, ~l.138).** The elimination of E_z now uses `ϵzt = (ϵ31, ϵ32)` and `μzt = (μ31, μ32)` without `conj`, since D_z = ε₃₁Eₓ + ε₃₂E_y + ε₃₃E_z. The change only affects tensors with complex ε₃₁/ε₃₂ (or μ₃₁/μ₃₂). The cached fields and figures all predate it, but the stacks studied so far have zero or real xz/yz components, so they should be unaffected. **To do:** rerun `Gz_ClosedForm_Foundation_Validation.jl` and `Gz_ClosedForm_Point_Validation.jl` (the gyrotropic/biaxial cases) to confirm the fix, and add a case with complex ε₃₁ ≠ ε₁₃ if none covers it.
- The comparison modules `mod_GilaRun.jl`, `mod_OwnSolverRun.jl`, `mod_StackCompare.jl` and `compare_gila_own_solver.jl` are back in the folder and in use. The "Cleanup" list below records that they were moved out on 2026-09-24.

## Project contents

Load everything with `include("ClosedForm.jl")`.

| File | Role |
|---|---|
| `mod_ModeSolver.jl` | Berreman eigenmodes (`Modes_Stack`; closed form for iso/uniaxial, `_berreman_core` + `eigen` otherwise), materials, `_is_azimuthally_symmetric_stack`, `_rotate_modes`. |
| `mod_ScatteringMatrix.jl` | Interface/propagation S-matrices, Redheffer star product (`S_all_ingredients`, `S_all_Matrix`). |
| `mod_Sources.jl` | `SourceDef`, `ModalSeed`, `Dipole`/`Dipole_constructor`, `Sheet`. |
| `mod_SourceProj.jl` | `Source_projection_Berreman_xyz`: the point seed. |
| `mod_FieldSolver.jl` | `_R_cavity`, `_seed_from_sources`, `_cavity_resummation`, `_walk_stack`, `_cavity_transfer_columns` (→ `T^o`); `Modal_amp`/`Modal_amp_profile` (direct per-kt solve, the foundation test's reference). |
| `mod_KxGrid.jl` | `adaptive_kx_grid` (per-probe Gauss–Kronrod, merged partition), `branch_points`, `build_kphi_grid`, tail helpers. |
| `mod_UniformSource.jl` | Uniform DFT grid + FFT pair, `point_current_sheet`. |
| `mod_CavityTransfer.jl` | `CavityTransferCache1D` / `2D` (+ builders): `T^o`, `Q_src`, point seeds (`sb = −sib_raw`), mode field basis, `ϵ_src`. |
| `mod_ClosedFormField.jl` | Synthesis: `amplitude_at`, `direct_amplitude`, `interior_slab_at`, `ez_local_density`, `closed_form_G` (1D J0/J1/J2; 2D exact sum or NUFFT; `box_lateral`, `Δz_box`), `closed_form_pixel`, `box_lateral_factor`. |
| `mod_ClosedFormAssembly.jl` | Multi-source FFT assembly: `build_closed_form_green`, `apply_ghat`, `apply_closed_form`, `closed_form_field`. |
| `mod_ClosedFormGrid.jl` | Adaptive grids: `point_kx_grid` (1D, point) and `box_kρ_grid` (2D, box-aware), `box_tail_cutoff`. |
| `mod_Dipz_SemiAnalytical.jl` | Semi-analytical point-dipole model (vacuum/3-layer), kept only as an independent reference for the point validation. |
| `ClosedForm.jl` | Loader. |
| `Gz_ClosedForm_*_Validation.jl`, `Gz_ClosedForm_TestHelpers.jl` | Validation suite (below). |
| `compare_gila_own_solver.jl`, `mod_StackCompare.jl`, `mod_OwnSolverRun.jl`, `mod_GilaRun.jl` | Comparison with GilaElectromagnetics (below). Outputs: `figures/`, `field_cache/` (both solvers' fields), `greens_cache/` (Gila's vacuum operator). |
| `ShapeCompare_figs/` | Figures from the last legacy-vs-new comparison, plus your copy of that script (it needs the removed legacy modules, so it is an archive and won't run). |

## Validation suite (no legacy code; `julia --project=. -t 8 <file>`, each prints PASS/FAIL)

- **`Gz_ClosedForm_Foundation_Validation.jl`** (~5 s): `T^o·[S₊;S₋]` = `Modal_amp` / `Modal_amp_profile`, iso/uniaxial/biaxial/gyrotropic, 6.5e-15.
- **`Gz_ClosedForm_Point_Validation.jl`** (~2 min):
  - `point_kx_grid` + 1D path vs the semi-analytical model: 6.9e-11 external, and internal for rows |ζ| > 0.3, where the model is itself converged.
  - Batched = Σ singles: 3e-16.
  - 2D rotation fill = full solve = 1D: 6e-15; NUFFT = exact sum: 1e-13.
  - 90° rotation covariance on biaxial/gyrotropic: 5e-14.
  - Reciprocity between layers, iso and biaxial: 2e-14.
  - [f] Absolute normalization at ω = 2π, π, 1, 3.7: near-vacuum stack vs the analytic free-space dipole, 1D and 2D (added 2026-10-18 with the ω/2π fix).
- **`Gz_ClosedForm_Box_Validation.jl`** (~10 min):
  - Lateral factor = footprint average of point fields.
  - Exterior `D` = exact slab average (1D, 2D biaxial/gyrotropic; thick box with |Im Q|·Δz/2 ≈ 600; box touching an interface): 6e-13 or better.
  - Interior: `σ_b = −1`, `Ez_local` sign (1/kρ² decay), bulk 3×3 Maxwell solve 2e-15, face continuity, interior = exact average of point fields 1e-15.
  - Writes `Gz_ClosedForm_SourcePlane.png`.
- **`Gz_ClosedForm_Grid_Validation.jl`** (~5 min):
  - `point_kx_grid` vs a dense rule.
  - `box_kρ_grid` vs a dense rule per row group (self slab / neighbour voxel / across interface), compared with a point-probed axis. Converged to the tail bound; see open items.

## Gila comparison (2026-09-24)

`julia --project=. -t 8 compare_gila_own_solver.jl [stack]`, where `stack` is a constructor from `mod_StackCompare.jl` (default `positive_uniaxial_stack`). Fields are cached: a replot costs seconds; a fresh stack costs ~21 min (own solver) + ~5 min (Gila), at 1/32.

- **Setup.** 3-layer stacks: vacuum 1.5λ / slab 0.5λ / vacuum 2.0λ, 3λ × 3λ laterally, voxel s = 1/32 (96×96×128). Moment [1,1,−1]/√3 (own convention; ẑ_own = −ẑ_Gila, so Gila gets [1,1,1]/√3 and its Ez is negated on remap).
- **Source.** Gila fills one voxel with p/s³. The own source is now the same voxel: an s×s×s box (`box_kρ_grid` → `build_cavity_transfer_2d` → `closed_form_field`, `box_lateral=(s,s)`, `Δz_box=s`). `gila_source_cell` reproduces Gila's index rounding. It puts the source 0.484λ above the interface (not the requested 0.5λ) and half a voxel off the lateral centre; both codes use that cell. `shape_tag="point"` is kept as a baseline; the legacy puck is gone.
- **Quadrature (`default_own_opts`).**
  - `rtol=1e-3`, `kρ_cap=3000`, `N_phi=256`, `ρ_far` = half the lateral extent, NUFFT; 7305 kρ nodes.
  - The uniform φ axis needs `N_phi ≳ kρ·ρ`. Rows ≥ 7 voxels from the source converge with 256 (4e-9), but rows ±1 need 2048 (vs a 4096 reference: 2e-5 at 1024, 5e-9 at 2048).
  - So rows `|row − src| ≤ 6` use `N_phi_near=2048`. A single cache that size doesn't fit in 16 GB, so it is built and summed in chunks of 256 φ nodes (`_field_phi_chunked`, exact to 5e-15).
- **Source row: not converged.** It moves ~40% between `kρ_cap` 3000 and 6000 and 20–30% in φ up to 4096. This is the algebraic interior tail (Newell subtraction, open items), not a node-count problem.
- **Calibration.** The script fits one global complex factor c on the source axis in the top vacuum. With the box it comes out |c| = 0.997–0.999, arg c = 1.8–1.9°: there is no unit factor between the codes. Default is now raw (`CALIBRATE = false`, c = 1); figures are tagged `_raw` or `_cal`. Raw matches better in the top vacuum (~5% vs ~6%): the fitted phase was averaging a slow +3° → 0° drift toward the source into a −1.9° global offset.
- **Figures (8 per stack).**
  - Intensity slices xz/yz: TF | Gila | relative error on the total intensity.
  - Phase-per-component slices xz/yz: 3×3, TF arg | Gila arg | Δφ, with components below 0.1% of the intensity masked.
  - On-/off-axis intensity lines with relative intensity error.
  - On-/off-axis unwrapped phase per component with Δφ.
- **Results, raw** (complex-field vector error, L2 rel). Full suite, 2026-09-24:

  | stack (slab ε) | Gila | Gila time | top vacuum | near-source band | source row | slab | bottom vacuum | full |
  |---|---|---|---|---|---|---|---|---|
  | isotropic (2.25) | converged | ~5 min | 5.0% | 27% | 31% | 17% | 17% | 27% |
  | positive uniaxial (2.25, 2.25, 4.3) | converged | ~5 min | 4.9% | 27% | 31% | 19% | 17% | 27% |
  | gyrotropic (2.1 ± 3.14i off-diag, ε_zz = −1.0) | converged, ~13k mat-vecs | 78 min | 4.0% | 27% | 31% | 28% | 26% | 27% |
  | negative uniaxial (ε_z = −5.2) | **diverged** (cap) | 2 h 36 min | 100% | 86% | 48% | 100% | 100% | 100% |
  | anisotropic, small domain (2.25, 4.6, −3.2) | **stalled** (cap) | 58 min | 10% | 8.8% | 9.9% | 68% | 51% | 9.8% |

  - Converged stacks: around the source and in the top vacuum, the agreement is a few percent in intensity and ≤ 3° in phase. The legacy near-source artifact is gone (on-axis intensity error ~3–4% through the band). The calibration fit is |c| = 0.997–1.0003, |arg c| ≤ 1.9°.
  - It degrades in and below the slab and toward the domain edges. On axis, the intensity error peaks at ~20–40% near the bottom. The error lobes in the 2D maps reach the domain boundary, and for the gyrotropic slab the error saturates at the slab's lateral ends (x ≈ ±1.5λ), where Gila's slab is cut off.
  - Isotropic and uniaxial are nearly identical, and the isotropic case is the one validated against the semi-analytical model to 1e-11. So the degradation is attributed to Gila's truncated domain (its half-spaces and layers stop at the box edge; the own solver's are semi-infinite and laterally infinite), not to the own solver. This is not yet tested (see open items).
  - **Gila convergence (bicgstabl, reltol 1e-5, cap 20,000 mat-vecs, unpreconditioned).** Gila fails on the strongly negative-ε_z slabs.
    - Negative uniaxial (ε_z = −5.2): the residual went 3.2e4 → 6.0e2 (min), then grew from about iteration 2,500 to 2.1e5 at the cap. The field is not a solution (|c| fit 0.14, arg 37°), so no conclusion on the own solver.
    - Anisotropic (ε_z = −3.2): 9.2e3 → ~1e2 within ~500 iterations, then stagnant (min 63) until the cap. The field is partially usable near the source (|c| 1.018, arg 3.8°), not in or below the slab.
    - Gyrotropic (ε_zz = −1.0) converges, but ~15× slower than the positive stacks.
    - The figures of a non-converged run carry "-- GILA NOT CONVERGED" in their titles, and the script warns.
  - Own-solver cost: ~20–22 min per 3λ stack (7k kρ, 1.9M coarse nodes + 15M near-row nodes), whether or not the rotation fill applies (the gyrotropic stack's per-node Berreman solves cost about as much as the symmetric stacks' synthesis). ~8 min for the small-domain anisotropic stack (3.5k kρ).
- `default_gyrotropic_stack` is the stack of the legacy comparison. Its `legacy_path` Gila pin (`field_321deb6fef810c3.jld2`) is not in this folder, so it was recomputed (converged).

## Physics notes worth keeping

- **Point source in its own plane (ζ = 0).** The integral doesn't converge there. The code drops the direct term at exactly ζ = 0 (strict inequalities, as `Modal_amp_profile`). Near-source rows need the box.
- **Deep-evanescent nodes (`Im Qf·d_src > 709`).** `_R_cavity` overflows, so `T^o` is non-finite and the node's reflected part is dropped (it is ~e^{−2κ·dist}). The direct term is kept, evaluated with the combined decaying exponent.
- **Box.** Seen from outside its own z-slab, a box is exactly a point source with seeds `S(kt)·D·s`, where `D = sinc(Q·Δz/2)` per mode. It's evaluated in face-difference form, so `D` is never formed on its own. Inside the slab: face ramps + particular solution `−(iM)⁻¹Js/Δz` + `Ez_local = −p_z/(ϵzz·Δz)` (per node, spectrally; was `i·p_z/(ϵzz·Δz)` before the ω/2π fix). The legacy code had no `Ez_local`.
- **Last legacy comparison** (figures in `ShapeCompare_figs/`). The legacy point source and legacy box show a bright stripe along the whole source row; the new box is smooth. Legacy box build + first apply took 525–1170 s vs 80–130 s for the new box.

## Cleanup (2026-09-24)

- **Moved to `~/.Trash/G_z-z_Box-Source_cleanup_2026-09-24/`** (not hard-deleted; empty the Trash to discard for good):
  - legacy modules: `mod_Green{AdaptiveGrid,Grid,PixelKernels,Precompute,Solve}.jl`, `mod_VoxelGreen.jl`, `mod_Plotting.jl`, `mod_GilaRun.jl`, `mod_OwnSolverRun.jl`, `mod_StackCompare.jl`;
  - all legacy/legacy-comparison scripts and demos, including `compare_gila_own_solver.jl` and the main-folder `Gz_ShapeCompare_Validation.jl`;
  - the Phase 0–8 scripts (they compared against legacy code) and `legacy_backup/`.
- **Pruned from the kept modules:**
  - `ModeSolver`: `BerremanMatrix` wrappers, vacuum solvers, `_is_biaxial`, the `Combinatorics` import. The Berreman matrix itself (`_berreman_core`) and the general eigen path stay.
  - `ScatteringMatrix`: vacuum S-matrices.
  - `Sources`: `PlaneWave`, `Launch`, `ExtendedSource`.
  - `FieldSolver`: the `PlaneWave` seed.
  - `SourceProj`: the Phase-5b-only box wrapper.
  - `KxGrid`: `nearest_interface_distance`.
- **Added:** `point_kx_grid`, replacing the legacy `adaptive_layer_kx_grid` the 1D path used to borrow. `mod_BoxGrid.jl` is renamed `mod_ClosedFormGrid.jl`.
- **Fixed:** an observation plane exactly on a box face could round into the interior branch with `a = 0` and error; it's now clamped (continuous there).
- `Project.toml` still lists packages that only the removed code used: HCubature, KernelAbstractions, ProfileSVG, BenchmarkTools, Combinatorics. They are left in place so the Manifest isn't touched. Still needed: StaticArrays, FFTW, FINUFFT, Bessels, FastGaussQuadrature, QuadGK, and Plots for the figures. The Gila comparison needs GilaElectromagnetics, VoxelShapes, IterativeSolvers and JLD2 again.

## Open items (updated 2026-10-12)

Setup first:

- **Move to git.** Clone the GitHub repository into a new folder, copy the files listed under "Repository state", and add the `.gitignore`. Commit the `mod_ModeSolver.jl` fix separately, then work on branches.
- **Check the `conj` fix** with the validation suite (see "Changes since 2026-09-24").
- `Project.toml` still lists packages that only the removed legacy code used: HCubature, KernelAbstractions, ProfileSVG, BenchmarkTools, Combinatorics. Removing them changes the Manifest, so do it in its own commit.

Code and physics:

- **Quasi-static (Newell–Williams–Dunlop) subtraction.** The interior box integrand decays only algebraically (~1/kρ). This is also what keeps the Gila comparison's source row unconverged (31% there). `box_kρ_grid` converges the self/neighbour-voxel field to its tail bound (7e-8 at rtol 1e-4), but the field itself still moves 6–13% between kρ = 3000 and 6000. This should also cut the node count (5–7k kρ today) and the faint source-row artifacts seen in the ShapeCompare figures.
  2026-10-22 (`comparisons/gila_airy/`, vacuum, box λ/32): off the source row the own field matches an analytic point dipole to 9e-5. On the source row only the component along the slice (Ex from p_x on an x slice, i.e. the in-plane longitudinal one) is off, by 16–30% out to 1λ. Ey and Ez there match to 4e-3 and 6e-4. The error Δ/p_x is real and changes sign with ρ, the shape of a truncated kρ tail. kρ_cap 3000 → 6000 changes it but does not reduce it. Gila is within 1–6% of the dipole on that row, so the gap is on our side. N_phi_near = 2048 also aliases the source row beyond ~0.5λ (error ×50–100); 8192 removes that.
- **Speed-up of the 2D box materialization.** It is 60–110 s on the 60-row ShapeCompare volume, spent in the per-row phase/strength loops over ~375–600k nodes plus 9 NUFFTs per row.
- `CavityTransferCache2D`'s full-solve branch could use a y-mirror shortcut (≈2× fewer solves for mirror-symmetric biaxial stacks).
- §5.4 (observation-voxel averaging) is not done. Gila's field is a voxel collocation, while the own field is sampled at voxel centres; this matters most next to the source.
- **Lossless stacks are not supported, but small losses are enough except in hyperbolic layers.** The kρ quadrature runs on the real axis (breaks at the light lines only, no contour deformation or pole extraction), so lossless guided-mode poles sit on the path: every probe's finite band stalls, and `box_kρ_grid` now raises (2026-10-13). For iso/uniaxial stacks, δ = 1e-6 on every layer converges at the same cost and is ≈ 3e-5 from the lossless limit (see the loss sweep above), so no fix is needed there. Low-loss biaxial (incl. hyperbolic) layers: handled down to δ = 1e-3 by the φ-dependent kρ grid (2026-10-14); hyperbolic at 1e-6 converges per group but needs ≈ 56k kρ per φ. Pole extraction (PV + iπ·residue) or contour deformation (complex kρ, incompatible with the FINUFFT lateral synthesis) would address true lossless poles if ever needed.
- ~~Biaxial εzz = +2+0.3i source layer, far-lateral probes~~ (fixed 2026-10-13: a far-lateral probe sat outside the source layer, see above).
- **Near-source φ resolution.** Uniform φ with `N_phi ≳ kρ·ρ` is wasteful at low kρ. A kρ-dependent φ count per node (the cache is a product grid today) would replace the near/far split and the chunking.

Gila comparison:

- ~~Gila domain-truncation test~~ (done 2026-09-28/29, see `RESULTS_gila_anisotropic_comparison.md`). The lateral truncation is confirmed as the dominant error, and the edge absorber mostly removes it.
- **Converging Gila solve on a 3.0λ negative-ε_zz stack** (parked on purpose 2026-09-29; don't start unprompted). Needed to confirm the anisotropic conclusion on a large interior. Unpreconditioned bicgstabl diverges on R9 and R12. Candidates: restarted GMRES, a preconditioner in `mod_GilaRun.jl`, or slightly more in-plane loss.
