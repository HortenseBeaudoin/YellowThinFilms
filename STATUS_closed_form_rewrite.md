# Status: closed-form rewrite (`PLAN_closed_form_point_source_rewrite.md`)

Last updated 2026-10-12. Phases 0–8 of the plan are implemented and validated, and the legacy implementation has been removed from the project (see "Cleanup"). The comparison against GilaElectromagnetics has been ported to the closed-form box and run (see "Gila comparison"), and followed by the anisotropic study of 2026-09-28/30 (see "Changes since 2026-09-24"). Next: move the project into a git checkout of the GitHub repository (see "Repository state"), then the quasi-static (Newell) subtraction (§4.2) and the speed-up of the 2D box materialization.

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
  - cross-layer and biaxial scattered: low-rank skeletons in z (measured ranks ~10 of 32 at 1e-6, except on a hyperbolic layer's own side), with dense per-pair as the fallback;
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
- **`Gz_ClosedForm_Operator_Validation.jl`** (~45 s): term split [t]; operator = `closed_form_field` for random moments, all layers, point/box, source layers 1/2, structured / toeplitz_dense / dense [o]: ≤ 1.3e-15; cost [m].
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
  - Biaxial εzz = +2+0.3i (Gila R3) is not in the table: see open items.
- **Next:** low-rank skeletons in z for the cross-layer blocks and the biaxial scattered block (step 3). The cross-layer dense block now dominates memory and build time; layer 1 → 3 (48×64) would be ≈ 4 GB at this size.

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
- **Box.** Seen from outside its own z-slab, a box is exactly a point source with seeds `S(kt)·D·s`, where `D = sinc(Q·Δz/2)` per mode. It's evaluated in face-difference form, so `D` is never formed on its own. Inside the slab: face ramps + particular solution `−(iM)⁻¹Js/Δz` + `Ez_local = i·p_z/(ϵzz·Δz)` (per node, spectrally). The legacy code had no `Ez_local`.
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
- **Speed-up of the 2D box materialization.** It is 60–110 s on the 60-row ShapeCompare volume, spent in the per-row phase/strength loops over ~375–600k nodes plus 9 NUFFTs per row.
- `CavityTransferCache2D`'s full-solve branch could use a y-mirror shortcut (≈2× fewer solves for mirror-symmetric biaxial stacks).
- §5.4 (observation-voxel averaging) is not done. Gila's field is a voxel collocation, while the own field is sampled at voxel centres; this matters most next to the source.
- **Lossless stacks are not supported.** The kρ quadrature runs on the real axis (breaks at the light lines only, no contour deformation or pole extraction), so lossless guided-mode poles sit on the path. On lossless iso and uni stacks every `box_kρ_grid` probe's finite band stalls (panels down to ~1e-13). `box_kρ_grid` still returns a grid, with only a warning. Options: pole extraction (PV + iπ·residue, analytic per pole; keeps the real-axis grid and the NUFFT), contour deformation (needs complex kρ, so the FINUFFT lateral synthesis would have to change), or limiting absorption (reference only).
- **Biaxial εzz = +2+0.3i source layer, far-lateral probes.** With `ρ_far = 1` the grid needs 65k kρ at Δ = 1/8 and stalls at Δ = 1/16; with `ρ_far = 0` it converges with 1,290 kρ in 10 s. Only for biaxial does the node count grow with N_phi (7k / 19k / 65k at 8 / 64 / 256 φ). Not the φ resolution (iso is as under-resolved at kρρ > 256 and converges) and not mode-solve jumps (single-φ integrands are smooth in kρ, including φ = 0, π/2). Unexplained.
- **Near-source φ resolution.** Uniform φ with `N_phi ≳ kρ·ρ` is wasteful at low kρ. A kρ-dependent φ count per node (the cache is a product grid today) would replace the near/far split and the chunking.

Gila comparison:

- ~~Gila domain-truncation test~~ (done 2026-09-28/29, see `RESULTS_gila_anisotropic_comparison.md`). The lateral truncation is confirmed as the dominant error, and the edge absorber mostly removes it.
- **Converging Gila solve on a 3.0λ negative-ε_zz stack** (parked on purpose 2026-09-29; don't start unprompted). Needed to confirm the anisotropic conclusion on a large interior. Unpreconditioned bicgstabl diverges on R9 and R12. Candidates: restarted GMRES, a preconditioner in `mod_GilaRun.jl`, or slightly more in-plane loss.
