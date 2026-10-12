# Plan: closed-form (z,z')-separable rewrite, point + box sources, FFT-based multi-source assembly kept

## Purpose of this document

Standalone implementation plan for a new session with no memory of the conversation that
produced it. It rewrites the field-synthesis layer of this codebase (everything above the
eigenmode/scattering-matrix/cavity-solve foundation) around one idea: **the dependence on
source depth `z` and observation depth `z'` can be made closed-form and separable,
computed once per `(source_layer,obs_layer)` pair, instead of re-solved or re-sampled per
source position.** On top of that, the box source is treated as a genuine **volumetric
source in reciprocal space**: its 3D Fourier transform, sampled on the layer's own
dispersion surface, enters as an exact per-mode factor on the modal seeds — not as a
real-space average of point-source fields. See §5 — the part of this rewrite that changes
the most relative to today's code (and that also reshapes the adaptive grid, §4.2).

**Scope, revised from an earlier draft of this plan:**
- Sources: **point** and **box** only (no puck/`RadialShape`, no `GeneralShape`). A box is
  always a full 3D rectangle `Δx_box×Δy_box×Δz_box` (square footprint allowed). There is no
  "point laterally but extended in z" source — don't build special cases for one.
- Media: isotropic, uniaxial, biaxial, gyrotropic — all four, not just isotropic/uniaxial.
- Multi-source lateral assembly via **FFT/IFFT on a uniform transverse grid is kept** —
  this earlier draft proposed dropping it for NUFFT-only; that's reversed. NUFFT still has
  a role, but a narrower, different one — see §4.4.
- The 1D, analytically-azimuthal-reduced (`J0,J1,J2` Bessel) fast path is **narrowed**: it
  applies **only** to a point source on an isotropic/uniaxial stack. Box (on *any* medium)
  and point-on-biaxial/gyrotropic both go through the 2D `(kρ,φ)` path.
- The 2D path itself has two fill strategies, dispatched on medium symmetry, both kept:
  the azimuthally-symmetric shortcut (solve once at `φ=0`, get every other `φ` column by
  an exact rotation — "the integral with the phi rotation factors in it") for
  isotropic/uniaxial stacks, and full independent per-`(kρ,φ)` solves ("the full
  computation of those") for biaxial/gyrotropic stacks.

Work phase by phase (§8); validate each phase against the existing code before moving on
(§9).

---

## 1. The physics being implemented

### 1.1 Master identity

```
E_q(Δx,Δy,z,z') = ∫∫ d²kt/(2π)² S(kt)·e^{ikt·(Δx,Δy)} Ê_q(kt,z,z')     kt=(kx,ky)=(kρcosφ,kρsinφ)
```

`S(kt)` is the lateral shape factor: `S≡1` for a point; for a rectangular box of size
`Δx_box×Δy_box`, `S(kx,ky)=sinc_phys(kx·Δx_box/2)·sinc_phys(ky·Δy_box/2)`,
`sinc_phys(u)=sin(u)/u` — this **is** the box footprint's own 2D Fourier transform (exact
identity, not an approximation), so it composes as a plain multiplicative factor on the
integrand, evaluated at whatever `(kx,ky)` the node represents — no separate φ-quadrature
is needed for it (see §4.3).

`S(kt)` is only the **lateral** half of the box's spectrum. The box's full 3D transform is
`J̃(kt,kz) = −iωp·S(kt)·sinc_phys(kz·Δz_box/2)`. `kt` is conserved through the stack, so
`S(kt)` is one scalar per node. `kz` is **not** conserved: each layer only carries
`kz = Q_n(kt)`, its four Berreman eigenvalues. The axial factor therefore enters per mode, at
the modal-seed level (§5.1), not as a scalar integrand multiply.

`Ê_q(kt,z,z')` is the per-plane-wave response from the stack's eigenmode +
Redheffer-star-product machinery — the physics from §1.2 below, applied identically
whether `kt` sits on a 1D radial grid or a 2D `(kρ,φ)` grid (the closed-form-in-`(z,z')`
argument never assumed anything about how the lateral/azimuthal axis is organized).

### 1.2 The closed-form (z,z') separation — the core idea, unchanged from the physics derivation

For one fixed `kt` node:

```
Ê_{β,m}(kt,z,z') = A_{β,m}(kt;z) · (mode field basis)_{β,m}(kt) · Φ_{β,m}(z')

Φ_{f,m}(z')  = exp(i·Qf,obs[m](kt)·z')            -- obs layer's own forward Q, ref'd to z=0
Φ_{b,m}(z')  = exp(i·Qb,obs[m](kt)·(z'-d_obs))     -- obs layer's own backward Q, ref'd to z=d_obs

A_{β,m}(kt;z) = Σ_{γ∈{f,b}} Σ_{n=1,2}  T^{o}_{β,m;γ,n}(kt) · s_{γ,n}(kt) · Ψ_{γ,n}(z)

Ψ_{f,n}(z) = exp(i·Qf,src[n](kt)·(d_src−z))        -- exactly `_SProp(d-zpos,Qf)`
Ψ_{b,n}(z) = exp(−i·Qb,src[n](kt)·z)                -- exactly `_SProp(-zpos,Qb)`
```

`β,γ∈{f,b}` (observation/source branch), `m,n∈{1,2}` (observation/source polarization,
always exactly 2 — the two Berreman eigenmodes of whichever layer).

**Why exact:** the source's bare excitation `s_{γ,n}(kt)` (`sif[n]`/`sib[n]` from
`Source_projection_Berreman_xyz`) is `z`-independent. It is multiplied by exactly one
mode-diagonal exponential of `z` (`_SProp`), then passed through a chain of purely
linear, `z`-independent matrix operations (the cavity resummation
`(I-ΦF·RL·ΦB·RR)^{-1}`, then the Redheffer star-product walk to `obs_layer`). A linear
map of a sum of exponentials is the same sum of exponentials — so `A_{β,m}(kt;z)` is
*exactly* a ≤4-term combination of `Ψ_{f,1},Ψ_{f,2},Ψ_{b,1},Ψ_{b,2}`, for **any** `z`,
with coefficients `T^{o}_{β,m;γ,n}(kt)` depending on `(kt,source_layer,obs_layer)` only —
never `z`, `z'`, shape, or moment.

`T^{o}(kt)` is a fixed **4×4 complex matrix per `kt` node per `obs_layer`**, built once
from `_R_cavity`'s `RL,RR,RR_right,Qf,Qb,d` (`FieldSolver`, already `z`-independent) and
the `Sall` star-product matrices (`ScatteringMatrix`, already cached, already
`z`-independent) — see §4.1 for construction.

**Box source:** the only change to this structure is the seed.
`s_{γ,n} → S(kt)·D_{γ,n}(kt;Δz_box)·s_{γ,n}`, with `D_{γ,n} = sinc_phys(Qγ,src[n]·Δz_box/2)`,
and `z` is the box's centre `z̄`. This is exact for every observation point outside the
source's z-slab (§5.1–5.2). `T^{o}`, `Ψ` and `Φ` are unchanged, so **one cache serves point
sources and boxes of any size** — `S` and `D` are applied at synthesis time (§4.3), never
baked into the cache.

### 1.3 Same-layer specialization (`source_layer == obs_layer`)

`Q_obs = Q_src` there, so the *reflected* part (everything in §1.2 above) collapses
further: `Ψ_{γ,n}(z)·Φ_{β,m}(z')` combines two same-`Q[n]` exponentials into one
exponential of `σ=z+z'` (up to fixed boundary offsets) — a closed-form sum of ≤2
exponentials, not a sampled array.

Separately, there is a **direct term** — the source's own free-space-like radiation,
present only on the causal side of the source (`ζ=z'-z ≥0` forward branch, `≤0` backward
branch). It's *also* closed-form and simpler still: `sif[n]·exp(iQf[n]·ζ)` for `ζ≥0`
(`sif` straight from `Source_projection_Berreman_xyz`, no cavity solve needed for this
piece at all). The only non-closed-form aspect is the **discrete branch choice** — a
Heaviside step in `ζ`, not representable as `f(z)f'(z')`. Keep this as an explicit branch,
never try to eliminate it. For a box it becomes a **three-way** branch —
`ζ ≥ Δz_box/2` (forward, seed-scaled), `ζ ≤ −Δz_box/2` (backward, seed-scaled), and
`|ζ| < Δz_box/2` (inside the source slab, §5.3) — at **every** lateral position, not just
inside the box footprint.

---

## 2. What's kept as-is (foundation layer)

| Module | Role | Notes |
|---|---|---|
| `mod_ModeSolver.jl` (`ModeSolver`) | Per-layer Berreman eigenmode solve (`Modes`, `Modes_Stack`), material types, `_is_azimuthally_symmetric_stack`, `_rotate_modes` | Kept exactly. `_rotate_modes` is central to §4.2's 2D-grid fill dispatch. |
| `mod_ScatteringMatrix.jl` (`ScatteringMatrix`) | `S_all_Matrix`, `S_all_ingredients`, `ScatteringM` (Redheffer star product) | Kept exactly. |
| `mod_FieldSolver.jl` (`FieldSolver`) | `_R_cavity`, `_cavity_solve`, `Modal_amp`, `Modal_amp_profile`, `_SProp` | Kept, **plus one addition** — §3. |

**Also required, not in the original 3-module list — confirm before deleting anything else:**
- `mod_SourceProj.jl` (`SourceProjection`, `Source_projection_Berreman_xyz`) — bare
  `z`-independent point seed `s_{γ,n}`. Keep as-is: it stays the **point** seed, and the
  cache stores it. Box factors (`S`, `D`) are applied downstream (§4.3/§5.1), so the cache
  never depends on box size. Add one thin wrapper,
  `Source_projection_Berreman_xyz_box(kt,ω,ϵ,modes; Δx,Δy,Δz)` returning
  `S(kt)·D·s_point`. It's used **only** for Phase 5b (§8), to test the box seed on today's
  unchanged pipeline.
- `mod_Sources.jl` (`SourceHandling`: `Dipole`, `SourceDef`, `Sheet`, `Launch`/`Right`/`Left`)
  — types consumed by both of the above. Keep as-is. `Sheet` in particular is the
  multi-point-source current-density container the FFT assembly (§4.4) consumes.

## 3. The one required addition to the foundation layer

`_cavity_solve` (`FieldSolver`) currently takes `sources` and immediately evaluates each
source's `z`-exponential before the cavity resummation. To build `T^{o}(kt)` (§1.2), you
need the resummation-and-walk **linear map itself**, evaluated at the 4 unit basis inputs
`(S₊,S₋)=e_1..e_4` instead of the physical seed (exact, since everything downstream of
`S₊,S₋` is linear).

**Pick one, note the choice in the new code's docstring:**
1. **Minimal FieldSolver extension (recommended):** factor `_cavity_solve` into
   `_seed_from_sources(...)` (unchanged) + `_cavity_resummation(S₊,S₋,...)` (the linear
   part, currently inline) + one new exported `_cavity_transfer_columns(stack,layer,modes,Sall)`
   that runs the resummation+walk at the 4 unit bases and returns `T^{o}` per target
   `obs_layer`. Pure extract-and-reuse, no behavior change to existing functions.
2. **No FieldSolver edits:** duplicate the ~30-line resummation+walk in the new module,
   calling `FieldSolver._R_cavity` (already reachable via qualified import) for the
   topology matrices.

---

## 4. New modules

### 4.1 `mod_CavityTransfer.jl` — the "Layer B" cache, built on EITHER grid type

Two build routines sharing the same `_cavity_transfer_columns` core (§3):

- `build_cavity_transfer_1d(stack,ω,source_layer; obs_layers) -> CavityTransferCache1D`
  — for the **point + isotropic/uniaxial fast path only**. Uses `adaptive_kx_grid`
  (§4.2), solves modes at `kt=(kx,0)` only (mirrors today's `build_field_cache`), fills
  `T^{o}(kx)` per radial node.
- `build_cavity_transfer_2d(stack,ω,source_layer; obs_layers) -> CavityTransferCache2D`
  — for **everything else**: box on any medium, or point on biaxial/gyrotropic. Uses
  `build_kphi_grid` for the `φ` axis (§4.2). Fills `T^{o}(kρ,φ)` via:
  - `_is_azimuthally_symmetric_stack(stack)==true`: solve `Modes_Stack` **once** at
    `φ=0` per `kρ`, then get every other `φ` column via `_rotate_modes` (exact, `Rot4(φ)`
    on the Berreman 4-vector — verified to ~1e-15 already in this codebase). This is "the
    integral with the phi rotation factors in it."
  - `_is_azimuthally_symmetric_stack(stack)==false`: solve `Modes_Stack` independently at
    every `(kρ,φ)` node. This is "the full computation of those."

Both cache structs hold, per node: `T` (4×4 per `obs_layer`), `Qf_src,Qb_src` (source
layer, node-only), `Qf_obs,Qb_obs,d_obs` (per `obs_layer`), the mode field basis
`(Ex,Ey,Ez)_{β,m}` (per `obs_layer`, ports `_mode_field_core`/`_mode_field_eps` — currently
in `mod_GreenSolve.jl`, port verbatim into this module, they don't belong to the modules
being retired conceptually, just happened to live there).

`s_{γ,n}(kt;q)` (`Source_projection_Berreman_xyz` output, per canonical moment
`q∈{x̂,ŷ,ẑ}`) is stored alongside, on whichever grid was built. Store the **point** seed
only, never `D·s` — `Qf_src,Qb_src` are already there, which is all `D` needs (§5.1).
For the interior-slab term (§5.3), also store the source layer's `ϵ` (3×3). It's needed
for the bulk `kz=0` solve and the local `Ez` term.

### 4.2 Reciprocal-space quadrature — port of `mod_KxGrid.jl`, with box-aware probing

`adaptive_kx_grid`, `branch_points`, `evanescent_umax` (radial, 1D fast path),
`build_kphi_grid` (φ axis, 2D path) port as-is, **except for what the probes see**. Note the
1D and 2D grids are generally **not** the same `kρ` nodes — each is its own
independently-converged adaptive quadrature (the 1D one never needed the `φ` multiplier the
2D one carries), so build them separately, don't try to share.

**Box-aware grid (2D path).** The grid is converged against probe integrands. For a box, the
probes must be built from the **box** response — `S(kt)·D·s_point`, plus the §5.3
interior term when the probe point is inside the slab — not the point response. The two
integrands have very different tails, and converging on the wrong one wastes the
volumetric source's benefit, or under-resolves:

- **Point:** decays like `e^{−kρ|ζ|}`, and not at all at `ζ=0`. That's why today's grids need
  huge `kρ` ranges and `kx_ratio_max` caps.
- **Box, observation outside the slab:** envelope ≈ `|S(kt)|·e^{−kρ·h}/(kρΔz_box)`, where
  `h = |ζ| − Δz_box/2` is the distance to the box's **near face**. So `evanescent_umax`'s
  `Leff` must be `h` (plus the usual distance-to-interface for reflected terms), **not**
  `|ζ|`. The z-extent makes the exponential decay slightly *worse*, not better. What it
  adds is algebraic decay: `1/(kρΔz_box)`, times the lateral sinc product.
- **Box, observation inside the slab:** decays algebraically (lateral sinc ×
  `1/kρ` from the face ramps; `1/kρ²` for the combined particular + local term, §5.3).

So the box integrand is absolutely integrable even at `h=0`, e.g. for adjacent voxels. It
has an **a-priori-known algebraic envelope**, roughly
`tail(K) ~ log K/(K·Δx_box·Δy_box·Δz_box)`, which can set an upper `kρ` bound directly from
the tolerance. Use it as the tail cutoff when the exponential factor is absent or weak
(`h·kmax ≲ 1`), instead of relying on `kx_ratio_max`. The lateral sinc's oscillation
(period `2π/Δx_box`) is already resolved wherever `e^{ikt·Δr}` is (`|Δr| ≳ Δx_box`); for
the self-voxel pixels, make sure the `φ`/`kρ` probes include one at `Δr≈0`.

**Optional, later — quasi-static subtraction.** At large `kρ`, the box integrand tends to
the static field of a uniformly polarised rectangular prism. That static field is known in
closed form in real space (Newell, Williams & Dunlop 1993, prism demagnetising tensor;
for diagonal `ϵ`, reduce by a coordinate rescaling). Subtract the asymptote per node and add
back the closed form in real space: the residual decays much faster. Only do this if the
algebraic tail turns out to dominate cost for self-/neighbour-voxel terms.

### 4.3 `mod_ClosedFormField.jl` — the per-pixel synthesis

```julia
amplitude_at(cache, obs_layer, zsrc_local, p::SVector{3,ComplexF64};
             Δz_box=0.0) -> (Af, Ab)   # per node, §1.2 + §5.1
```
computed **once**, before any pixel loop (mirrors today's `Axf`/`Axb` hoist exactly) —
`O(N_kt)` cheap exponential evals + `4×4` matvecs, no solve, for either grid type.
`zsrc_local` is the box **centre** `z̄`. `Δz_box>0` applies `D` to the seed (§5.1), with
`D≡1` for a point; the lateral `S(kt)` is applied at materialization (below). The
interior-slab piece (§5.3) is a separate routine, `interior_slab_at(...)`. It is called
only when `obs_layer==source_layer` and the observation plane `z'` lies inside
`[z̄−Δz_box/2, z̄+Δz_box/2]`.

Then materialize `G(x,y,z')` on the target uniform pixel grid (§4.4 needs this on a
uniform grid; a one-off query can skip the grid and evaluate directly at any `(Δx,Δy)`):

- **Point + isotropic/uniaxial** (`cache::CavityTransferCache1D`): port the structure of
  today's `_integrate_pixel`/`_integrate_pixel_rotated` — the `(J0,J1,J2)` Jacobi–Anger
  reduction is exact and unchanged, still applies to a point source. Keep the Bessel row
  cache (`_build_bessel_row_cache_vol`). **Drop** everything shape-dispatch related
  (`_integrate_pixel_shaped`, `RadialShape`/`GeneralShape` machinery, `BoxShape`'s old
  real-space `Ng²`-footprint-average route entirely — box never uses this path anymore).
- **Everything else** (`cache::CavityTransferCache2D`): port `integrate_pixel_2d` and its
  NUFFT accelerator `_transform_pixels_nufft_2d!`/`_nufft_nodes_2d` (`mod_VoxelGreen.jl`)
  — this is a type-1 NUFFT from the non-uniform `(kρ,φ)` frequency nodes onto the uniform
  pixel grid, replacing the brute-force `O(Nx·Ny·Nk·Nφ)` sum. For `S≢1` (box), fold `S(kρ,φ)`
  in as a single elementwise multiply on the phase kernel/NUFFT ramp (`_fill_G_2d!`/
  `_fill_G_2d_nufft!`'s existing pattern) — evaluated directly at each node's own
  `(kx,ky)=(kρcosφ,kρsinφ)`, no extra quadrature, exactly as derived in the source
  conversation's shape-factor discussion. **This is NUFFT's role in this rewrite** —
  accelerating one source's own materialization from a non-uniform quadrature onto a
  uniform pixel grid. It is *not* a replacement for the multi-source FFT step below;
  that's a separate, subsequent operation on an already-uniform array.

### 4.4 Multi-source lateral assembly — port `mod_UniformSource.jl` + `apply_voxel_green`'s FFT skeleton, largely as-is

**Kept per revision, not dropped.** For a given `zsrc_local` (one source depth — every
source at that depth, at whatever lateral positions, shares one `G`), §4.3 materializes
`G(x,y,z')` on the rigid uniform-DFT grid (`uniform_kxky_grid`), forward-FFTs it once to
`Ĝ(kx,ky,z')` (`_forward_transform`), and the **existing** `_apply_ghat` skeleton handles
any number of sources at that depth via
```
E(x,y,z') = sheet.weight · F⁻¹{ Ĝ(kx,ky,z')·F{sheet.J} }
```
— `mod_UniformSource.jl`'s `uniform_kxky_grid`, `_centering_ramp`, `_forward_transform`,
`_inverse_transform`, FFT plan caching, and `point_current_sheet`/`Sheet` construction
port essentially unchanged; none of it is shape- or z-handling-specific. Multiple source
**depths** each need their own `G`/`Ĝ` (built once per depth via §4.3, using the same
`CavityTransferCache`, cheaply — no re-solve) and are summed; multiple sources at the
**same** depth are handled for free by `sheet.J`'s own superposition.

---

## 5. What changes most: the box as a volumetric source in reciprocal space

Today's `_uniform_z_nodes`/`_gauss_z_nodes` (`mod_VoxelGreen.jl`) average a box voxel's field
numerically, in real space, over its z-extent: several z-shifted point-source evaluations
are weight-summed. The prototype `Gz_ZSpread_AnalyticProjection_Demo.jl` moved this to a
per-mode `sinc` scaling applied **after** the cavity solve. This rewrite goes one step
further and treats the box as what it is: a source with its own 3D plane-wave spectrum.
The spectrum enters at the **modal projection**, before the cavity; nothing downstream
changes. None of the `_uniform_z_nodes`/`_gauss_z_nodes` machinery survives.

### 5.1 Derivation: the box seed, and its operator form

**Conventions** (as in `Source_projection_Berreman_xyz`):
- **Tangential field:** `Ψ=(Ex,Ey,Hx,Hy)`. In a homogeneous layer, source-free, it obeys
  `dΨ/dz = iMΨ`, with `M` the 4×4 Berreman matrix at this `kt`.
- **Eigendecomposition:** `M·Vs = Vs·Λ`, with `Vs=[Vf Vb]` and `Λ=diag(Qf₁,Qf₂,Qb₁,Qb₂)`.
  So `M = VsΛVs⁻¹`, and a homogeneous solution propagates as `Ψ(z+u) = e^{iMu}Ψ(z)`.
- **Point sheet at `z₀`:** the source adds `Js·δ(z−z₀)` to the right-hand side, so `Ψ` jumps
  by `Js` across `z₀`.
- **Seeds:** `s = [sif;sib] = (Ws'Vs)⁻¹Ws'Js`. This equals `Vs⁻¹Js` exactly, since `Vs` is
  square and invertible; `Ws'` is only there for conditioning. So `Js = Vf·sif + Vb·sib`,
  and `Vs(Ws'Vs)⁻¹Ws' = I` just says "split `Js` into modes, then rebuild it".
- **Direct field of the sheet:** above it, `Vf·e^{iQf(z−z₀)}·sif`. Below it,
  `σ_b·Vb·e^{iQb(z−z₀)}·sib`, where `σ_b=±1` is whatever sign `FieldSolver`'s backward
  branch uses. A jump of exactly `Js` implies `σ_b=−1`; the implementer must confirm this
  against `FieldSolver` (§5.3 gives a built-in check).

**Box.** Take the lateral factor `S(kt)` out (a scalar, §1.1). The box of thickness `Δz`
centred at `z̄` has source term `Js·χ(z)/Δz`, with `χ` the indicator of `[z̄−Δz/2,z̄+Δz/2]`.
This is a continuum of point sheets at `z'=z̄+u`, each with weight `du/Δz`, the same `Js`
and the same seeds. By linearity, for an observation point `z` **above the whole box**:

```
Ψ(z) = (1/Δz)∫ Vf·e^{iQf(z−z̄−u)}·sif du
     = Vf·e^{iQf(z−z̄)}·[(1/Δz)∫_{−Δz/2}^{Δz/2} e^{−iQf u} du]·sif
     = Vf·e^{iQf(z−z̄)}·sinc_phys(Qf·Δz/2)·sif
```

The backward branch below the box works the same way. So **seen from outside its own
z-slab, the box is a point source at `z̄` with seeds**

```
s_box = S(kt) · D · s_point ,     D = diag(sinc_phys(Λ·Δz/2))
```

**Operator form (conceptual, not how it's coded).** The jump a point sheet at `z̄` would need
to produce these seeds is:

```
Js_box = Vs·s_box = S(kt) · Vs·D·Vs⁻¹ · Js = S(kt) · sinc(M·Δz/2) · Js
sinc(M·Δz/2) = (1/Δz)∫_{−Δz/2}^{Δz/2} e^{iMu} du
```

That is, each sheet's jump is carried to `z̄` by the layer's own propagator, then averaged.
The steps split → filter → rebuild give `sinc(MΔz/2)` instead of `I`.
`I − sinc(MΔz/2) ≈ M²Δz²/24` measures how far the box is from a point. Apply the factor
**once**: either `D` on the seeds, or `sinc(M·Δz/2)` on `Js` — never both.

**Why it has to be per mode.** For biaxial/gyrotropic layers `Qf₁≠Qf₂≠−Qb`, so `D` is not a
scalar. The axial spectrum is sampled at each mode's own `kz`; only the lateral `S(kt)` is a
plain scalar. For isotropic/uniaxial stacks, `Q` is invariant under `_rotate_modes`, so `D`
commutes with the φ-rotation fill (§4.1).

**Sanity properties:**
- `D=I` at `Q=0` and `S(0)=1`, so the total dipole moment is preserved (no normalisation).
- `sinc` is even, so `σ_b` doesn't enter `D`.
- `Δz→0` recovers `s_point` exactly.

### 5.2 Where D is applied, and numerics

- **Not in the cache** (§4.1 stores point seeds only), so one cache serves every box size.
  `D` is applied in `amplitude_at` (§4.3), where it multiplies `Ψ_{γ,n}(z̄)` anyway.
- **Range.** `D` grows like `e^{|Im Q|·Δz/2}` for evanescent modes. Everything downstream
  multiplies it by at least `e^{−|Im Q|·(distance from box face to interface)}`. The product
  is bounded, and floating-point multiplication keeps relative precision, so this is a
  **range** problem only:
  - `D` overflows once `|Im Q|·Δz/2 ≳ 700`.
  - `Inf·0` gives NaN when `_SProp`/`_phase` clamps a small exponential to 0.

  So never form `D` alone. In `amplitude_at`, form the product `D·Ψ_{γ,n}(z̄)` directly as
  the **face difference**:
  ```
  D·Ψ(z̄) = [ e^{iQ(ℓ+Δz/2)} − e^{iQ(ℓ−Δz/2)} ] / (iQΔz)
  ```
  with `ℓ = d_src−z̄` (forward) or `−z̄` (backward), both exponentials through the same
  clamped `_phase`. Use a small-`|QΔz|` guard (Taylor → `e^{iQℓ}`), as `ramp_forward` does
  in the demo. Physically this reads as the box being **two sheets at its faces**, with
  modal weights `±1/(iQ_nΔz)`.
- **Validity.** This is exact for **every** term whose source-side sheets all lie on the same
  side of the observation point. That covers all reflected terms, all cross-layer terms,
  and the same-layer direct term for `|ζ|≥Δz/2`, where `ζ=z'−z̄`.

### 5.3 Inside the source slab: face ramps + particular solution + local Ez

This applies only to the **same-layer direct term** with `|ζ|<Δz/2`, at **any** lateral
position (the slab is laterally infinite in reciprocal space). The sheets split into those
below `z'` (forward) and those above (backward). With `z_lo=z̄−Δz/2`, `z_hi=z̄+Δz/2`:

```
Ψ_dir(z') = Vf · [(e^{iQf(z'−z_lo)} − 1)/(iQfΔz)] · sif                 -- sheets below z'
          + σ_b·Vb · [(1 − e^{iQb(z'−z_hi)})/(iQbΔz)] · sib             -- sheets above z'
```

These are the demo's `ramp_forward`/`ramp_backward`; implement them with the same
small-`Q` guard, not as separate face terms, so `Q→0` is regular. Each ramp is a **face
wave** (`e^{…}/(iQΔz)`) plus a **constant**. With `σ_b=−1` the constants sum to
`Ψ_p = −(iM)⁻¹·Js/Δz`, the constant particular solution of `dΨ/dz = iMΨ + Js/Δz`.

**Check to implement:** numerically verify `Ψ_p = −(iM)⁻¹Js/Δz` at a few nodes. It pins down
`σ_b` unambiguously. In the residue picture, `Ψ_p` is the extra pole at `kz=0` that the box's
own axial spectrum `sinc(kzΔz/2)` contributes when `z'` is inside the box.

**Local Ez term — currently missing everywhere in the code.** `Ez` is algebraic in the
Berreman formalism: the ẑ row of ∇×H gives `Ez = (mode part) + Ez_local`, with
`Ez_local ∝ Jz/(ωϵ_zz)`. `_mode_field_core` (`mod_GreenSolve.jl:554`) builds `Ez` from mode
vectors only, so it has no `Ez_local`.
- **Point source:** it's a `δ(ζ)`, harmless to drop.
- **Box:** it's a finite depolarisation field (≈ `−p_z/(ϵ_zz·V)` in SI-like units), present
  wherever `z'` is inside the slab.

Rules for implementing it:
- **Add it per node, spectrally:** `S(kt)·Ez_local` with the box density `1/Δz`, in the same
  `(kt)` sum as the ramps. **Never** add it separately in real space. The modal particular
  `Ez(Ψ_p)` and `Ez_local` are each **flat** in `kρ` (non-decaying); only their sum decays,
  like `1/kρ²`. Isotropic TM check, done by hand: sum `= Ez_local·k0²ϵ/Q²`; modal part alone
  `→ −Ez_local` at large `kρ`.
- **Prefactor and sign:** derive them from the **same relation that produced
  `J_Et = −i·kt·Jz/(ϵzz·ω)`** in `Source_projection_Berreman_xyz`. `J_Et` *is* the
  Et-equation's `Ez`-coupling applied to `Ez_local`, so `Ez_local` is fixed by
  `_berreman_core`'s own convention (the code's `H` scaling is not assumed here).
  **Self-check:** with the right sign, `|Ez(Ψ_p)+Ez_local|` decays like `1/kρ²` at large
  `kρ`. With the wrong sign, it tends to `2·|Ez_local|`.
- **No other local terms:** `Et` and `H` get none. The `ϵ_tz·Jz/ϵ_zz` term in `J_Ht` is
  `Ez_local` feeding `Dt`; it's already in `Js`.

**Independent cross-check (Phase 7):** at each node, `Ψ_p`+`Ez_local` must equal the
**bulk** response of the source layer at `k=(kx,ky,0)` to the z-uniform current `J/Δz`.
That is one 3×3 Maxwell solve, `[(k²−ω²μϵ)−kkᵀ]` form in the code's units. It checks the
particular solution and the local term together, independently of the modal basis.

**Continuity checks:** at `z'=z_hi`, `Ψ_dir` must equal the §5.1 exterior formula (forward);
at `z'=z_lo`, the backward exterior formula. This follows from
`(e^{iy}−1)/(iy) = e^{iy/2}·sinc(y/2)`.

### 5.4 Optional, not in the main phasing: observation-voxel averaging

If the downstream use needs the field **averaged over an observation voxel** (e.g.
voxel-to-voxel coupling, Galerkin testing), it's the mirror image on the observation side:
- **Axial:** `Φ_{β,m}` gets `sinc_phys(Qβ,obs[m]·Δz_obs/2)`, in face-difference form, same
  numerics as §5.2.
- **Lateral:** `S(kt)` of the observation footprint multiplies in, so `|S|²` when source and
  observation voxels are equal. The spectrum then decays much faster, which is a large
  gain for the §4.2 tail.
- **Inside the slab:** the §5.3 interior becomes a double average. `kz=0` becomes a double
  pole, giving terms of the form `(e^{iQL}−1−iQL)/Q²`, plus the averaged `Ez_local`. Same
  structure, one more integration.

Do this only if a caller actually needs averaged observations. The main phasing (§8)
assumes point observations.

---

## 6. Modules to retire without porting

`mod_VoxelGreen.jl` (superseded — its orchestration *pattern* is what §4.3/§4.4 follow,
but none of its shape/`ζ,σ`-lattice/`zpos`-grid code survives, including the
`_uniform_z_nodes`/`_gauss_z_nodes`/`_shape_z_quadrature` z-averaging, replaced by §5), most of
`mod_GreenPixelKernels.jl`'s shape-dispatch code (`_integrate_pixel_shaped`,
`RadialShape`/`GeneralShape`, `BoxShape`'s old `Ng²` real-space route,
`_build_bessel_row_cache_box`), most of `mod_GreenSolve.jl` (superseded by §4.1/§4.3,
except port `_mode_field_core`/`_mode_field_eps` verbatim — the box's local `Ez` term
(§5.3) is added separately in the interior-slab routine, not inside `_mode_field_core`), `mod_GreenAdaptiveGrid.jl`'s
shape/voxel-specific probing (keep only the core adaptive-convergence-probe idea, already
folded into §4.2), `mod_GreenPrecompute.jl` (`StackGrid`'s role is now
`CavityTransferCache1D`/`2D`).

## 7. Deliberately out of scope

- `puck_shape`/`RadialShape`, `GeneralShape` — not carried forward at all (revised scope:
  point + box only). If a disk shape is wanted later, it's a `RadialShape`-style
  `S(kρ)` multiply on the 2D path, same mechanism as box, cheap to re-add.
- Anything requiring an azimuthally-**asymmetric** shape on the 1D path (moot — box
  always goes through the 2D path now, so the old tension between box's `φ`-dependence
  and the 1D path's analytic reduction doesn't arise; the 1D path is point-only).

## 8. Suggested phasing

1. **Phase 0:** §3's FieldSolver extension. Unit-test: feed the actual physical
   `(sif,sib)` as a linear combination of the 4 unit-basis results, must reproduce
   `Modal_amp`'s existing `Ef_tot`/`Eb_tot` bit-for-bit.
2. **Phase 1:** point + isotropic/uniaxial, cross-layer only (cleanest, fully separable,
   no direct-term branch). `CavityTransferCache1D` + the 1D half of `mod_ClosedFormField.jl`.
   Validate against today's `build_voxel_green`/`apply_voxel_green` (`PointShape()`,
   single `zsrc_local`) at several `(source_layer,obs_layer,z,z')`.
3. **Phase 2:** same-layer, point, isotropic/uniaxial — add the direct-term branch.
   Validate against `build_voxel_green_layer`/`apply_voxel_green_layer`'s existing exact
   ζ/σ output.
4. **Phase 3:** §4.4's FFT multi-source assembly, wired to Phases 1-2's closed-form `G`.
   Validate against `Gz_VoxelMultiDipole_Validation.jl` (point-only scenarios) and
   `Gz_fft-transverse_Validation.jl`.
5. **Phase 4:** 2D path, `CavityTransferCache2D`, point source on a biaxial/gyrotropic
   stack — validate the `_rotate_modes` branch first by running it on a *deliberately
   isotropic* stack and checking it reproduces Phase 1/2's 1D results exactly (a stack
   that's azimuthally symmetric should give identical answers via either path); then
   validate the full-solve branch against `Gz_ShapeCompare_Biaxial_Validation.jl`.
6. **Phase 5:** box shape, lateral factor only (§1.1's `sinc` on the 2D path,
   `Δz_box=0`). Validate against today's `box_shape` (`force_2d=true` route specifically,
   since that's today's closed-form-`S(kρ,φ)` path, the correct ground truth — not the
   `Ng²` real-space route, which this rewrite supersedes).
7. **Phase 5b (quick, de-risks §5 before any new synthesis code):** plug
   `Source_projection_Berreman_xyz_box` (§2) into **today's unchanged pipeline**, with
   `S≡1` so only `D` is tested. Compare with today's numerical `Δz_spread` at observation
   planes **outside** the source slab (`|ζ|≥Δz/2`, plus cross-layer). They must agree to
   quadrature tolerance. **Inside** the slab they must *disagree*; that difference is the
   §5.3 work, not a bug. At moderate `kρ·Δz` this also confirms the seed-level `D` doesn't
   hit the §5.2 range problem.
8. **Phase 6:** §5.1–5.2 in the new code — `D` applied in `amplitude_at` in face-difference
   form, exterior only (reflected + cross-layer + direct with `|ζ|≥Δz/2`). Validate against
   Phase 5b and against `Gz_ZSpread_AnalyticProjection_Demo.jl`'s post-cavity scaling (same
   physics, different placement: must match to roundoff). Include strongly evanescent nodes
   and a box touching an interface (`d−z̄ = Δz/2`) to exercise the range guard.
9. **Phase 7:** §5.3, inside the source slab. In order:
   (a) `Ψ_p = −(iM)⁻¹Js/Δz` numerically at a few nodes, which fixes `σ_b`;
   (b) `Ez_local` sign via the `1/kρ²` self-check;
   (c) `Ψ_p`+`Ez_local` against the bulk `k=(kx,ky,0)` 3×3 solve, on a **biaxial** layer
       (not just isotropic);
   (d) continuity at `z'=z_lo, z_hi` with Phase 6's exterior;
   (e) brute-force fine-node average of point-source fields **plus** `Ez_local`, as the
       final oracle.
   `Gz_ZSpread_Demo.jl`'s same-layer numbers are **not** a valid oracle inside the slab for
   `p_z` (§9).
10. **Phase 8:** box-aware adaptive grid (§4.2). Probes built from the box response,
    `Leff = h` (face distance), and the a-priori algebraic tail cutoff when `h·kmax ≲ 1`.
    Validate by convergence: refine the grid until the box field changes by less than the
    tolerance, and check the new grid reaches it with fewer nodes than a point-probed grid
    (the expected payoff), especially for self- and neighbour-voxel pixels. The optional
    quasi-static (Newell) subtraction comes after this, only if the algebraic tail dominates
    cost.

## 9. Validation strategy

Don't delete any existing module until the new code reproduces its output. Keep
`Gz_fft-transverse_Validation.jl`, `Gz_Voxel_Validation.jl`,
`Gz_VoxelLayerFastPath_Validation.jl`, `Gz_VoxelMultiDipole_Validation.jl`,
`Gz_ShapeCompare_Biaxial_Validation.jl`, `Gz_BoxClosedForm_Validation.jl`,
`Gz_ZSpread_Demo.jl`, `Gz_ZSpread_CrossLayer_Demo.jl` as oracles. For matched
`(kt,z,z',shape)`, old and new code compute the *same* physics through different
bookkeeping — not different approximations — so any disagreement beyond
quadrature/roundoff tolerance is a bug in the new code. Only retire §6's modules once all
of §8's phases pass this bar.

**One known exception — the oracles are incomplete inside a box's own z-slab.** Today's
code has no local `Ez_local ∝ Jz/(ωϵ_zz)` term (§5.3; `_mode_field_core` builds `Ez` from
mode vectors only). Its numerical z-spread (`_uniform_z_nodes` averaging of point-source
fields) is therefore missing that term wherever the observation plane is inside the source
slab. For `p_z` (and `p_x,p_y` on layers with `ϵ_zt≠0`), the new code must differ from
those oracles there, by exactly the spectrally-summed `S(kt)·Ez_local`. Check that
difference explicitly rather than "fixing" the new code to match. Everywhere outside the
slab, the oracles remain exact references.
