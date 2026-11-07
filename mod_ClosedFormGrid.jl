"""
mod_ClosedFormGrid.jl -- reciprocal-space quadrature for the closed-form caches
(`PLAN_closed_form_point_source_rewrite.md` §4.2,
`HANDOUT_quadrature_hyperbolic_gyrotropic.md`):

  - `point_kx_grid`    -- 1D, point source, azimuthally symmetric stack;
  - `point_kρ_grid_2d` -- 2D `(kρ, φ)`, point source, any stack (biaxial, tilted,
    gyrotropic): one kρ rule shared by every φ, `N_φ` from the geometry;
  - `box_kρ_grid`      -- 2D, box source, any stack (`box_kρ_grid_grouped`: one kρ
    rule per φ group).

Every builder starts from `KSpaceFeatures.kspace_features`: light lines (full ε
and μ), real-axis poles from a scan of the source layer's cavity response, and a
cut-off set by the geometry (`target_decay/h_min`). The poles are inside the
finite band `[0, Kcut]`; there is no tail. Then one of two engines
(`KSpaceFeatures.choose_engine`, from the materials):

  - `:fixed`    -- feature-driven composite Gauss–Legendre (`fixed_kρ_rule`):
    graded onto the singular light lines, panel ≈ pole width over every pole
    cluster, coarsening past them while following `e^{−kρh}` and `e^{ikρρ}`.
    Default for stacks with a metallic, hyperbolic or gyrotropic layer.
  - `:adaptive` -- h-adaptive Gauss–Kronrod (`KxGrid.adaptive_kx_grid`) on probe
    integrands evaluated through the SAME closed-form synthesis the field uses,
    seeded with the features. Default for all-dielectric stacks; always used by
    the box grid (its probes include the box's interior term).

`check=true` (point builders) rebuilds with every panel halved (and `2N_φ` in 2D),
evaluates the probes on both rules and reports the relative change in `info.check`.

── `box_kρ_grid` ──

The kρ axis of the 2D `(kρ,φ)` grid is converged against probe integrands
built from the BOX response -- `S(kt)·D·s_point` for observation points outside
the source slab, plus the §5.3 interior term (face ramps, particular solution,
local Ez) inside it -- instead of the point response. The φ axis stays the fixed uniform `build_kphi_grid`.

Each probe is a function of kρ alone: the full box Green's tensor at one
probe point, evaluated through the SAME synthesis the field uses
(`closed_form_G` on a one-kρ-node `CavityTransferCache2D`, φ-summed on the
fixed φ grid), so what is converged is exactly what is later integrated. The
per-kρ cache is built once and evaluated at every probe; the probe values are
memoized (not the cache, which is too large to keep for every visited kρ).

Probes (for each box-centre depth in `zbar`):
  - self voxel `(0,0,z̄)` and lateral neighbour `(Δx_box,0,z̄)` -- inside the slab,
    no exponential decay at all (`h = 0`);
  - vertical neighbours `(0,0,z̄ ± Δz_box)` in the same layer, when they fit
    (`h = Δz_box/2`);
  - the two adjacent layers, just across each interface (reflected/transmitted
    content, `h` = face-to-interface distance + a small offset);
  - with `ρ_far > 0`: a far lateral point just outside the slab and one inside it
    (`(ρ_far,0,z̄)`): the interior integrand at large `kρ·ρ` oscillates without any
    decay, so the off-centre pixels of the self slab need this probe for the
    band to resolve `e^{i kρ ρ cosφ}` out to the cut-off.

Cut-off. Outside the slab the box integrand decays like `e^{−kρ·h}` with `h` the
distance to the box's NEAR face (not to its centre). Inside the slab there is no
exponential factor: the integrand decays only algebraically, with the a-priori
envelope `tail(K) ~ log(KΔ)/(KΔ)`, `Δ = min(Δx_box, Δy_box, Δz_box)`. Each probe's
cutoff is whichever cuts first, `min(target_decay/h, K_box)` with `K_box` from
`log(KΔ)/(KΔ) ≤ rtol` (`box_tail_cutoff`); inside the slab (`h = 0`) only `K_box`
applies. The probes share one grid, so the cut-off is the largest of these, at
least 80·kmax and at most `kρ_cap` (`KxGrid.tail_bound`). In practice
`K_box ≫ kρ_cap` as soon as an interior probe is present, so `kρ_cap` is what
truncates the self-slab band: the field inside the source slab is converged to
that bound, not absolutely. The φ grid must resolve `kρ·ρ` up to that bound for the
off-centre probes -- `N_phi` is the caller's. The optional quasi-static (Newell)
subtraction of §4.2, which would remove that truncation, is not implemented.

Losses. The kρ path is the real axis (no contour deformation or pole
extraction), so a lossless stack puts its guided-mode poles on the path. A fully
lossless stack gets a warning up front (`is_lossless`); a stalled adaptive band
raises (`on_stall=:error`, the default) instead of returning an unconverged grid.
Iso/uniaxial stacks converge with δ = Im ε ≈ 1e-6 on every layer, ≈ 3e-5 from the
lossless limit (`diagnostics/loss_sweep.jl`). The fixed engine never stalls, but
its pole panels scale with the pole width: a lossless pole has none, and the scan
reports it at the floor width `1e-12·q`.
"""
module ClosedFormGrid

export point_kx_grid, point_kρ_grid_2d, chunked_2d, box_kρ_grid, box_kρ_grid_grouped, phi_groups, box_tail_cutoff, is_lossless

import ..ModeSolver: AbstractLayer, _is_azimuthally_symmetric_stack
import ..KxGrid: adaptive_kx_grid, kronrod_rule, branch_points, build_kphi_grid, thinnest_finite_layer,
                decay_cutoff, tail_bound, gl_rule, halve_panels
import ..KSpaceFeatures: kspace_features, fixed_kρ_rule, adaptive_seeds, choose_engine, _grade_for
import ..CavityTransfer: build_cavity_transfer_1d, build_cavity_transfer_2d
import ..ClosedFormField: closed_form_G, closed_form_pixel
using Base.Threads
using LinearAlgebra: norm

"""
    box_tail_cutoff(Δ, rtol) -> K

Smallest `K` with `log(K·Δ)/(K·Δ) ≤ rtol` (the algebraic box-tail envelope,
§4.2), by fixed-point iteration on `K = log(K·Δ)/(rtol·Δ)`.
"""
function box_tail_cutoff(Δ::Real, rtol::Real)
    K = 10 / (rtol * Δ)
    for _ in 1:50
        K = max(log(K * Δ), 1.0) / (rtol * Δ)
    end
    return K
end

"""
    is_lossless(stack) -> Bool

`true` when no ε (or μ) component of any layer has a nonzero imaginary part:
the guided-mode poles then sit on the real kρ axis.
"""
is_lossless(stack::Vector{<:AbstractLayer}) =
    all(l -> _lossless(l.ϵ) && (!hasproperty(l, :μ) || _lossless(l.μ)), stack)
_lossless(m) = all(iszero(imag(m[i, j])) for i in 1:3, j in 1:3)

"""
    box_kρ_grid(stack, ω, source_layer; box, zbar=nothing, N_phi=64, phi=nothing, lateral_dir=0.0, ρ_far=0.0,
                rtol=1e-4, atol=0.0, kr_order=7, maxevals=10_000, kρ_cap=2000.0, target_decay=25.0,
                include_interior=true, on_stall=:error, scan_poles=true, N_φ_scan=32)
        -> (kρ_nodes, kρ_weights, phi_nodes, phi_weights, info)

`box = (Δx_box, Δy_box, Δz_box)`. `zbar` (local depths of the box centre in
`source_layer`; default: the layer centre, plus both near-face positions when
they fit) are the representative source depths probed. `phi = (φ_nodes, φ_weights)`
replaces the uniform `N_phi` grid (one group of `box_kρ_grid_grouped`).
`lateral_dir` (angle θ) places the lateral-neighbour and far-lateral probes along
`(cos θ, sin θ)` instead of x̂: a φ node's lateral oscillation `e^{i kρ ρ cos(φ − θ)}` is
fastest at `φ = θ`, so a φ group needs its probes along its own φ (on the full
φ grid the φ = 0 nodes see the x̂ probes at full rate).
`include_interior=false`
drops the inside-the-slab probes (then every probe decays exponentially and the
tail bound is `target_decay/h_min`, see the module notes). `kρ_cap` (default 2000) is a
cost ceiling, not derived from the physics; it wins over the 80·kmax floor. `info` merges
`adaptive_kx_grid`'s info with the probe list, `h_min` (smallest probe distance), the
cut-off (`tail_bound`), the number of distinct kρ solves and the `features`.

Seeds. The adaptive refinement starts from `KSpaceFeatures.adaptive_seeds`: the
light lines at the grid's φ, the real-axis poles found by the scan
(`scan_poles=true`; over the grid's own φ when there are at most `N_φ_scan` of
them, e.g. one group of `box_kρ_grid_grouped`, else over `N_φ_scan` directions),
and log-spaced seeds from `kmax` to the cut-off. The band runs straight to the cut-off
(no `t/(1−t)` tail since 2026-11), so a pole far past the light lines is inside
the band from the first pass.

`on_stall`: what to do when a probe's finite bands do not reach `rtol` within
`maxevals` (real-axis poles of a lossless or nearly lossless stack, or an
under-resolved probe): `:error` (default) throws, listing the stalled probes;
`:warn` returns the unconverged grid with `adaptive_kx_grid`'s warnings
(diagnostics only).
"""
function box_kρ_grid(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                     box::NTuple{3,<:Real}, zbar=nothing, N_phi::Int=64, phi=nothing, lateral_dir::Real=0.0, ρ_far::Real=0.0,
                     rtol::Real=1e-4, atol::Real=0.0, kr_order::Int=7, maxevals::Int=10_000,
                     kρ_cap::Real=2000.0, target_decay::Real=25.0, include_interior::Bool=true,
                     on_stall::Symbol=:error, scan_poles::Bool=true, N_φ_scan::Int=32)
    on_stall in (:error, :warn) || error("box_kρ_grid: on_stall=$on_stall must be :error or :warn")
    N = length(stack)
    (1 <= source_layer < N) || error("box_kρ_grid: source_layer=$source_layer must be in 1:$(N-1)")
    Δx, Δy, Δz = Float64.(box)
    d = stack[source_layer].d
    Δz <= d || error("box_kρ_grid: Δz_box=$Δz does not fit in layer $source_layer (d=$d)")
    ωf = Float64(ω)
    is_lossless(stack) && @warn "box_kρ_grid: the stack is lossless (no Im ε or Im μ in any layer). The kρ quadrature " *
        "runs on the real axis, so guided-mode poles sit on the path and the finite bands will likely stall. " *
        "Add a small loss (Im ε ≈ 1e-6 on every layer converges for iso/uniaxial stacks)."
    φs, wφ = phi === nothing ? build_kphi_grid(N_phi) : (collect(Float64, phi[1]), collect(Float64, phi[2]))

    # representative box-centre depths (local, caller's frame), inside the layer
    zb = zbar === nothing ? unique([d / 2, Δz / 2 + 1e-3 * d, d - Δz / 2 - 1e-3 * d]) : collect(Float64, zbar)
    zb = [z for z in zb if Δz / 2 - 1e-12 <= z <= d - Δz / 2 + 1e-12]
    isempty(zb) && error("box_kρ_grid: no box-centre depth fits in the layer")
    zpos_of(z) = source_layer == 1 ? d - z : z
    cθ, sθ = cos(lateral_dir), sin(lateral_dir)

    # probes: (obs_layer, z_obs, Δx, Δy, h, tag); h = distance from the box's near face (Inf-free)
    probes = Tuple{Int,Float64,Float64,Float64,Float64,Tuple{Float64,String}}[]
    for z̄ in zb
        zp = zpos_of(z̄)
        if include_interior
            push!(probes, (source_layer, zp, 0.0, 0.0, 0.0, (z̄, "self voxel")))
            push!(probes, (source_layer, zp, Δx * cθ, Δy * sθ, 0.0, (z̄, "lateral neighbour")))
        end
        zp + Δz <= d && push!(probes, (source_layer, zp + Δz, 0.0, 0.0, Δz / 2, (z̄, "vertical neighbour +")))
        zp - Δz >= 0 && push!(probes, (source_layer, zp - Δz, 0.0, 0.0, Δz / 2, (z̄, "vertical neighbour −")))
        δ = 0.5 * min(Δz, thinnest_finite_layer(stack))
        if source_layer > 1
            dup = stack[source_layer - 1].d
            push!(probes, (source_layer - 1, source_layer - 1 == 1 ? dup - δ : dup - δ, 0.0, 0.0, zp - Δz / 2 + δ, (z̄, "layer above, near interface")))
        end
        push!(probes, (source_layer + 1, δ, 0.0, 0.0, (d - zp - Δz / 2) + δ, (z̄, "layer below, near interface")))
        if ρ_far > 0
            # just outside the slab, on a side that stays in the layer (a z_obs past the layer's
            # faces makes the integrand grow like e^{+kρ·overshoot})
            zf = zp + Δz / 2 + δ <= d ? zp + Δz / 2 + δ : zp - Δz / 2 - δ
            zf >= 0 && push!(probes, (source_layer, zf, ρ_far * cθ, ρ_far * sθ, δ, (z̄, "far lateral")))
            include_interior && push!(probes, (source_layer, zp, ρ_far * cθ, ρ_far * sθ, 0.0, (z̄, "far lateral, inside the slab")))
        end
    end
    obs_layers = sort(unique(p[1] for p in probes))

    # one-kρ-node cache per kρ, evaluated at every probe at once (threaded); only the probe values
    # are memoized (a 2D cache is ~0.5 MB at 256 φ without the rotation shortcut, and the
    # quadrature visits 10^4–10^5 kρ)
    function probe_val(c, p)
        obs, zo, px, py, _, (z̄, _) = p
        # closed_form_G takes the SOURCE depth in the caller's frame and z_obs in the obs layer's frame
        return vec(closed_form_G(c, obs, z̄, [zo], [px], [py]; box_lateral=(Δx, Δy), Δz_box=Δz))
    end
    memo = Dict{Float64,Vector{Vector{ComplexF64}}}()
    vals_at(kρ) = get!(memo, kρ) do
        c = build_cavity_transfer_2d(stack, ωf, source_layer, [kρ], [1.0], φs, wφ; obs_layers=obs_layers)
        v = Vector{Vector{ComplexF64}}(undef, length(probes))
        @threads for i in eachindex(probes)         # the cache is read-only in closed_form_G
            v[i] = probe_val(c, probes[i])
        end
        v
    end
    fs = Any[(kρ -> vals_at(kρ)[i]) for i in eachindex(probes)]

    hs = [p[5] for p in probes]
    h_min = max(minimum(h for h in hs if h > 0; init=Δz / 2), 1e-6 * thinnest_finite_layer(stack))
    kmax = maximum(branch_points(stack, ωf; φs=unique(mod.(φs, π))); init=ωf)
    # per probe, whichever cuts first: e^{−kρh} decay, or the box's algebraic envelope (alone at h = 0);
    # the probes share one grid, so it takes the largest
    Kbox = box_tail_cutoff(min(Δx, Δy, Δz), rtol)
    Kcut = tail_bound(maximum(h > 0 ? min(decay_cutoff(h; target_decay=target_decay), Kbox) : Kbox for h in hs),
                      kmax; kρ_cap=kρ_cap)
    ρ_probe = maximum(hypot(p[3], p[4]) for p in probes)
    feat = kspace_features(stack, ωf, source_layer; h_min=h_min, ρ_max=ρ_probe, dims=2, Kcut=Kcut,
                           φ_scan=(length(φs) <= N_φ_scan ? φs : nothing), N_φ_scan=N_φ_scan, scan=scan_poles)

    # grading capped at 2⁻³⁰: the general eigen path (biaxial, gyrotropic) fails within ~1e-14 of a light line
    kρ, wρ, info = adaptive_kx_grid(fs, adaptive_seeds(feat; rtol=rtol, grade=min(30, _grade_for(rtol))); kcut=Kcut, rtol=Float64(rtol), atol=Float64(atol),
                                    kr_order=kr_order, maxevals=maxevals, warn_stall=on_stall === :warn)
    info = merge(info, (probes=[(p[1], p[2], p[3], p[4], p[5], p[6][2], p[6][1]) for p in probes],
                        h_min=h_min, tail_bound=Kcut, n_kρ_solves=length(memo), features=feat))
    stalled = [i for i in eachindex(probes) if info.per_probe[i].stalled]
    if on_stall === :error && !isempty(stalled)
        list = join(["  layer $(probes[i][1]), z̄ = $(probes[i][6][1]), $(probes[i][6][2]) (error estimate " *
                     "$(round(info.per_probe[i].E; sigdigits=2)))" for i in stalled], "\n")
        error("box_kρ_grid: $(length(stalled))/$(length(probes)) probes' finite bands did not reach rtol=$rtol " *
              "within maxevals=$maxevals:\n$list\n" *
              (is_lossless(stack) ? "The stack is lossless: real-axis guided-mode poles cannot be integrated; add a small loss. " :
                                    "Likely near-real poles (low loss) or an under-resolved probe. ") *
              "Pass on_stall=:warn to get the unconverged grid anyway.")
    end
    return kρ, wρ, φs, wφ, info
end


"""
    phi_groups(stack, N_phi; classes_per_group=1) -> Vector{Vector{Int}}

Partition of the uniform φ grid's indices `1:N_phi` into groups that can share a
kρ grid. Azimuthally symmetric stack: one group (every φ has the same kρ
content). Diagonal ε and μ in every layer: the mirror classes
`{φ, −φ, π − φ, π + φ}` (the in-plane mirrors x → −x, y → −y map the stack onto
itself, so these φ have the same poles). Otherwise (in-plane off-diagonal
terms, e.g. gyrotropic): one φ per group. `classes_per_group` merges that many
neighbouring classes (in φ order) into one group: fewer, larger groups amortize
the per-solve overhead of the grid build, at the price of resolving a few
neighbouring φ's poles together.
"""
function phi_groups(stack::Vector{<:AbstractLayer}, N_phi::Int; classes_per_group::Int=1)
    _is_azimuthally_symmetric_stack(stack) && return [collect(1:N_phi)]
    merge_classes(cl) = [sort(vcat(cl[r]...)) for r in Iterators.partition(eachindex(cl), classes_per_group)]
    diag = all(l -> _offdiag_zero(l.ϵ) && (!hasproperty(l, :μ) || _offdiag_zero(l.μ)), stack)
    diag || return merge_classes([[p] for p in 1:N_phi])
    seen = falses(N_phi)
    groups = Vector{Int}[]
    for p in 0:N_phi-1                    # φ_p = 2πp/N_phi; orbit under p → −p and (N even) p → N/2 − p
        seen[p + 1] && continue
        orbit = Set([p, mod(-p, N_phi)])
        if iseven(N_phi)
            for q in collect(orbit)
                push!(orbit, mod(N_phi ÷ 2 - q, N_phi))
            end
        end
        g = sort([q + 1 for q in orbit])
        seen[g] .= true
        push!(groups, g)
    end
    return merge_classes(groups)              # classes come out in increasing min φ, i.e. φ order on [0, π/2]
end
_offdiag_zero(m) = all(iszero(m[i, j]) for i in 1:3, j in 1:3 if i != j)

"""
    box_kρ_grid_grouped(stack, ω, source_layer; N_phi=64, classes_per_group=1, kwargs...) -> (groups, info)

φ-dependent kρ grid: the uniform φ grid is split into `phi_groups(stack, N_phi)`
and `box_kρ_grid` is run on each group with its own φ nodes (`kwargs` as for
`box_kρ_grid`). `groups[g] = (kρ, wρ, φ, wφ)`; together they partition the φ
grid, so a quantity linear in the quadrature sum (the cache's `G`, an operator
block) is the sum over groups of its value on each group's product grid
(`GreenOperator.build_layer_operator_grouped`).

Why: in a biaxial layer the guided-mode poles, and a hyperbolic layer's
high-kρ Fabry–Pérot comb, sit at φ-dependent kρ. A shared kρ grid has to
resolve every φ's poles once they are narrower than their shift between φ
nodes, so its size grows with `N_phi` (35k / 51k / 76k kρ at 8 / 32 / 128 φ, biaxial
hyperbolic at δ = 1e-3); one φ needs 18–24k. An azimuthally symmetric stack
gets a single group, identical to `box_kρ_grid`.

`info`: `n_groups`, `n_nodes` (Σ kρ × φ), `nkρ` per group, `any_stalled`,
`per_group` (each group's `box_kρ_grid` info).
"""
function box_kρ_grid_grouped(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int; N_phi::Int=64,
                             classes_per_group::Int=1, kwargs...)
    φs, wφ = build_kphi_grid(N_phi)
    idxs = phi_groups(stack, N_phi; classes_per_group=classes_per_group)
    groups = Vector{NamedTuple{(:kρ, :wρ, :φ, :wφ),NTuple{4,Vector{Float64}}}}(undef, length(idxs))
    infos = Vector{Any}(undef, length(idxs))
    # groups in parallel: a small group's evaluations are too cheap to thread well inside
    try
        @threads for n in eachindex(idxs)
            idx = idxs[n]
            # lateral probes along the group's own φ (its mirror images then oscillate no faster)
            kρ, wρ, φg, wφg, info = box_kρ_grid(stack, ω, source_layer; kwargs..., phi=(φs[idx], wφ[idx]),
                                                lateral_dir=(length(idx) == N_phi ? 0.0 : φs[idx[1]]))
            groups[n] = (kρ=kρ, wρ=wρ, φ=φg, wφ=wφg)
            infos[n] = info
        end
    catch e                                  # a group's own error (e.g. a stall), not the task wrapper
        while e isa CompositeException || e isa TaskFailedException
            e = e isa CompositeException ? e.exceptions[1] : e.task.exception
        end
        throw(e)
    end
    info = (n_groups=length(groups), n_nodes=sum(length(g.kρ) * length(g.φ) for g in groups),
            nkρ=[length(g.kρ) for g in groups], any_stalled=any(i -> i.any_stalled, infos),
            per_group=infos)
    return groups, info
end


"""
    point_kx_grid(stack, ω, source_layer; ρ_max=0.0, pz=nothing,
                  z_fracs=[0.1,0.3,0.5,0.7,0.9], engine=:auto, rtol=1e-6, atol=0.0,
                  kr_order=7, maxevals=10_000, kρ_cap=nothing, kρ_cap_rel=1e5, target_decay=25.0,
                  check=false, scan_poles=true, rule=(;), h_min=nothing)
        -> (kx_nodes, wt_nodes, info)

Layer-level radial grid for a POINT source anywhere in `source_layer` of an
isotropic/uniaxial stack (the `CavityTransferCache1D` path): source-agnostic,
built once per `(stack, ω, source_layer)` and reused for every depth, moment and
lateral position.

Resolution. `pz` (default `thinnest_finite_layer(stack)/20`) is the grid's
resolution: a point source has no size of its own, its integrand decays only like
`e^{−kρ|ζ|}`, so the band is cut at `Kcut = target_decay/(pz/2)`
(`KxGrid.decay_cutoff`), at least 2·kmax and at most `kρ_cap` (cost ceiling, in the
caller's k units; default `nothing` = `kρ_cap_rel·kmax`, unit-independent; it wins over
the floor, with a warning and `info.features.capped = true` when it cuts below the
geometric cut-off). Content closer to the source than `pz/2` is not resolved. `h_min` overrides `pz/2`
(e.g. `2h` for the scattered field of a source at height `h` on its own plane,
`terms = :scattered`, the direct term added analytically: handout §2g).
With `h_min` given, the probes (adaptive engine, `check`) only use source depths at
least `h_min` from the layer's real interfaces. `ρ_max` is the largest lateral
offset the grid must resolve.

Engine (`KSpaceFeatures.choose_engine`): `:auto` takes `:fixed` when a layer is
metallic, hyperbolic or gyrotropic, `:adaptive` when every layer is dielectric.
  - `:fixed`: `fixed_kρ_rule(features; rtol, rule...)` (`rule`: keyword overrides,
    e.g. `(; order=12, grade=40)`). No probe is evaluated to build it.
  - `:adaptive`: Gauss–Kronrod converged on probe integrands, seeded with the
    features. Probes, for each representative depth `z = z_frac·d` and each of the
    canonical moments `x̂`, `ẑ` (`ŷ` needs the same kρ content as `x̂` on an
    azimuthally symmetric stack), at lateral offsets `ρ = 0` and `ρ = ρ_max`: same
    layer `pz/2` above and below the source, both layer edges, and `pz/2` into each
    neighbouring layer. Each is the point field there (3 components) through
    `closed_form_pixel` on a one-node cache (memoized per kx).

`check=true`: the probes above are evaluated on the rule and on the rule with every
panel halved; `info.check = (rel_change, per_probe)` (relative change per probe,
3-vector norm, against a floor of 1e-6 × the largest probe).

`info`: `engine`, `n_nodes`, `panels`, `any_stalled`, `tail_bound` (= `Kcut`),
`n_probes`, `features`, `check`, and with `:adaptive` the per-probe errors
(`adaptive_kx_grid`'s info).
"""
function point_kx_grid(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                       ρ_max::Real=0.0, pz::Union{Nothing,Real}=nothing,
                       z_fracs::AbstractVector{<:Real}=[0.1, 0.3, 0.5, 0.7, 0.9], engine::Symbol=:auto,
                       rtol::Real=1e-6, atol::Real=0.0, kr_order::Int=7, maxevals::Int=10_000,
                       kρ_cap::Union{Nothing,Real}=nothing, kρ_cap_rel::Real=1e5, target_decay::Real=25.0, check::Bool=false,
                       scan_poles::Bool=true, rule=(;), h_min::Union{Nothing,Real}=nothing)
    N = length(stack)
    (1 <= source_layer < N) || error("point_kx_grid: source_layer=$source_layer must be in 1:$(N-1)")
    ωf = Float64(ω)
    pzf = pz === nothing ? thinnest_finite_layer(stack) / 20 : Float64(pz)
    dist = h_min === nothing ? pzf / 2 : Float64(h_min)
    obs_layers = sort(unique(filter(l -> 1 <= l <= N, [source_layer - 1, source_layer, source_layer + 1])))
    # an explicit h_min: probe only sources at least h_min from the layer's real interfaces
    zf_probe = h_min === nothing ? z_fracs : _check_depths(stack, source_layer, z_fracs, dist) ./ stack[source_layer].d
    probes = _point_probes(stack, source_layer, zf_probe, dist, Float64(ρ_max))

    feat = kspace_features(stack, ωf, source_layer; h_min=dist, ρ_max=ρ_max, dims=1,
                           target_decay=target_decay, kρ_cap=kρ_cap, kρ_cap_rel=kρ_cap_rel, scan=scan_poles)
    eng = choose_engine(stack, engine)
    if eng === :fixed
        kx, wt, panels = fixed_kρ_rule(feat; rtol=rtol, rule...)
        info = (engine=:fixed, n_nodes=length(kx), n_panels=length(panels), panels=panels, any_stalled=false)
        refined = () -> gl_rule(halve_panels(panels); order=get(rule, :order, 8), singular=feat.singular_lines)
    else
        memo = Dict{Float64,Any}()
        cache_at(kx) = get!(memo, kx) do
            build_cavity_transfer_1d(stack, ωf, source_layer, [kx], [1.0]; obs_layers=obs_layers)
        end
        fs = Any[(kx -> Vector(closed_form_pixel(cache_at(kx), obs, z, zo, ρ, 0.0, p))) for (obs, zo, z, ρ, p) in probes]
        kx, wt, ainfo = adaptive_kx_grid(fs, adaptive_seeds(feat; rtol=rtol); kcut=feat.Kcut, rtol=Float64(rtol), atol=Float64(atol),
                                         kr_order=kr_order, maxevals=maxevals)
        info = merge(ainfo, (engine=:adaptive, n_kx_solves=length(memo)))
        refined = () -> kronrod_rule(halve_panels(ainfo.panels); kr_order=kr_order)
    end
    chk = nothing
    if check
        v = _probe_values_1d(stack, ωf, source_layer, obs_layers, probes, kx, wt)
        v2 = _probe_values_1d(stack, ωf, source_layer, obs_layers, probes, refined()...)
        chk = _rel_change(v, v2)
    end
    info = merge(info, (n_probes=length(probes), tail_bound=feat.Kcut, z_fracs=collect(z_fracs), ρ_max=ρ_max,
                        features=feat, check=chk))
    return kx, wt, info
end

# (obs_layer, z_obs, z_src, ρ, moment) for the point builders: same layer ±dist and both
# edges, dist into each neighbour, moments x̂ and ẑ, at ρ = 0 and ρ_max
function _point_probes(stack, source_layer::Int, z_fracs, dist::Float64, ρ_max::Float64)
    d = stack[source_layer].d
    ε = 1e-9 * max(d, 1.0)
    probes = Tuple{Int,Float64,Float64,Float64,Vector{ComplexF64}}[]
    for zf in z_fracs
        z = clamp(zf * d, ε, d - ε)
        zp = source_layer == 1 ? d - z : z
        pts = Tuple{Int,Float64}[]
        zp + dist <= d && push!(pts, (source_layer, zp + dist))
        zp - dist >= 0 && push!(pts, (source_layer, zp - dist))
        push!(pts, (source_layer, ε), (source_layer, d - ε))
        source_layer > 1 && push!(pts, (source_layer - 1, stack[source_layer - 1].d - dist))
        push!(pts, (source_layer + 1, min(dist, stack[source_layer + 1].d)))
        for (obs, zo) in pts, p in (ComplexF64[1, 0, 0], ComplexF64[0, 0, 1]), ρ in unique([0.0, ρ_max])
            push!(probes, (obs, zo, z, ρ, p))
        end
    end
    return probes
end

"""
    chunked_2d(f, stack, ω, source_layer, kρ, wρ, φ, wφ; obs_layers, max_nodes=1_500_000)

`Σ_chunks f(cache_chunk)`: `f(cache)` evaluated on 2D caches built over consecutive
chunks of the kρ rule (at most `max_nodes` kρ × φ nodes each) and summed. Exact
for anything linear in the quadrature sum (`closed_form_G`, with any `terms`), and
keeps the memory to one chunk's cache (~2.5 kB per node: 6M nodes do not fit in
16 GB).
"""
function chunked_2d(f, stack, ω, source_layer, kρ, wρ, φ, wφ; obs_layers, max_nodes::Int=1_500_000)
    step = max(1, max_nodes ÷ max(1, length(φ)))
    acc = nothing
    for r in Iterators.partition(eachindex(kρ), step)
        c = build_cavity_transfer_2d(stack, ω, source_layer, kρ[r], wρ[r], φ, wφ; obs_layers=obs_layers)
        v = f(c)
        acc = acc === nothing ? v : acc .+ v
        c = nothing; GC.gc(false)
    end
    return acc
end

# source depths (caller's local frame) for the probes: `z_fracs·d`, kept only where the
# source is at least `hmin` from every real interface of its layer (the bottom one for
# layer 1, whose top is the open half-space); if none is, the depth `hmin` from the
# nearest real interface. A source closer to an interface than the grid's resolution
# is not something the grid was built to resolve, so it must not be probed.
function _check_depths(stack, s::Int, z_fracs, hmin::Float64)
    d = stack[s].d
    dist_iface(z) = s == 1 ? z : min(z, d - z)          # layer 1: local z = height above its bottom interface
    zs = [clamp(zf * d, 1e-9 * d, d - 1e-9 * d) for zf in z_fracs]
    ok = [z for z in zs if dist_iface(z) >= hmin]
    isempty(ok) && push!(ok, min(hmin, d))
    return ok
end

function _probe_values_1d(stack, ω, s, obs_layers, probes, kx, wt)
    c = build_cavity_transfer_1d(stack, ω, s, kx, wt; obs_layers=obs_layers)
    return [Vector{ComplexF64}(closed_form_pixel(c, obs, z, zo, ρ, 0.0, p)) for (obs, zo, z, ρ, p) in probes]
end

# relative change v → v2 per probe (floor: 1e-6 × the largest probe) and its maximum
function _rel_change(v, v2)
    scale = maximum(norm, v2; init=0.0)
    rel = [norm(a - b) / max(norm(b), 1e-6 * scale) for (a, b) in zip(v, v2)]
    return (rel_change=maximum(rel; init=0.0), per_probe=rel)
end


"""
    point_kρ_grid_2d(stack, ω, source_layer; ρ_max, pz=nothing, z_fracs=[0.1,0.5,0.9],
                     N_phi=nothing, rtol=1e-6, kρ_cap=nothing, kρ_cap_rel=1e5, target_decay=25.0,
                     c_φ=2.5, N_φ_scan=32, check=false, scan_poles=true, rule=(;),
                     h_min=nothing, terms=:all, max_nodes=1_500_000)
        -> (kρ_nodes, kρ_weights, phi_nodes, phi_weights, info)

`(kρ, φ)` tensor-product grid for a POINT source in `source_layer` of ANY stack
(biaxial, tilted, gyrotropic: the `CavityTransferCache2D` path, which shares one kρ
rule across every φ). Handout P4.

  - kρ: `fixed_kρ_rule` on `kspace_features(...; dims=2)`: light lines at every
    scanned φ, the union over φ of the real-axis poles (merged along their tracks),
    cut-off `target_decay/h_min` (`h_min = pz/2` unless given) as in `point_kx_grid`.
  - φ: uniform trapezoid, `N_phi` nodes (default `features.N_φ`, even, so the
    `φ + π` pairs are on the grid), the largest of 64 and:
      - `c_φ·Kcut·ρ_max`: the lateral factor `e^{i kρ ρ cos(φ−ψ)}` has harmonics up
        to `|m| ≈ kρ·ρ`, which the trapezoid resolves only for `N_φ > kρ·ρ`
        (handout §2f: `c_φ` ≈ 2.4 reached 1e-6 on the MoO₃ map);
      - `1.5·log(10/rtol)/η_min`: a pole that moves with φ crosses each kρ at some φ,
        where the integrand peaks with a width `η ≈ Γ/|dq/dφ|` in φ, and the
        trapezoid converges as `e^{−N_φη}` (InSb at Ω = 1.09, ±30 µm: 8e-2 / 6e-3 at
        188 / 376 φ against 752).

No φ grouping: the cache's kρ rule must resolve every φ's poles anyway.

`check=true` evaluates the full Green's tensor (3 moments) at the probes (sources at
the depths `z_fracs·d` that are at least `h_min/2` from the layer's real interfaces,
or one at `h_min/2`; observation `h_min` above and below in the same layer, `h_min`
into each neighbour,
at lateral offsets `0`, `±ρ_max x̂`, `ρ_max ŷ`) on the rule, on the rule with every
kρ panel halved, and with `2N_φ`; `info.check = (kρ = …, φ = …)`, each
`(rel_change, per_probe)`. It costs about four field evaluations; every rule is
evaluated in kρ chunks of at most `max_nodes` (kρ × φ) nodes (`chunked_2d`), and
`2N_φ` as the mean of the `N_φ` rule and the same rule shifted by `π/N_φ`, so the
memory stays that of one `max_nodes` cache.
`terms = :scattered` (a grid for the scattered field, the direct term added
analytically): the check uses the scattered field in the source layer and also
probes the source plane itself (`Δz = 0`).
"""
function point_kρ_grid_2d(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                          ρ_max::Real, pz::Union{Nothing,Real}=nothing,
                          z_fracs::AbstractVector{<:Real}=[0.1, 0.5, 0.9], N_phi::Union{Nothing,Int}=nothing,
                          rtol::Real=1e-6, kρ_cap::Union{Nothing,Real}=nothing, kρ_cap_rel::Real=1e5, target_decay::Real=25.0,
                          c_φ::Real=2.5, N_φ_scan::Int=32, check::Bool=false, scan_poles::Bool=true, rule=(;),
                          h_min::Union{Nothing,Real}=nothing, terms::Symbol=:all, max_nodes::Int=1_500_000)
    N = length(stack)
    (1 <= source_layer < N) || error("point_kρ_grid_2d: source_layer=$source_layer must be in 1:$(N-1)")
    ωf = Float64(ω)
    pzf = pz === nothing ? thinnest_finite_layer(stack) / 20 : Float64(pz)
    dist = h_min === nothing ? pzf / 2 : Float64(h_min)
    feat = kspace_features(stack, ωf, source_layer; h_min=dist, ρ_max=ρ_max, dims=2, N_φ_scan=N_φ_scan,
                           target_decay=target_decay, kρ_cap=kρ_cap, kρ_cap_rel=kρ_cap_rel, scan=scan_poles, c_φ=c_φ, rtol=rtol)
    kρ, wρ, panels = fixed_kρ_rule(feat; rtol=rtol, rule...)
    Nφ = N_phi === nothing ? feat.N_φ : N_phi
    φs, wφ = build_kphi_grid(Nφ)

    chk = nothing
    if check
        d = stack[source_layer].d
        pts = Tuple{Int,Float64,Float64}[]                 # (obs, z_obs local, z_src local)
        for z in _check_depths(stack, source_layer, z_fracs, dist / 2)
            zp = source_layer == 1 ? d - z : z
            # z_obs in the obs layer's own frame, the source depth in the caller's (as in `_point_probes`)
            zp + dist <= d && push!(pts, (source_layer, zp + dist, z))
            zp - dist >= 0 && push!(pts, (source_layer, zp - dist, z))
            source_layer > 1 && push!(pts, (source_layer - 1, stack[source_layer - 1].d - dist, z))
            push!(pts, (source_layer + 1, min(dist, stack[source_layer + 1].d), z))
        end
        if terms === :scattered                            # the source plane itself
            for z in _check_depths(stack, source_layer, z_fracs, dist / 2)
                push!(pts, (source_layer, source_layer == 1 ? d - z : z, z))
            end
        end
        lat = ρ_max > 0 ? [(0.0, 0.0), (ρ_max, 0.0), (-ρ_max, 0.0), (0.0, ρ_max)] : [(0.0, 0.0)]
        obs_layers = sort(unique(p[1] for p in pts))
        f(c) = [vec(closed_form_G(c, o, z, [zo], [x], [y]; terms=(o == source_layer ? terms : :all)))
                for (o, zo, z) in pts for (x, y) in lat]
        φn, wn = build_kphi_grid(Nφ)
        run(kn, kw, φv) = chunked_2d(f, stack, ωf, source_layer, kn, kw, φv, wn; obs_layers=obs_layers, max_nodes=max_nodes)
        v = run(kρ, wρ, φn)
        k2, w2 = gl_rule(halve_panels(panels); order=get(rule, :order, 8), singular=feat.singular_lines)
        # 2N_φ trapezoid = mean of the N_φ rule and the same rule shifted by π/N_φ (one N_φ cache at a time)
        v2φ = 0.5 .* (v .+ run(kρ, wρ, φn .+ π / Nφ))
        chk = (kρ=_rel_change(v, run(k2, w2, φn)), φ=_rel_change(v, v2φ))
    end
    info = (engine=:fixed, n_kρ=length(kρ), N_phi=Nφ, n_nodes=length(kρ) * Nφ, n_panels=length(panels), panels=panels,
            tail_bound=feat.Kcut, features=feat, check=chk)
    return kρ, wρ, φs, wφ, info
end

end
