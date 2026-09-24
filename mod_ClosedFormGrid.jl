"""
mod_ClosedFormGrid.jl -- adaptive reciprocal-space quadrature for the closed-form
caches (`PLAN_closed_form_point_source_rewrite.md` §4.2): `point_kx_grid` (1D, point
source, isotropic/uniaxial) and `box_kρ_grid` (2D, box-aware). Both converge probe
integrands evaluated through the SAME closed-form synthesis the field uses, with
`KxGrid.adaptive_kx_grid` (h-adaptive Gauss–Kronrod per probe, merged panel
boundaries). The 1D and 2D grids are independent quadratures -- build them
separately.

── `box_kρ_grid` ──

The kρ axis of the 2D `(kρ,φ)` grid is converged against probe integrands
built from the BOX response -- `S(kt)·D·s_point` for observation points outside
the source slab, plus the §5.3 interior term (face ramps, particular solution,
local Ez) inside it -- instead of the point response. The φ axis stays the fixed uniform `build_kphi_grid`.

Each probe is a function of kρ alone: the full box Green's tensor at one
probe point, evaluated through the SAME synthesis the field uses
(`closed_form_G` on a one-kρ-node `CavityTransferCache2D`, φ-summed on the
fixed φ grid), so what is converged is exactly what is later integrated. The
per-kρ cache is memoized and shared by every probe. `KxGrid.adaptive_kx_grid`
(kept as-is) does the h-adaptive Gauss–Kronrod refinement, per probe, and
merges the panel boundaries.

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
    tail panels to resolve `e^{i kρ ρ cosφ}` out to the tail bound.

Tail. Outside the slab the box integrand decays like `e^{−kρ·h}` with `h` the
distance to the box's NEAR face (not to its centre), so `Leff = h` (plus the
interface distance for reflected terms) is what the fallback evanescent band
uses. Inside the slab there is no exponential factor: the integrand decays
only algebraically, with the a-priori envelope `tail(K) ~ log(KΔ)/(KΔ)`,
`Δ = min(Δx_box, Δy_box, Δz_box)`. When such a probe is present, the upper kρ
bound is set from the tolerance, `log(KΔ)/(KΔ) ≤ rtol`, capped by `kρ_cap`
(instead of the point grid's fixed `kx_ratio_max = 80`). The φ grid must resolve
`kρ·ρ` up to that bound for the off-centre probes -- `N_phi` is the caller's.
The optional quasi-static (Newell) subtraction of §4.2 is not implemented.
"""
module ClosedFormGrid

export point_kx_grid, box_kρ_grid, box_tail_cutoff

import ..ModeSolver: AbstractLayer
import ..KxGrid: adaptive_kx_grid, branch_points, build_kphi_grid, thinnest_finite_layer
import ..CavityTransfer: build_cavity_transfer_1d, build_cavity_transfer_2d
import ..ClosedFormField: closed_form_G, closed_form_pixel

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
    box_kρ_grid(stack, ω, source_layer; box, zbar=nothing, N_phi=64, ρ_far=0.0,
                rtol=1e-4, atol=0.0, kr_order=7, maxevals=10_000, kρ_cap=2000.0,
                include_interior=true)
        -> (kρ_nodes, kρ_weights, phi_nodes, phi_weights, info)

`box = (Δx_box, Δy_box, Δz_box)`. `zbar` (local depths of the box centre in
`source_layer`; default: the layer centre, plus both near-face positions when
they fit) are the representative source depths probed. `include_interior=false`
drops the inside-the-slab probes (then every probe decays exponentially and the
point-grid `kx_ratio_max = 80` tail cap is used). `info` merges
`adaptive_kx_grid`'s info with the probe list, `Leff`, the tail bound and the
number of distinct kρ solves.
"""
function box_kρ_grid(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                     box::NTuple{3,<:Real}, zbar=nothing, N_phi::Int=64, ρ_far::Real=0.0,
                     rtol::Real=1e-4, atol::Real=0.0, kr_order::Int=7, maxevals::Int=10_000,
                     kρ_cap::Real=2000.0, include_interior::Bool=true)
    N = length(stack)
    (1 <= source_layer < N) || error("box_kρ_grid: source_layer=$source_layer must be in 1:$(N-1)")
    Δx, Δy, Δz = Float64.(box)
    d = stack[source_layer].d
    Δz <= d || error("box_kρ_grid: Δz_box=$Δz does not fit in layer $source_layer (d=$d)")
    ωf = Float64(ω)
    φs, wφ = build_kphi_grid(N_phi)

    # representative box-centre depths (local, caller's frame), inside the layer
    zb = zbar === nothing ? unique([d / 2, Δz / 2 + 1e-3 * d, d - Δz / 2 - 1e-3 * d]) : collect(Float64, zbar)
    zb = [z for z in zb if Δz / 2 - 1e-12 <= z <= d - Δz / 2 + 1e-12]
    isempty(zb) && error("box_kρ_grid: no box-centre depth fits in the layer")
    zpos_of(z) = source_layer == 1 ? d - z : z

    # probes: (obs_layer, z_obs, Δx, Δy, h, tag); h = distance from the box's near face (Inf-free)
    probes = Tuple{Int,Float64,Float64,Float64,Float64,Tuple{Float64,String}}[]
    for z̄ in zb
        zp = zpos_of(z̄)
        if include_interior
            push!(probes, (source_layer, zp, 0.0, 0.0, 0.0, (z̄, "self voxel")))
            push!(probes, (source_layer, zp, Δx, 0.0, 0.0, (z̄, "lateral neighbour")))
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
            push!(probes, (source_layer, zp + Δz / 2 + δ, Float64(ρ_far), 0.0, δ, (z̄, "far lateral")))
            include_interior && push!(probes, (source_layer, zp, Float64(ρ_far), 0.0, 0.0, (z̄, "far lateral, inside the slab")))
        end
    end
    obs_layers = sort(unique(p[1] for p in probes))

    # one-kρ-node cache per kρ, shared by every probe
    memo = Dict{Float64,Any}()
    cache_at(kρ) = get!(memo, kρ) do
        build_cavity_transfer_2d(stack, ωf, source_layer, [kρ], [1.0], φs, wφ; obs_layers=obs_layers)
    end
    function probe_fn(p)
        obs, zo, px, py, _, (z̄, _) = p
        # closed_form_G takes the SOURCE depth in the caller's frame and z_obs in the obs layer's frame
        return kρ -> vec(closed_form_G(cache_at(kρ), obs, z̄, [zo], [px], [py]; box_lateral=(Δx, Δy), Δz_box=Δz))
    end
    fs = Any[probe_fn(p) for p in probes]

    kbreaks = branch_points(stack, ωf)
    kmax = kbreaks[end]
    hs = [p[5] for p in probes]
    has_interior = any(==(0.0), hs)
    Leff = max(minimum(h for h in hs if h > 0; init=Δz / 2), 1e-6 * thinnest_finite_layer(stack))
    Kcut = has_interior ? min(box_tail_cutoff(min(Δx, Δy, Δz), rtol), Float64(kρ_cap)) : 80.0 * kmax
    kx_ratio_max = max(Kcut / kmax, 1.0 + 1e-6)

    kρ, wρ, info = adaptive_kx_grid(fs, kbreaks; rtol=Float64(rtol), atol=Float64(atol), kr_order=kr_order,
                                    maxevals=maxevals, Leff=Leff, kx_ratio_max=kx_ratio_max)
    info = merge(info, (probes=[(p[1], p[2], p[3], p[4], p[5], p[6][2], p[6][1]) for p in probes],
                        Leff=Leff, tail_bound=Kcut, n_kρ_solves=length(memo)))
    return kρ, wρ, φs, wφ, info
end


"""
    point_kx_grid(stack, ω, source_layer; ρ_max=0.0, pz=nothing,
                  z_fracs=[0.1,0.3,0.5,0.7,0.9], rtol=1e-6, atol=0.0,
                  kr_order=7, maxevals=10_000, kx_ratio_max=80.0)
        -> (kx_nodes, wt_nodes, info)

Layer-level adaptive radial grid for a POINT source anywhere in `source_layer`
of an isotropic/uniaxial stack (the `CavityTransferCache1D` path): source-agnostic,
built once per `(stack, ω, source_layer)` and reused for every depth, moment and
lateral position.

Probes, for each representative source depth `z = z_frac·d` (clamped inside the
layer) and each of the two canonical moments `x̂`, `ẑ` (on an azimuthally
symmetric stack `ŷ` needs the same kρ content as `x̂`), at lateral offsets
`ρ = 0` and `ρ = ρ_max`:
  - same layer, `pz/2` above and below the source (the near-source content);
  - same layer, at both layer edges;
  - across each interface, `pz/2` into the neighbouring layer.
Each probe is `kx ↦` the point field at that point (3 components), evaluated with
`closed_form_pixel` on a one-node 1D cache (memoized per kx, shared by every probe).
`pz` (default: `thinnest_finite_layer(stack)/20`) sets the near-source probe
distance and hence `Leff = max(min(pz/2, source-to-interface distance), floor)` for
the tail fallback. The tail is capped at `kx_ratio_max·kmax` as in
`adaptive_kx_grid`: a point source's integrand decays like `e^{−kρ|ζ|}`, so this
cap is what fixes the grid's resolution at the smallest probed `|ζ|`.
"""
function point_kx_grid(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                       ρ_max::Real=0.0, pz::Union{Nothing,Real}=nothing,
                       z_fracs::AbstractVector{<:Real}=[0.1, 0.3, 0.5, 0.7, 0.9],
                       rtol::Real=1e-6, atol::Real=0.0, kr_order::Int=7, maxevals::Int=10_000,
                       kx_ratio_max::Real=80.0)
    N = length(stack)
    (1 <= source_layer < N) || error("point_kx_grid: source_layer=$source_layer must be in 1:$(N-1)")
    ωf = Float64(ω)
    d = stack[source_layer].d
    pzf = pz === nothing ? thinnest_finite_layer(stack) / 20 : Float64(pz)
    dist = pzf / 2
    obs_layers = sort(unique(filter(l -> 1 <= l <= N, [source_layer - 1, source_layer, source_layer + 1])))

    memo = Dict{Float64,Any}()
    cache_at(kx) = get!(memo, kx) do
        build_cavity_transfer_1d(stack, ωf, source_layer, [kx], [1.0]; obs_layers=obs_layers)
    end

    ε = 1e-9 * max(d, 1.0)
    probes = Tuple{Int,Float64,Float64,Float64,Vector{ComplexF64}}[]   # (obs, z_obs, z_src, ρ, p)
    min_h = Inf
    for zf in z_fracs
        z = clamp(zf * d, ε, d - ε)
        zp = source_layer == 1 ? d - z : z
        pts = Tuple{Int,Float64}[]
        zp + dist <= d && push!(pts, (source_layer, zp + dist))
        zp - dist >= 0 && push!(pts, (source_layer, zp - dist))
        push!(pts, (source_layer, ε), (source_layer, d - ε))
        source_layer > 1 && push!(pts, (source_layer - 1, stack[source_layer - 1].d - dist))
        push!(pts, (source_layer + 1, min(dist, stack[source_layer + 1].d)))
        min_h = min(min_h, zp, d - zp)
        for (obs, zo) in pts, p in (ComplexF64[1, 0, 0], ComplexF64[0, 0, 1]), ρ in unique([0.0, Float64(ρ_max)])
            push!(probes, (obs, zo, z, ρ, p))
        end
    end
    fs = Any[(kx -> Vector(closed_form_pixel(cache_at(kx), obs, z, zo, ρ, 0.0, p))) for (obs, zo, z, ρ, p) in probes]

    kbreaks = branch_points(stack, ωf)
    Leff = max(min(dist, min_h), 1e-6 * thinnest_finite_layer(stack))
    kx, wt, info = adaptive_kx_grid(fs, kbreaks; rtol=Float64(rtol), atol=Float64(atol), kr_order=kr_order,
                                    maxevals=maxevals, Leff=Leff, kx_ratio_max=Float64(kx_ratio_max))
    info = merge(info, (n_probes=length(probes), Leff=Leff, n_kx_solves=length(memo), z_fracs=collect(z_fracs), ρ_max=ρ_max))
    return kx, wt, info
end

end
