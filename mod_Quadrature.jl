module Quadrature
# TODO Rename QuadratureRules, without the mod_
export choose_engine, fixed_kρ_rule, adaptive_seeds, adaptive_kx_grid
export kronrod_rule, graded_panels, gl_rule, halve_panels, build_kphi_grid

import ..Materials: AbstractLayer
import ..Features: material_class
using FastGaussQuadrature
using LinearAlgebra: norm
using QuadGK: quadgk_segbuf, cachedrule, Segment

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    _kronrod_reference_rule(x, w, wg) -> (ξ, wt)

Expands QuadGK's own packed Gauss-Kronrod rule representation into 
an explicit, full list of reference abscissas `ξ ∈ [-1,1]` and 
weights `wt` such that `Σ wt_i * h(ξ_i) ≈ ∫_{-1}^1 h(ξ) dξ` for 
the Kronrod rule.
"""
function _kronrod_reference_rule(x::Vector{Float64}, w::Vector{Float64}, wg::Vector{Float64})
    n1 = 1 - (length(x) & 1)
    ξ  = Float64[]
    wt = Float64[]
    if n1 == 0
        push!(ξ, 0.0); push!(wt, w[end])
    else
        push!(ξ, 0.0);         push!(wt, w[end])
        push!(ξ, x[end-1]);    push!(wt, w[end-1])
        push!(ξ, -x[end-1]);   push!(wt, w[end-1])
    end
    for i in 1:(length(wg) - n1)
        push!(ξ, x[2i]);    push!(wt, w[2i])
        push!(ξ, -x[2i]);   push!(wt, w[2i])
        push!(ξ, x[2i-1]);  push!(wt, w[2i-1])
        push!(ξ, -x[2i-1]); push!(wt, w[2i-1])
    end
    return ξ, wt
end

"""
    _panels_to_nodes(segs, x, w, wg) -> (nodes, weights)

Materializes an explicit composite quadrature rule from a QuadGK segment
decomposition.
"""
function _panels_to_nodes(segs, x::Vector{Float64}, w::Vector{Float64}, wg::Vector{Float64})
    ξ, wt = _kronrod_reference_rule(x, w, wg)
    m = length(ξ)
    n = length(segs)
    nodes = Vector{Float64}(undef, n * m)
    wts   = Vector{Float64}(undef, n * m)
    idx = 1
    for s in segs
        a, b = Float64(s.a), Float64(s.b)
        h = 0.5 * (b - a)
        c = 0.5 * (a + b)
        for i in 1:m
            nodes[idx] = c + h * ξ[i]
            wts[idx]   = h * wt[i]
            idx += 1
        end
    end
    return nodes, wts
end

"""
    _grade_for(rtol)

Grading depth from rtol, only used in the adaptive engine. 
"""
_grade_for(rtol) = min(50, ceil(Int, 2 * log2(10 / Float64(rtol))))


# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

# Engine choice and shared functions ──────────────────────────

"""
    choose_engine(stack, engine=:auto) -> :fixed or :adaptive

`:auto`: `:adaptive` (seeded Gauss-Kronrod) when every layer is `:dielectric`,
`:fixed` (feature-driven Gauss-Legendre) as soon as one layer is metallic,
hyperbolic or gyrotropic.
"""
function choose_engine(stack::Vector{<:AbstractLayer}, engine::Symbol=:auto)
    engine in (:auto, :fixed, :adaptive) || error("engine must be :auto, :fixed or :adaptive (got :$engine)")
    engine !== :auto && return engine
    return all(l -> material_class(l) === :dielectric, stack) ? :adaptive : :fixed
end

"""
    build_kphi_grid(N_phi::Int) -> (phi_nodes, phi_weights)

Returns the uniform φ grid. 
"""    
function build_kphi_grid(N_phi::Int)
    phi_nodes   = [2π * p / N_phi for p in 0:(N_phi - 1)]
    phi_weights = fill(2π / N_phi, N_phi)
    return phi_nodes, phi_weights
end

"""
    halve_panels(panels::AbstractVector{<:NTuple{2,Real}})

`panels` with every panel split in two (the self-check's refined rule).
"""
halve_panels(panels::AbstractVector{<:NTuple{2,Real}}) =
    [p for (a, b) in panels for p in ((Float64(a), (a + b) / 2), ((a + b) / 2, Float64(b)))]

# Adaptive engine ──────────────────────────

"""
    kronrod_rule(panels; kr_order=7) -> (nodes, weights)

QuadGK's Kronrod rule of order `kr_order` on each panel. 
"""
function kronrod_rule(panels::AbstractVector{<:NTuple{2,Real}}; kr_order::Int=7)
    x, w, wg = cachedrule(Float64, kr_order)
    return _panels_to_nodes([(a=Float64(a), b=Float64(b)) for (a, b) in panels], x, w, wg)
end

"""
    adaptive_seeds(feat) -> Vector{Float64}

Returns a sorted vector of breakpoints between `[0, cutoff]`. It's built from
the features `feat` of the integrand and give the delimination of the 
propagation bands and the evanescent region. Typically inputed as a function 
in adaptive_kx_grid, served as the kbreaks input. 
"""
function adaptive_seeds(feat; rtol::Real=1e-6, n_Γ::Real=4.0, grade::Int=_grade_for(rtol))
    s = copy(feat.light_lines)
    for L in feat.singular_lines, j in 1:grade, sg in (-1, 1)
        push!(s, L * (1 + sg * 2.0^-j))
    end
    for c in feat.clusters
        append!(s, (c.a - n_Γ * c.Γ, c.a, c.b, c.b + n_Γ * c.Γ))
    end
    k = 2 * feat.kmax
    while k < feat.Kcut; push!(s, k); k *= 2; end
    return sort!(unique!([x for x in s if 0 < x < feat.Kcut]))
end

"""
    adaptive_kx_grid(fs, kbreaks; kcut, maxevals=10_000) -> (kx_nodes, wt_nodes, info)

h-adaptive Gauss-Kronrod rule on the finite band `[0, kcut]`, converged for every
probe in `fs`, and merged into one node/weight rule. Computed from QuadGK, with
in-house functions for panel recovery. 

# Arguments
- `fs`: list of probe integrand, see documentation for details. 
- `kbreaks`: split of the domain by light-lines and poles of R.
- `kcut`: decay cutoff from either `tail_bound` or set by geometry. 
- `maxevals`: maximum number of evaluation after which the integrand is stalled. 
  Default returns a warning in such cases. 

# Returns
- `(kx_nodes, wt_nodes, info)`: `k` nodes with their associated weight and integration
  infos.
"""
function adaptive_kx_grid(fs::Vector, kbreaks::AbstractVector{<:Real}; kcut::Real,
                          rtol::Float64=1e-6, atol::Float64=0.0,
                          kr_order::Int=7, maxevals::Int=10_000, warn_stall::Bool=true)
    isempty(fs) && error("adaptive_kx_grid: fs cannot be empty")
    kc = Float64(kcut)
    kc > 0 || error("adaptive_kx_grid: kcut must be positive (got $kcut)")
    x, w, wg = cachedrule(Float64, kr_order)

    pts = sort!(unique!(vcat(0.0, [Float64(k) for k in kbreaks if 0 < k < kc], kc)))
    seeds = [Segment(pts[i], pts[i+1], 0.0, 0.0) for i in 1:length(pts)-1]

    breaks = Set{Float64}(pts)
    per_probe = Vector{NamedTuple}(undef, length(fs))
    for (i, f) in enumerate(fs)
        I, E, segs = quadgk_segbuf(f, 0.0, kc; eval_segbuf=seeds, rtol=rtol, atol=atol,
                                   order=kr_order, maxevals=maxevals)
        stalled = E > max(atol, rtol * norm(I))
        stalled && warn_stall && @warn "adaptive_kx_grid: probe $i did not reach (rtol=$rtol, atol=$atol) within " *
            "maxevals=$maxevals (error estimate $E); its panels are kept as far as it got."
        for s in segs
            push!(breaks, Float64(s.a)); push!(breaks, Float64(s.b))
        end
        per_probe[i] = (E=E, stalled=stalled)
    end
    bp = sort!(collect(breaks))
    segs = [(a=bp[i], b=bp[i+1]) for i in 1:length(bp)-1]
    kx_nodes, wt_nodes = _panels_to_nodes(segs, x, w, wg)
    info = (n_panels=length(segs), n_nodes=length(kx_nodes), per_probe=per_probe,
            any_stalled=any(p -> p.stalled, per_probe),
            min_panel_width=isempty(segs) ? Inf : minimum(s.b - s.a for s in segs), kcut=kc,
            panels=[(s.a, s.b) for s in segs])
    return kx_nodes, wt_nodes, info
end

# Fixed engine ──────────────────────────

"""
    graded_panels(breaks, width) -> Vector{NTuple{2,Float64}}

Partition of `[breaks[1], breaks[end]]` into panels that end on every break and
whose width never exceeds the local target `width(k)` anywhere inside them.
"""
function graded_panels(breaks::AbstractVector{<:Real}, width; α::Real=0.5, max_panels::Int=10^7)
    bs = sort!(unique!(collect(Float64, breaks)))
    panels = NTuple{2,Float64}[]
    for i in 1:length(bs)-1
        a, b = bs[i], bs[i+1]
        x = a
        while x < b
            s = width(x) / (1 + α)
            s > 0 || error("graded_panels: non-positive width $(s) at k=$x")
            y = x + s >= b - 1e-14 * abs(b) ? b : x + s
            push!(panels, (x, y))
            length(panels) > max_panels && error("graded_panels: more than $max_panels panels; check the width function")
            x = y
        end
    end
    return panels
end

"""
    gl_rule(panels; order=8, singular=Float64[]) -> (nodes, weights)

Composite Gauss-Legendre rule, `order` nodes on each panel. A panel with an end
exactly on a value in `singular` is integrated in `u` with `kρ = L ∓ u²`, 
`dkρ = 2u du`. 
"""
function gl_rule(panels::AbstractVector{<:NTuple{2,Real}}; order::Int=8, singular=Float64[])
    ξ, wξ = gausslegendre(order)
    S = Set{Float64}(Float64.(singular))
    n = length(panels)
    nodes = Vector{Float64}(undef, n * order); wts = similar(nodes)
    idx = 1
    for (a, b) in panels
        a, b = Float64(a), Float64(b)
        L, sg = a in S ? (a, 1.0) : (b in S ? (b, -1.0) : (NaN, 0.0))
        if sg == 0
            h = 0.5 * (b - a); c = 0.5 * (a + b)
            for j in 1:order
                nodes[idx] = c + h * ξ[j]; wts[idx] = h * wξ[j]; idx += 1
            end
        else                                   # kρ = L + sg·u², u ∈ [0, √(b − a)]
            U = sqrt(b - a)
            for j in 1:order
                u = 0.5U * (1 + ξ[j])
                nodes[idx] = L + sg * u^2; wts[idx] = 0.5U * wξ[j] * 2u; idx += 1
            end
        end
    end
    return nodes, wts
end

"""
    fixed_kρ_rule(feat; order=8, rtol=1e-6, α=0.5, grade=12, r_rel=0.05,
                  c_h=4.0, c_ρ=4.0, pole_safety=0.5) -> (nodes, weights, panels)

Composite Gauss-Legendre rule on `[0, cutoff]`, breaks at every light line. 
Panels from `graded_panels` with the local target width based on the smallest 
feature coming from geometry, φ-singularities, light-lines or poles of R.
Halving every panel is the convergence check.

# Arguments
- `feat`: features of the integrand, geometry, light-lines, poles of R. 

# Returns
- `(nodes, wts, panels)`: fixed nodes (kρ, φ), their weight and the quadrature
  panels. 
"""
function fixed_kρ_rule(feat; order::Int=8, rtol::Real=1e-6, α::Real=0.5, grade::Int=12,
                       r_rel::Real=0.05, c_h::Real=4.0, c_ρ::Real=4.0, pole_safety::Real=0.5)
    r_rel <= α || error("fixed_kρ_rule: r_rel must be ≤ α")
    ρB = (1 / Float64(rtol))^(1 / (2order)); εB = (ρB - 1 / ρB) / 2
    wΓ(Γ) = pole_safety * 2Γ / εB
    w_flat = min(c_h / feat.h_min, feat.ρ_max > 0 ? c_ρ / feat.ρ_max : Inf)
    w_floor = feat.kmax / 40
    sing = feat.singular_lines; cl = feat.clusters
    function width(k)
        w = min(w_flat, max(r_rel * k, w_floor))
        for L in sing
            w = min(w, max(L * 2.0^-grade, α * abs(k - L)))
        end
        for c in cl
            dist = k < c.a ? c.a - k : (k > c.b ? k - c.b : 0.0)
            w = min(w, max(wΓ(c.Γ), α * dist))
        end
        return w
    end
    breaks = vcat(0.0, [L for L in feat.light_lines if L < feat.Kcut], feat.Kcut)
    panels = graded_panels(breaks, width; α=α)
    nodes, wts = gl_rule(panels; order=order, singular=feat.singular_lines)
    return nodes, wts, panels
end

end
