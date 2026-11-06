module KxGrid

export branch_points, layer_light_lines, build_kphi_grid
export thinnest_finite_layer, evanescent_umax, decay_cutoff, tail_bound
export adaptive_kx_grid, kronrod_rule, graded_panels, gl_rule, halve_panels

import ..ModeSolver: AbstractLayer, _layer_μ
using FastGaussQuadrature
using LinearAlgebra: norm, det, inv
using StaticArrays
using QuadGK: quadgk_segbuf, cachedrule, Segment


# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

_tensor3(m) = SMatrix{3,3,ComplexF64}(ntuple(n -> ComplexF64(m[(n - 1) % 3 + 1, (n - 1) ÷ 3 + 1]), 9))

"""
    layer_light_lines(layer, ω, φ) -> Vector{Float64}

In-plane wavenumbers `kρ` at which a plane wave travelling along `(cos φ, sin φ)`
in `layer` has `k_z = 0`: the layer's light lines, i.e. the square-root branch
points of its `k_z` (singular for the two semi-infinite layers, harmless kinks
for the finite ones).

With `k = kρ k̂` in the plane, Maxwell's equations give
`det(ω²ε + kρ²·K̂ μ⁻¹ K̂) = 0`, `K̂` the cross-product matrix of `k̂`. `K̂` has rank
2, so the determinant is a quadratic in `q = kρ²`; its coefficients are read off
three evaluations. Each root `q` with `Re q > 0` gives the light line
`kρ = Re √q` (the real-axis position of the branch point of a lossy medium,
`ω·Re√ε` in the isotropic case). Roots with `Re q ≤ 0` (metals, the negative axis
of a hyperbolic medium) have no light line on the real axis.

Full ε and μ tensors: uniaxial gives `ω√(ε_t μ_t)` (TE) and `ω√(ε_z μ_t)` (TM,
for isotropic μ), biaxial and tilted media depend on φ, gyrotropic off-diagonal
terms enter the determinant. The light lines are even in `k̂`, so `φ` and `φ + π`
agree.
"""
function layer_light_lines(layer::AbstractLayer, ω::Real, φ::Real)
    ε = _tensor3(layer.ϵ); μ = _tensor3(_layer_μ(layer))
    c, s = cos(φ), sin(φ)
    K = @SMatrix ComplexF64[0 0 s; 0 0 -c; -s c 0]
    M = K * inv(μ) * K
    A = Float64(ω)^2 * ε
    sc = Float64(ω)^2 * max(maximum(abs, ε) * maximum(abs, μ), eps())
    p0 = det(A); p1 = det(A + sc * M); p2 = det(A + 2sc * M)
    c2 = (p2 - 2p1 + p0) / (2sc^2); c1 = (p1 - p0) / sc - c2 * sc; c0 = p0
    roots = if abs(c2) * sc^2 <= 1e-12 * (abs(c1) * sc + abs(c0))
        abs(c1) > 0 ? ComplexF64[-c0 / c1] : ComplexF64[]
    else
        D = c1^2 - 4c2 * c0
        Δ = abs(D) <= 1e-10 * abs(c1)^2 ? zero(D) : sqrt(D)       # double root (isotropic, TE = TM)
        ComplexF64[(-c1 + Δ) / (2c2), (-c1 - Δ) / (2c2)]
    end
    # a double root (isotropic: TE = TM) splits by ~√eps under round-off: merge
    return _dedup(sort!([real(sqrt(q)) for q in roots if real(q) > 0]); rtol=1e-7)
end

"""
    branch_points(stack, ω; φs=(0.0, π/2)) -> Vector{Float64}

Sorted, de-duplicated light lines (`layer_light_lines`) of every layer, at every
in-plane direction in `φs`. For an isotropic/uniaxial stack with isotropic μ this
is `ω·Re√ε_ii` over the diagonal entries with a light line, as before; it now
also includes μ (`ω·Re√(ε μ)`), and handles biaxial, tilted and gyrotropic
tensors through the full determinant. A biaxial or tilted layer's light line
moves with φ: the default `φs` gives its values along the two axes; pass the
φ nodes of a 2D grid to get all of them.

The poles are NOT here: `KSpaceFeatures.real_axis_poles` finds them.
"""
function branch_points(stack::Vector{<:AbstractLayer}, ω::Real; φs=(0.0, π / 2))
    ks = Float64[]
    for lay in stack, φ in φs
        append!(ks, layer_light_lines(lay, ω, φ))
    end
    return _dedup(sort!(ks))
end

# merge values closer than `rtol` (relative), keeping the first of each cluster
function _dedup(ks::Vector{Float64}; rtol::Float64=1e-9)
    out = Float64[]
    for k in ks
        (isempty(out) || k - out[end] > rtol * max(abs(k), 1e-300)) && push!(out, k)
    end
    return out
end


"""
    decay_cutoff(h; target_decay=25.0) -> K

kρ at which an `e^{−kρ·h}` tail has decayed by `target_decay` e-folds:
`K = target_decay / h`, with `h` the smallest source-to-observation distance the
grid must resolve. `target_decay` is `log(1/rtol)` plus a margin for the
polynomial prefactors of the near field (`kρⁿ·e^{−kρh}`); e^{−25} ≈ 1.4e-11.
"""
decay_cutoff(h::Real; target_decay::Real=25.0) = Float64(target_decay) / Float64(h)


"""
    tail_bound(K, kmax; kρ_cap=Inf, kx_ratio_min=80.0) -> Float64

Cut-off of the finite kρ band (`adaptive_kx_grid`'s `kcut`, the fixed rule's last
break) from a geometry-driven cutoff `K` (`decay_cutoff`, `box_tail_cutoff`): at least `kx_ratio_min·kmax` (keeps the cut well past the
propagating band; 80·kmax is the fixed tail every grid used before the cutoff was
tied to the geometry), at most `kρ_cap` (cost ceiling). The ceiling wins over the
floor, so a caller's explicit `kρ_cap` (quick runs, rank studies below the pixel
band) is always honoured.
"""
tail_bound(K::Real, kmax::Real; kρ_cap::Real=Inf, kx_ratio_min::Real=80.0) =
    min(max(Float64(K), Float64(kx_ratio_min) * kmax), Float64(kρ_cap))


"""
    evanescent_umax(kmax, Leff; target_decay=25.0, kx_ratio_max=80.0)

(Kept for `mod_Dipz_SemiAnalytical.jl`'s fixed reference grid; the live grids no
longer use a cosh-substituted tail.)

Effective length scale that sets the decay. In u, for change of variables kρ = kmax * cosh(u), 
the tail decays as exp(-kmax * Leff * sinh(u)).
"""
function evanescent_umax(kmax::Float64, Leff::Float64;
                         target_decay::Float64=25.0, kx_ratio_max::Float64=80.0)
    umax_phys = asinh(target_decay / (kmax * Leff))
    umax_cap  = acosh(max(kx_ratio_max, 1.0 + 1e-10))
    return min(umax_phys, umax_cap)
end


"""
    _evanescent_band_nodes(kmax, umax, N; δ=1e-6)

The `kmax -> ∞` cosh-substituted Gauss-Legendre band itself.
"""
function _evanescent_band_nodes(kmax::Float64, umax::Float64, N::Int; δ::Float64=1e-6)
    umax <= δ && return Float64[], Float64[]
    ξe, we = gausslegendre(N)
    u  = @. (umax - δ)/2 * ξe + (umax + δ)/2
    wu = @. we * (umax - δ)/2
    return kmax .* cosh.(u), wu .* (kmax .* sinh.(u))
end


"""
    thinnest_finite_layer(stack) -> Float64

Minimum thickness among `stack`'s genuinely finite layers.
"""
function thinnest_finite_layer(stack::Vector{<:AbstractLayer})
    N = length(stack)
    N <= 1 && return Inf
    return minimum(stack[j].d for j in 1:N-1)
end


"""
    _kronrod_reference_rule(x, w, wg) -> (ξ, wt)

Expands QuadGK's own packed Gauss-Kronrod rule representation (`x`, `w`,
`wg`, as returned by `QuadGK.cachedrule` -- the half-rule-by-symmetry storage
`QuadGK.evalrule` itself unpacks inline) into an explicit, full list of
reference abscissas `ξ ∈ [-1,1]` and weights `wt` such that `Σ wt_i *
h(ξ_i) ≈ ∫_{-1}^1 h(ξ) dξ` for the KRONROD (not embedded Gauss) rule --
i.e. the exact rule `QuadGK.evalrule` uses for its own returned segment
integral `Segment.I`. The indexing below is a literal transcription of
`QuadGK.evalrule`'s accumulation loop (`adapt.jl`/`evalrule.jl`), just
collecting `(node, weight)` pairs instead of evaluating `f` at them -- see
`adaptive_kx_grid`'s own docstring for why reproducing QuadGK's EXACT rule
(not a fresh, differently-nodED rule) is the point: a fresh rule risks
completely missing sub-panel structure (a narrow resonance, say) that just
happens to fall between its own node positions even though the panel was
correctly judged converged AT QuadGK's actual sample points.
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
decomposition: places QuadGK's OWN Kronrod rule (`_kronrod_reference_rule`,
from `cachedrule(Float64, kr_order)`'s `(x,w,wg)`) on each accepted panel
`[a,b]`. See `adaptive_kx_grid`'s own docstring for why the frozen rule must
be QuadGK's actual rule, not an independently-chosen one.
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


# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    adaptive_kx_grid(fs, kbreaks; kcut, rtol=1e-6, atol=0.0, kr_order=7,
                     maxevals=10_000, warn_stall=true) -> (kx_nodes, wt_nodes, info)

h-adaptive Gauss–Kronrod rule on the finite band `[0, kcut]`, converged for every
probe integrand in `fs` (each `f(kρ)`, scalar or vector valued) and merged into
ONE node/weight rule.

`kcut` is the geometric cut-off (`decay_cutoff`/`tail_bound`): there is no
semi-infinite tail any more. Content past `kcut` has decayed by `target_decay`
e-folds by construction, and every pole below `kcut` lies inside the band, where
the refinement sees it. (Until 2026-11 the band stopped at the largest light line
and the rest went through a `kρ = kmax + t/(1−t)` tail; in hyperbolic and metallic
stacks every polariton pole landed in that tail, which stalled -- see
`HANDOUT_quadrature_hyperbolic_gyrotropic.md` §2a.)

`kbreaks` seeds the initial partition (any order, any number; values outside
`(0, kcut)` are dropped): light lines, pole and band edges, and the log-spaced
seeds that `KSpaceFeatures.kspace_features` returns. The seeds are passed to
QuadGK as its starting segments (`eval_segbuf`), so their number costs no
compilation.

Merging. Each probe is refined on its own (its own `rtol` scale: stacking probes
into one vector integrand lets a small probe hide inside a large one's norm),
then the union of every probe's accepted panel boundaries is taken and QuadGK's
OWN Kronrod rule (`_kronrod_reference_rule`) is placed on the merged partition.
The merged partition refines every probe's own, so each probe is integrated at
least as accurately as by its own run, and there is a single rule (no double
counting). The exact Kronrod rule, not a fresh Gauss–Legendre one, is required:
a panel judged converged at QuadGK's sample points can hide a narrow peak
between the nodes of a different rule.

A probe that does not reach `(rtol, atol)` within `maxevals` is flagged stalled
(`info.per_probe[i].stalled`; warned unless `warn_stall=false`); its panels are
kept as far as it got.

`info`: `n_panels`, `n_nodes`, `per_probe` (`(E, stalled)` per probe),
`any_stalled`, `min_panel_width`, `kcut`, `panels` (the merged partition).
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

"""
    kronrod_rule(panels; kr_order=7) -> (nodes, weights)

QuadGK's Kronrod rule of order `kr_order` on each panel (the rule
`adaptive_kx_grid` returns, rebuilt from `info.panels`; with
`halve_panels(info.panels)`, the self-check's refined rule).
"""
function kronrod_rule(panels::AbstractVector{<:NTuple{2,Real}}; kr_order::Int=7)
    x, w, wg = cachedrule(Float64, kr_order)
    return _panels_to_nodes([(a=Float64(a), b=Float64(b)) for (a, b) in panels], x, w, wg)
end


"""
    graded_panels(breaks, width; α=0.5, max_panels=10^7) -> Vector{NTuple{2,Float64}}

Partition of `[breaks[1], breaks[end]]` into panels that end on every break and
whose width never exceeds the local target `width(k)` anywhere inside them.

`width` must be positive and Lipschitz with constant ≤ `α` (true for the width
functions built by `KSpaceFeatures`: every term is either constant, `r·k` with
`r ≤ α`, or `max(w₀, α·dist(k, feature))`). Then a step `s = width(a)/(1+α)` from
`a` satisfies `s ≤ width(k)` on all of `[a, a+s]`, so marching from left to right
grades geometrically onto features ahead as well as behind.
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

Composite Gauss–Legendre rule, `order` nodes on each panel. A panel with an end
exactly on a value in `singular` (a `k_z` branch point `L`, where the integrand
behaves like `g(kρ)/√|kρ − L|` plus a smooth part) is integrated in `u` with
`kρ = L ∓ u²`, `dkρ = 2u du`: both parts become smooth in `u`, so Gauss–Legendre
converges geometrically on that panel instead of with the `√(width)` error of a
panel ending on the singularity. The nodes stay at a relative distance
`≳ 1e-4 × (panel width)` from `L` -- the general Berreman `eigen` path returns
garbage within ~1e-14 of a light line, which deep geometric grading would hit.
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

"`panels` with every panel split in two (the self-check's refined rule)."
halve_panels(panels::AbstractVector{<:NTuple{2,Real}}) =
    [p for (a, b) in panels for p in ((Float64(a), (a + b) / 2), ((a + b) / 2, Float64(b)))]

function build_kphi_grid(N_phi::Int)
    phi_nodes   = [2π * p / N_phi for p in 0:(N_phi - 1)]
    phi_weights = fill(2π / N_phi, N_phi)
    return phi_nodes, phi_weights
end


end
