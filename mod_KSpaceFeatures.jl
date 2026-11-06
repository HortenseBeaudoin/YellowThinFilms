"""
mod_KSpaceFeatures.jl -- where the reciprocal-space integrand has structure, found
BEFORE any quadrature is built (`HANDOUT_quadrature_hyperbolic_gyrotropic.md`, P1–P4).

The grid builders (`mod_ClosedFormGrid.jl`) used to know only the light lines and
the largest of them (`kmax`); the polariton poles of hyperbolic, metallic and
gyrotropic layers sit far past `kmax`, have a width set by the loss, and move with
φ. `kspace_features` collects everything a rule needs, once per
`(stack, ω, source_layer, geometry)`:

  - light lines of every layer (`KxGrid.layer_light_lines`: full ε and μ
    tensors), the two semi-infinite layers' ones flagged as singular (their `k_z`
    branch point, `1/k_z` integrably singular);
  - real-axis poles `(q, Γ, φ)` from a scan of the source layer's cavity response
    (`real_axis_poles`), merged into clusters;
  - the cut-off `Kcut = target_decay/h_min` from the geometry (`tail_bound`, at
    least 2·kmax, at most `kρ_cap`): the poles are inside the finite band, there is
    no tail;
  - in 2D, `N_φ` from the geometry (`c_φ·Kcut·ρ_max`).

Two engines use the same features:
  - `fixed_kρ_rule`: composite Gauss–Legendre on graded panels -- the recipe of
    the handout's §3, from the features instead of by hand. Cannot stall; its
    convergence check is the halved rule (`KxGrid.halve_panels`).
  - `KxGrid.adaptive_kx_grid`, seeded with `adaptive_seeds(features)`.

`choose_engine` picks between them from the materials (`material_class`).
"""
module KSpaceFeatures

export material_class, choose_engine, real_axis_poles, kspace_features, fixed_kρ_rule, adaptive_seeds

import ..ModeSolver: AbstractLayer, Modes_Stack, _layer_μ, _is_azimuthally_symmetric_stack
import ..ScatteringMatrix: S_all_Matrix, S_all_ingredients, _Redheffer
import ..FieldSolver: _SProp
import ..KxGrid: layer_light_lines, decay_cutoff, tail_bound, graded_panels, gl_rule, _dedup, _tensor3
using LinearAlgebra
using StaticArrays
using Base.Threads


# ─────────────────────────────────────────────────────────────────────────────
#  Material classes and engine choice
# ─────────────────────────────────────────────────────────────────────────────

"""
    material_class(layer) -> Symbol

What kind of medium `layer` is at the frequency its ε and μ were evaluated at
(a value property, not a type: hBN is hyperbolic inside its reststrahlen bands
and an ordinary dielectric outside them).

  - `:gyrotropic`  -- ε or μ not symmetric (`ε ≠ εᵀ`: magneto-optic, Voigt/Faraday);
  - otherwise, from the signs of the eigenvalues of `Re ε` and `Re μ` (the diagonal
    entries for a diagonal tensor, the principal values for a tilted one):
      `:dielectric` all positive, `:metallic` all of ε negative (μ positive),
      `:hyperbolic` anything else (mixed signs, or a negative μ).
"""
function material_class(layer::AbstractLayer)
    ε = _tensor3(layer.ϵ); μ = _tensor3(_layer_μ(layer))
    (ε != transpose(ε) || μ != transpose(μ)) && return :gyrotropic
    eε = eigvals(Symmetric(Matrix(real.(ε)))); eμ = eigvals(Symmetric(Matrix(real.(μ))))
    all(>(0), eε) && all(>(0), eμ) && return :dielectric
    all(<(0), eε) && all(>(0), eμ) && return :metallic
    return :hyperbolic
end

"""
    choose_engine(stack, engine=:auto) -> :fixed or :adaptive

`:auto`: `:adaptive` (seeded Gauss–Kronrod) when every layer is `:dielectric`,
`:fixed` (feature-driven Gauss–Legendre) as soon as one layer is metallic,
hyperbolic or gyrotropic -- the stacks whose poles are narrow, far past the light
lines, or φ-dependent. `:fixed` / `:adaptive` are returned as given.
"""
function choose_engine(stack::Vector{<:AbstractLayer}, engine::Symbol=:auto)
    engine in (:auto, :fixed, :adaptive) || error("engine must be :auto, :fixed or :adaptive (got :$engine)")
    engine !== :auto && return engine
    return all(l -> material_class(l) === :dielectric, stack) ? :adaptive : :fixed
end


# ─────────────────────────────────────────────────────────────────────────────
#  Pole scan
# ─────────────────────────────────────────────────────────────────────────────

function _σmax(A::SMatrix{2,2})
    G = A' * A
    t = real(G[1, 1] + G[2, 2]); dt = real(G[1, 1] * G[2, 2] - G[1, 2] * G[2, 1])
    return sqrt(max((t + sqrt(max(t^2 - 4dt, 0.0))) / 2, 0.0))
end

"""
    _cavity_response(stack, ω, s, kρ, φ) -> Float64

A scalar that peaks at the real-axis poles of the field of a source in layer `s`:
the largest singular value among the layer's edge reflection matrices `RL`
(stack above, seen from the top edge) and `RR` (stack below, seen from the bottom
edge) and, for a finite layer, the cavity resummation `(I − ΦF·RL·ΦB·RR)⁻¹`. For a
semi-infinite source layer that is the stack's reflection seen from it (`r_pp`,
`r_ss` and their mixing); inside the stack, the global modes (zeros of the cavity
denominator) and the modes bound to either edge. 0 where the solve fails.

`RR` is composed directly at the interface (`Sint[s] ⋆ SRight[s+1]`), not
`FieldSolver._R_cavity`'s, which walks through the layer and back
(`e^{−kz d}·e^{+kz d}`) and underflows to 0 once `Im kz·d ≳ 370` (the 300 nm Si
cover of the ZrS₂ case, past kρ ≈ 1.24 nm⁻¹).
"""
function _cavity_response(stack, ω, s::Int, kρ::Float64, φ::Float64)
    try
        modes = Modes_Stack(stack, ω, kρ, φ)
        Sint, Sprop = S_all_ingredients(stack, modes)
        Sall = S_all_Matrix(stack, modes, Sint, Sprop)
        RR = SMatrix{2,2,ComplexF64}(_Redheffer(Sint[s], Sall[s+1].SRight)[SVector(3, 4), SOneTo(2)])
        RL = SMatrix{2,2,ComplexF64}(Sall[s].SLeft[SOneTo(2), SVector(3, 4)])
        r = max(_σmax(RL), _σmax(RR))
        if s > 1
            d = stack[s].d
            ΦF = Diagonal(_SProp(d, modes[s].Qf)); ΦB = Diagonal(_SProp(-d, modes[s].Qb))
            M = SMatrix{2,2,ComplexF64}(I) - ΦF * RL * ΦB * RR
            r = max(r, _σmax(inv(M)))
        end
        return isfinite(r) ? r : 0.0
    catch
        return 0.0
    end
end

"""
    real_axis_poles(stack, ω, source_layer; φs=[0.0], k_lo, k_hi, scan_step=1e-3,
                    prominence=1.05, exclude=Float64[], floor_rel=1e-6, floor_abs=1e-10) -> Vector{NamedTuple{(:q,:Γ,:φ,:prom)}}

Real-axis poles of the field of a source in `source_layer`, along each in-plane
direction in `φs`:
  1. evaluate `_cavity_response` on a log-spaced kρ grid from `k_lo` to `k_hi`
     (relative step `scan_step`; the response is evaluated once per point, one
     4×4 S-matrix per layer);
  2. keep the local maxima that stand at least `prominence`× above the higher of
     the two valleys around them (searched over a factor 4 in kρ) and reach at least
     `floor_rel` × the largest response along that φ and `floor_abs` (the
     reflection of a homogeneous stack is round-off, whose wiggles are not poles);
  3. locate each maximum by golden-section search (`q`), and its half-width `Γ`
     where the response falls to `b + (peak − b)/√2` (`b` the valley level), the
     nearer side of the two: a pole `A/(kρ − q − iΓ)` has `|·|` at `1/√2` of its
     peak at `kρ = q ± Γ`, so `Γ ≈ Im` of the complex pole.

A peak narrower than the scan step is still found (its nearest sample is a local
maximum) as long as it stands out of the background. Peaks within `1e-6·q` of a
value in `exclude` (the light lines: mode normalisation, not poles) are dropped.
"""
function real_axis_poles(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                         φs=[0.0], k_lo::Real, k_hi::Real, scan_step::Real=1e-3,
                         prominence::Real=1.05, exclude=Float64[], floor_rel::Real=1e-6, floor_abs::Real=1e-10)
    ωf = Float64(ω)
    ks = collect(Float64(k_lo) .* exp.(0.0:Float64(scan_step):log(Float64(k_hi) / Float64(k_lo))))
    n = length(ks)
    W = ceil(Int, log(4.0) / scan_step)
    f(k, φ) = _cavity_response(stack, ωf, source_layer, k, φ)
    out = Vector{Vector{NamedTuple{(:q, :Γ, :φ, :prom),NTuple{4,Float64}}}}(undef, length(φs))
    @threads for m in eachindex(φs)
        φ = Float64(φs[m])
        y = [log(f(k, φ) + 1e-300) for k in ks]
        ymin = max(maximum(y) + log(floor_rel), log(floor_abs))   # round-off of a (nearly) reflectionless stack is not a pole
        found = NamedTuple{(:q, :Γ, :φ, :prom),NTuple{4,Float64}}[]
        for i in 2:n-1
            (y[i] > y[i-1] && y[i] >= y[i+1] && y[i] >= ymin) || continue
            jl, lmin = i, y[i]
            while jl > max(1, i - W) && y[jl-1] <= y[i]; jl -= 1; lmin = min(lmin, y[jl]); end
            jr, rmin = i, y[i]
            while jr < min(n, i + W) && y[jr+1] <= y[i]; jr += 1; rmin = min(rmin, y[jr]); end
            base = max(lmin, rmin)
            prom = y[i] - base
            prom >= log(prominence) || continue
            # locate the maximum
            a, b = ks[i-1], ks[i+1]
            g = (sqrt(5) - 1) / 2
            c, d = b - g * (b - a), a + g * (b - a); fc, fd = f(c, φ), f(d, φ)
            for _ in 1:60
                (b - a) <= 1e-12 * b && break
                if fc >= fd; b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c, φ)
                else;        a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d, φ)
                end
            end
            q = (a + b) / 2; P = max(f(q, φ), exp(y[i]))
            any(L -> abs(q - L) <= 1e-6 * q, exclude) && continue
            lvl = exp(base) + (P - exp(base)) / sqrt(2)
            # half-level crossings, bisection towards each valley
            function cross(lo, hi)             # f(lo) > lvl >= f(hi)
                for _ in 1:50
                    abs(hi - lo) <= 1e-12 * q && break
                    mid = (lo + hi) / 2
                    f(mid, φ) > lvl ? (lo = mid) : (hi = mid)
                end
                return (lo + hi) / 2
            end
            kl = ks[findlast(j -> y[j] == lmin, jl:i) + jl - 1]
            kr = ks[findfirst(j -> y[j] == rmin, i:jr) + i - 1]
            Γ = min(q - cross(q, kl), cross(q, kr) - q)
            push!(found, (q=q, Γ=max(Γ, 1e-12 * q), φ=φ, prom=prom))
        end
        out[m] = found
    end
    return reduce(vcat, out; init=NamedTuple{(:q, :Γ, :φ, :prom),NTuple{4,Float64}}[])
end

"""
    _clusters(poles, merge_rel) -> Vector{NamedTuple{(:a,:b,:Γ)}}

Intervals the fixed rule resolves at a pole's width: one `[q, q]` per pole, plus,
for every two poles adjacent in `q` that are within `merge_rel·q` of each other (or
within a few widths), the segment between them at the smaller of their two widths.
In 2D `merge_rel` is the φ step of the scan: a pole at a direction between two
scanned ones lies between their positions, with a width between theirs, so the
chain of segments covers its track without imposing the narrowest pole's width on
the whole range.
"""
function _clusters(poles, merge_rel::Float64)
    ps = sort(collect(poles); by=p -> p.q)
    cl = NamedTuple{(:a, :b, :Γ),NTuple{3,Float64}}[]
    for (i, p) in enumerate(ps)
        push!(cl, (a=p.q, b=p.q, Γ=p.Γ))
        if i > 1
            r = ps[i-1]
            p.q - r.q <= max(merge_rel * p.q, 4 * min(p.Γ, r.Γ)) && p.q > r.q &&
                push!(cl, (a=r.q, b=p.q, Γ=min(p.Γ, r.Γ)))
        end
    end
    return unique(cl)
end


# ─────────────────────────────────────────────────────────────────────────────
#  Features
# ─────────────────────────────────────────────────────────────────────────────

"""
    _pole_track_η(poles, φ_scan) -> Float64

Smallest distance `η` from the real φ axis of the φ-singularities that moving poles
put in the 2D integrand. At fixed kρ, a pole `q(φ) − iΓ` crosses kρ at some φ, where
the integrand peaks with a width `η ≈ Γ / |dq/dφ|` in φ; the trapezoid rule then
converges as `e^{−N_φ·η}`. `dq/dφ` is read from each pole and its nearest-`q`
partner at the neighbouring scanned direction. `Inf` with fewer than two scanned φ
or no pole track.
"""
function _pole_track_η(poles, φ_scan)
    φs = sort(unique(Float64.(φ_scan)))
    length(φs) < 2 && return Inf
    pairs = [(φs[i], φs[i+1], φs[i+1] - φs[i]) for i in 1:length(φs)-1]
    step = φs[2] - φs[1]
    abs(φs[end] - φs[1] + step - 2π) < 1e-9 && push!(pairs, (φs[end], φs[1], step))   # full circle: wrap
    η = Inf
    for (φ1, φ2, dφ) in pairs, p in poles, r in poles
        (p.φ == φ1 && r.φ == φ2) || continue
        dq = abs(p.q - r.q)
        0 < dq <= dφ * max(p.q, r.q) || continue            # same track (as `_clusters`)
        η = min(η, min(p.Γ, r.Γ) * dφ / dq)
    end
    return η
end


"""
    kspace_features(stack, ω, source_layer; h_min, ρ_max=0.0, dims=1, N_φ_scan=32,
                    target_decay=25.0, kρ_cap=Inf, scan=true, scan_step=1e-3,
                    prominence=1.05, c_φ=2.5, N_φ_min=64) -> NamedTuple

Everything the rules need (module notes). `h_min` is the smallest
source-to-observation vertical distance the grid must resolve (for the scattered
field on the source plane, twice the source-to-interface distance); `ρ_max` the
largest lateral offset. `dims = 2` scans the poles over several φ (a
`(kρ, φ)` grid shares one kρ rule for all φ): one direction for an azimuthally
symmetric stack, `N_φ_scan` over `[0, π/2]` for a diagonal anisotropic one (the
mirrors map the rest), `N_φ_scan` over `[0, 2π)` otherwise (a gyrotropic stack is
not symmetric under `φ → φ + π`).

`Kcut` overrides the geometric cut-off (the box grid has its own, algebraic one);
`φ_scan` overrides the scanned directions (a φ group of `box_kρ_grid_grouped`
scans its own φ).

Fields: `classes`, `light_lines` (all layers, every scanned φ), `singular_lines`
(semi-infinite layers only), `kmax`, `Kcut`, `h_min`, `ρ_max`, `poles`,
`clusters` (`_clusters`), `N_φ` (2D; the largest of `N_φ_min`, `N_φ_lateral` =
`c_φ·Kcut·ρ_max` (harmonics of `e^{ikρρcos(φ−ψ)}` up to `kρρ`) and `N_φ_poles` =
`c_η·log(10/rtol)/η_min`, `c_η = 1.5` (1.0 left InSb at 5.8e-6) (`_pole_track_η`: poles that move with φ make the integrand
peak in φ), rounded up to even, so the `φ + π` pairs the reciprocity checks need
are on the grid), `N_φ_lateral`, `N_φ_poles`, `η_min`, `φ_scan`.
"""
function kspace_features(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                         h_min::Real, ρ_max::Real=0.0, dims::Int=1, N_φ_scan::Int=32,
                         target_decay::Real=25.0, kρ_cap::Real=Inf, scan::Bool=true,
                         scan_step::Real=1e-3, prominence::Real=1.05, c_φ::Real=2.5, N_φ_min::Int=64,
                         Kcut::Union{Nothing,Real}=nothing, φ_scan=nothing, rtol::Real=1e-6, c_η::Real=1.5)
    dims in (1, 2) || error("kspace_features: dims must be 1 or 2")
    N = length(stack)
    ωf = Float64(ω)
    classes = [material_class(l) for l in stack]
    sym = _is_azimuthally_symmetric_stack(stack)
    diag = all(l -> (e = _tensor3(l.ϵ); m = _tensor3(_layer_μ(l)); all(iszero(e[i, j]) && iszero(m[i, j]) for i in 1:3, j in 1:3 if i != j)), stack)
    φ_scan, merge_rel = if φ_scan !== nothing
        sort(collect(Float64, φ_scan)), (length(φ_scan) > 1 ? minimum(diff(sort(collect(Float64, φ_scan)))) : 0.0)
    elseif dims == 1 || sym
        [0.0], 0.0
    elseif diag
        collect(range(0.0, π / 2; length=max(2, N_φ_scan ÷ 4 + 1))), (π / 2) / max(1, N_φ_scan ÷ 4)
    else
        [2π * p / N_φ_scan for p in 0:N_φ_scan-1], 2π / N_φ_scan
    end
    φ_ll = dims == 1 || sym ? [0.0] : unique(mod.(φ_scan, π))         # light lines are even in k̂

    ll = Float64[]; sing = Float64[]
    for (j, lay) in enumerate(stack), φ in φ_ll
        L = layer_light_lines(lay, ωf, φ)
        append!(ll, L)
        (j == 1 || j == N) && append!(sing, L)
    end
    ll = _dedup(sort!(ll)); sing = _dedup(sort!(sing))
    kmax = isempty(ll) ? ωf : ll[end]
    # from the geometry only (handout P1): the 80·kmax floor of the old tail is not applied,
    # only 2·kmax, so the band always extends past the last light line
    Kcut = Kcut === nothing ? tail_bound(decay_cutoff(h_min; target_decay=target_decay), kmax;
                                         kρ_cap=kρ_cap, kx_ratio_min=2.0) : Float64(Kcut)

    poles = if scan
        k_lo = 0.5 * (isempty(ll) ? ωf : ll[1])
        real_axis_poles(stack, ωf, source_layer; φs=φ_scan, k_lo=k_lo, k_hi=Kcut,
                        scan_step=scan_step, prominence=prominence, exclude=ll)
    else
        NamedTuple{(:q, :Γ, :φ, :prom),NTuple{4,Float64}}[]
    end
    clusters = _clusters(poles, merge_rel)
    N_lat = ceil(Int, c_φ * Kcut * ρ_max)
    η = _pole_track_η(poles, φ_scan)
    N_pole = isfinite(η) ? ceil(Int, c_η * log(10 / Float64(rtol)) / η) : 0
    N_φ = dims == 2 ? 2 * cld(max(N_φ_min, N_lat, N_pole), 2) : 0
    return (classes=classes, light_lines=ll, singular_lines=sing, kmax=kmax, Kcut=Kcut,
            h_min=Float64(h_min), ρ_max=Float64(ρ_max), poles=poles, clusters=clusters,
            N_φ=N_φ, N_φ_lateral=N_lat, N_φ_poles=N_pole, η_min=η, φ_scan=φ_scan, dims=dims)
end


# ─────────────────────────────────────────────────────────────────────────────
#  Engines
# ─────────────────────────────────────────────────────────────────────────────

"""
    fixed_kρ_rule(feat; order=8, rtol=1e-6, α=0.5, grade=12, r_rel=0.05,
                  c_h=4.0, c_ρ=4.0, pole_safety=0.5) -> (nodes, weights, panels)

Composite Gauss–Legendre rule (`order` nodes per panel) on `[0, feat.Kcut]`,
breaks at every light line, panels from `KxGrid.graded_panels` with the local
target width the smallest of:

  - background: `min(c_h/h_min, c_ρ/ρ_max, max(r_rel·kρ, kmax/40))` -- follows
    `e^{−kρh}` and the lateral `e^{ikρρ}` (≤ `c_*` radians per panel), and
    coarsens by ×10 per decade of kρ (`r_rel`);
  - each singular light line `L`: `max(L·2^-grade, α·|kρ − L|)` -- geometric
    grading onto the `1/k_z` branch point from both sides. The two panels touching
    `L` are integrated in `u`, `kρ = L ∓ u²` (`KxGrid.gl_rule`'s `singular`), which
    removes the singularity (the handout's Purcell-figure substitution), so 2⁻¹² is
    deep enough. Grading alone needed 2⁻⁴⁶ for 1e-6 (the last panel's error is
    `∝ √width`), which puts nodes within 1e-14 of `L`, where the general `eigen`
    path fails. A light line that moves with φ is a break only at the scanned φ;
    between them it falls inside the graded panels, not on a substituted one;
  - each pole or pole segment `[a, b]` of width `Γ` (`_clusters`): `max(w_Γ, α·dist(kρ, [a, b]))`, with
    `w_Γ = pole_safety · 2Γ/ε_B`, `ε_B = (ρ_B − 1/ρ_B)/2`, `ρ_B = rtol^(−1/2order)`:
    the panel half-width at which a pole at distance `Γ` from the real axis keeps
    the Gauss–Legendre error below `rtol` (Bernstein ellipse). For `rtol = 1e-6`,
    order 8: `w_Γ ≈ Γ`, i.e. a node spacing of `Γ/8`.

Every term is Lipschitz with constant ≤ `α`, which `graded_panels` needs. Halving
every panel (`KxGrid.halve_panels`, then `gl_rule(...; singular=feat.singular_lines)`)
is the convergence check.
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

_grade_for(rtol) = min(50, ceil(Int, 2 * log2(10 / Float64(rtol))))

"""
    adaptive_seeds(feat; rtol=1e-6, n_Γ=4.0, grade=2·log2(10/rtol)) -> Vector{Float64}

Starting breakpoints for `KxGrid.adaptive_kx_grid` from the features: the light
lines, geometric grading `L(1 ± 2^-j)`, `j = 1…grade`, onto the singular ones,
each pole segment's `[a − n_Γ·Γ, a, b, b + n_Γ·Γ]`, and log-spaced seeds (×2) from
`kmax` to `Kcut`, so the long band past the light lines starts out graded as the
old `t/(1−t)` tail was. Without the grading, Gauss–Kronrod judged a coarse panel
next to a lossless light line converged while 1e-2 off (dipole above a lossless
magnetic half-space); from the graded start it refines where it must.
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

end
