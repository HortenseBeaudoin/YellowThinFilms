module SpectralSingularities

export material_class, layer_light_lines, branch_points, real_axis_poles, kspace_features
export decay_cutoff, tail_bound, box_tail_cutoff, thinnest_finite_layer

import ..LayeredMaterials: AbstractLayer, _layer_μ, _is_azimuthally_symmetric_stack
import ..BerremanModes: Modes_Stack
import ..ScatteringMatrices: S_all_Matrix, S_all_ingredients, _Redheffer
import ..StackResponses: _SProp
using LinearAlgebra
using StaticArrays
using Base.Threads

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    _tensor3(m) -> SMatrix{3,3,ComplexF64}

Turns a material tensor into a plain dense 3×3 statis array. For speed,
when material type doesn't matter,
"""
_tensor3(m) = SMatrix{3,3,ComplexF64}(ntuple(n -> ComplexF64(m[(n - 1) % 3 + 1, (n - 1) ÷ 3 + 1]), 9))

"""
    _dedup(ks; rtol::Float64=1e-9) -> Vector{Float64}

Merges values closer than `rtol` (relative), keeping the first of each cluster
"""
function _dedup(ks::Vector{Float64}; rtol::Float64=1e-9)
    out = Float64[]
    for k in ks
        (isempty(out) || k - out[end] > rtol * max(abs(k), 1e-300)) && push!(out, k)
    end
    return out
end

"""
    material_class(layer) -> Symbol

What kind of medium `layer` is at the frequency its ε and μ were evaluated at.
  - `:gyrotropic` if ε or μ not symmetric (`ε ≠ εᵀ`: magneto-optic, Voigt/Faraday);
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

# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

# Light lines ──────────────────────────

"""
    layer_light_lines(layer, ω, φ) -> Vector{Float64}

In-plane wavenumbers at which a plane wave travelling along `(cos φ, sin φ)`
in `layer` has `k_z = 0`. Solves `det(ω²ε + kρ²·K̂ μ⁻¹ K̂) = 0`, where `K̂` is
the cross-product matrix of `k̂`. The roots are `q = kρ²`, and only those with
`Re q > 0` are light-lines. 

# Arguments
- `layer::AbstractLayer`: a single layer of material in the stack. 
- `ω::Real`: frequency.
- `φ::Real`: direction of travel in-plane, `(cos φ, sin φ)`. 

# Returns
- `Vector{Float64}`: containing the light-lines computed from the roots,
  `kρ = Re √q`. 
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

Calls `layer_light_lines` for each layer of the stack, combines the results and 
remove duplicates. By default, scans over φ ∈ [0.0, 2π]

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, with N the number of layers 
of the stack + 2 (half-plane and/or substrate). Each entry contains a layer's description. 
- `ω::Real`: frequency.

# Returns
- `Vector{Float64}`: containing the light-lines of the whole stack. 
"""
function branch_points(stack::Vector{<:AbstractLayer}, ω::Real; φs=(0.0, π / 2))
    ks = Float64[]
    for lay in stack, φ in φs
        append!(ks, layer_light_lines(lay, ω, φ))
    end
    return _dedup(sort!(ks))
end

#  Cut-off ──────────────────────────

"""
    decay_cutoff(h; target_decay=25.0) -> Float64

Cut-off of the evanescent tail, default at 25.0, i.e e^{−25} ≈ 1.4e-11.
`h` is the smallest source-to-observation distance the grid must resolve.
"""
decay_cutoff(h::Real; target_decay::Real=25.0) = Float64(target_decay) / Float64(h)

"""
    tail_bound(K, kmax; kρ_cap=Inf, kx_ratio_min=80.0) -> Float64

Bounds the decay cutoff between a floor and a ceiling, given a decay and 
the last light-line on the real axis `kmax`. 
"""
tail_bound(K::Real, kmax::Real; kρ_cap::Real=Inf, kx_ratio_min::Real=80.0) =
    min(max(Float64(K), Float64(kx_ratio_min) * kmax), Float64(kρ_cap))

"""
    box_tail_cutoff(Δ, rtol) -> K

Smallest `K` with `log(K·Δ)/(K·Δ) ≤ rtol`, found  by fixed-point iteration on
`K = log(K·Δ)/(rtol·Δ)`. Box-source counterpart of `decay_cutoff`.
"""
function box_tail_cutoff(Δ::Real, rtol::Real)
    K = 10 / (rtol * Δ)
    for _ in 1:50
        K = max(log(K * Δ), 1.0) / (rtol * Δ)
    end
    return K
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

#  Pole of R scan ──────────────────────────

"""
    _σmax(A::SMatrix{2,2})

Returns the largest singular value of a 2x2 matrix, using the closed-form 
which is its spectral norm ‖A‖₂.
"""
function _σmax(A::SMatrix{2,2})
    G = A' * A
    t = real(G[1, 1] + G[2, 2]); dt = real(G[1, 1] * G[2, 2] - G[1, 2] * G[2, 1])
    return sqrt(max((t + sqrt(max(t^2 - 4dt, 0.0))) / 2, 0.0))
end

"""
    _cavity_response(stack, ω, s, kρ, φ) -> Float64

Computes the singular values of `Rl`, `RR` and, for finite layer, of the cavity 
resummation `(I − ΦF·RL·ΦB·RR)⁻¹`. Returns the largest among them as the peak for
the given pair (kρ, φ). 
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
    real_axis_poles(stack, ω, source_layer; φs=[0.0], k_lo, k_hi) -> Vector{NamedTuple{(:q,:Γ,:φ,:prom)}}

Founds the real-axis poles along each in-plane direction in `φs`. Evaluates
`_cavity_response` on a log-spaced kρ grid from `k_lo` to `k_hi`, keeps the 
local maxima that standout and locate each more precisely by golden-section
search. A peak narrower than the scan step is still found, as long as it 
stands out of the background.

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, with N the number of layers 
of the stack + 2 (half-plane and/or substrate). Each entry contains a layer's description. 
- `ω::Real`: frequency.
- `source_layer::Int`: id of the layer containing the source.
- `φs=[0.0]`: direction of travel in-plane, `(cos φ, sin φ)`. 
- `k_lo::Real`: lower bound of the search log-spaced kρ grid.
- `k_hi::Real`: upper bound of the search log-spaced kρ grid.

# Returns
- `NamedTuple{(:q, :Γ, :φ, :prom)`: containing each pole `q`, its width `Γ`,
  its direction `φ` and its prominence `prom`. 
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

Returns 0 length interval `[q, q]` for each pole, along with the width `Γ`. For
neighbouring poles closer than the given tolerance, also return the segment 
between them `[a, b]`. 
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

#  Combined features ──────────────────────────

"""
    _pole_track_η(poles, φ_scan) -> Float64

Returns `η`, the smallest feature to resolve in φ when integrating in 2D. 
At a fixed kρ, a pole `q(φ) − iΓ` crosses kρ at some φ: those are the φ-singularities. 
At those crossing, the integrand peaks with a width `η ≈ Γ / |dq/dφ|`. `dq/dφ` is read 
from each pole and its nearest-`q` partner at the neighbouring scanned direction.
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
    kspace_features(stack, ω, source_layer; h_min) -> NamedTuple

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, with N the number of layers 
of the stack + 2 (half-plane and/or substrate). Each entry contains a layer's description. 
- `ω::Real`: frequency.
- `source_layer::Int`: id of the layer containing the source.
- `h_min`: smallest source-to-observation longitudinal distance, typically 
  discretization in z. 

# Returns
- `NamedTuple`: contains all features of the integrand. 
Fields: `classes`, `light_lines`, `singular_lines`, `kmax`, `Kcut`, `h_min`, 
`ρ_max`, `poles`, `clusters`, `N_φ`, `N_φ_lateral`, `N_φ_poles`, `η_min`, 
`φ_scan`, `K_geom`, `K_cap`, `capped`.
"""
function kspace_features(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int;
                         h_min::Real, ρ_max::Real=0.0, dims::Int=1, N_φ_scan::Int=32,
                         target_decay::Real=25.0, kρ_cap::Union{Nothing,Real}=nothing, kρ_cap_rel::Real=1e5, scan::Bool=true,
                         scan_step::Real=1e-3, prominence::Real=1.05, c_φ::Real=2.5, N_φ_min::Int=64,
                         Kcut::Union{Nothing,Real}=nothing, φ_scan=nothing, rtol::Real=1e-6, c_η::Real=1.5)
    dims in (1, 2) || error("kspace_features: dims must be 1 or 2")
    N = length(stack)
    ωf = Float64(ω)
    classes = [material_class(l) for l in stack]
    sym = _is_azimuthally_symmetric_stack(stack)
    diag = all(l -> (e = _tensor3(l.ϵ); m = _tensor3(_layer_μ(l)); all(iszero(e[i, j]) && iszero(m[i, j]) for i in 1:3, j in 1:3 if i != j)), stack)
    φ_scan, merge_rel = if dims == 1 || sym                 # every φ has the same poles
        [0.0], 0.0
    elseif φ_scan !== nothing
        sort(collect(Float64, φ_scan)), (length(φ_scan) > 1 ? minimum(diff(sort(collect(Float64, φ_scan)))) : 0.0)
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
    K_geom = decay_cutoff(h_min; target_decay=target_decay)
    K_cap = kρ_cap === nothing ? Float64(kρ_cap_rel) * kmax : Float64(kρ_cap)
    capped = false
    if Kcut === nothing
        Kcut = tail_bound(K_geom, kmax; kρ_cap=K_cap, kx_ratio_min=2.0)
        if Kcut < K_geom
            capped = true
            @warn "kspace_features: the cost ceiling cuts the k-band at $(round(Kcut; sigdigits=4)) " *
                  "below the geometric cut-off target_decay/h_min = $(round(K_geom; sigdigits=4)) " *
                  "(h_min = $(h_min)): content closer than $(round(target_decay / Kcut; sigdigits=3)) " *
                  "to the source is not resolved (integrand left at e^{-$(round(Kcut * h_min; sigdigits=3))}). " *
                  "Raise kρ_cap / kρ_cap_rel, or accept the truncation."
        end
    else
        Kcut = Float64(Kcut)
    end

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
            N_φ=N_φ, N_φ_lateral=N_lat, N_φ_poles=N_pole, η_min=η, φ_scan=φ_scan, dims=dims,
            K_geom=K_geom, K_cap=K_cap, capped=capped)
end

end
