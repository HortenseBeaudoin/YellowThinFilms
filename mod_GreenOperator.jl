"""
mod_GreenOperator.jl -- precomputed, z-structured Green's operator of one
source layer onto a set of observation layers, on uniform z grids.

The user declares the source layer (its `CavityTransferCache`), the uniform
grid of source planes in it, and the uniform grid of observation planes in
each wanted observation layer. Everything grid-dependent is precomputed;
`apply_operator` then does only FFTs, pointwise 3×3 products and (for dense
blocks) plane-by-plane products, all in k-space.

Lateral: as `mod_ClosedFormAssembly.jl` -- `G` sampled on the 2× zero-padded
offset grid, FFT'd once, so the lateral sum over sources is an exact linear
convolution.

Axial, one block per (source layer, observation layer):

  - same layer, structured (isotropic source layer, or uniaxial in an
    azimuthally symmetric stack -- the interfaces keep each mode's `Q`):
        G(zpos, z') = T(z' − zpos) + H(z' + zpos)
    `T` = direct term + matched-branch reflections (`terms=:toeplitz`),
    `H` = cross-branch reflections (`terms=:hankel`). On a uniform grid this
    is a Toeplitz plus a Hankel matrix in the plane indices: `Ns+No−1` planes
    each instead of `Ns·No`. Both are applied as convolutions along z by FFT
    (length `Lz ≥ Ns+No−1`), so this block's apply is 3D FFTs + pointwise
    3×3 products.
  - same layer, other media (biaxial/gyrotropic): the direct term is still
    Toeplitz (homogeneous layer); the scattered part is stored per
    (source plane, observation plane) pair -- dense.
  - cross layer: dense per pair (low-rank compression is the next step).

`structure=:auto` picks the structured form from the media and then checks it
numerically on a few pixels (`check=true`); it falls back to dense, with a
warning, if the Toeplitz/Hankel identity does not hold.

Conventions: source moments are per voxel/pixel (a dipole moment `p`, not a
density); `apply_operator(op, P)` takes `P[i, j, k, q]` = moment component `q`
at pixel `(i, j)` of `grid` and source plane `k` of `z_src`, and returns, per
observation layer, `E[i, j, k, c]` on that layer's plane `k`. The `z_src` values
are the source layer's local depths as `closed_form_G` takes them (`Dipole.z`).
"""
module GreenOperator

export LayerGreenOperator, build_layer_operator, apply_operator, dipole_moments, operator_summary

import ..ModeSolver: _is_azimuthally_symmetric_stack, IsotropicMaterial, UniaxialMaterial
import ..CavityTransfer: AbstractCavityTransferCache, CavityTransferCache1D, CavityTransferCache2D
import ..ClosedFormField: closed_form_G
import ..ClosedFormAssembly: padded_offsets, _plan, _padded_size
import ..SourceHandling: Dipole
import ..UniformSource: _nearest_node_index

using LinearAlgebra
using Base.Threads
using FFTW
using Printf

# ─────────────────────────────────────────────────────────────────────────────
#  Blocks
# ─────────────────────────────────────────────────────────────────────────────

"""
One (source layer → observation layer) block. `kind`:
`:structured` (`Tz`, `Hz` set), `:toeplitz_dense` (`Tz`, `D` set), `:dense` (`D` set).
`Tz`/`Hz` are `(Px, Py, Lz, 3, 3)`, already FFT'd along z (with `1/Lz` and, for
`Hz`, the output shift folded in). `D` is `(Px, Py, 3, 3, Ns, No)`.
"""
struct ZBlock
    obs_layer :: Int
    z_obs     :: Vector{Float64}
    kind      :: Symbol
    Lz        :: Int
    Tz        :: Union{Nothing,Array{ComplexF64,5}}
    Hz        :: Union{Nothing,Array{ComplexF64,5}}
    D         :: Union{Nothing,Array{ComplexF64,6}}
end

"""
    LayerGreenOperator

Precomputed operator of one source layer. `z_src` in the caller's order;
internally the source planes are ordered by increasing cavity-frame depth
`zpos` (`flip = true` for source layer 1, whose local `z` runs the other way).
"""
struct LayerGreenOperator
    source_layer :: Int
    grid         :: NamedTuple
    z_src        :: Vector{Float64}
    flip         :: Bool
    blocks       :: Vector{ZBlock}
    kwargs       :: NamedTuple
    timings      :: Dict{Int,Float64}
end

# ─────────────────────────────────────────────────────────────────────────────
#  Ĝ planes (lateral FFT of G on the padded offset grid)
# ─────────────────────────────────────────────────────────────────────────────

"""
`Ĝ` of one source depth at observation depths `zos`, `(Px, Py, length(zos), 3, 3)`:
`fft` over the padded offsets, centred → FFT-native shift as the sign `(−1)^(i+j)`,
scaled by `1/(Px·Py)` (moments, not densities).
"""
function _ghat_planes(cache, obs_layer, zsrc, zos, grid; terms=:all, kw...)
    xs, ys = padded_offsets(grid)
    G = closed_form_G(cache, obs_layer, zsrc, zos, xs, ys; terms=terms, kw...)
    FFTW.set_num_threads(Threads.nthreads())
    fft!(G, (1, 2))                       # sizes vary per call: estimate-planned, not cached
    Px, Py = size(G, 1), size(G, 2)
    scale = 1.0 / (Px * Py)
    @threads for j in 1:Py
        @inbounds for c in axes(G, 5), r in axes(G, 4), zi in axes(G, 3), i in 1:Px
            G[i, j, zi, r, c] *= iseven(i + j) ? scale : -scale
        end
    end
    return G
end

"""
Evaluates `Ĝ` for a list of `(source plane i, observation plane j) → slot` requests,
grouped by source plane (one `closed_form_G` call, all its rows, per plane).
`put!(slot, view)` stores each `(Px, Py, 3, 3)` result.
"""
function _fill_by_source!(put!, requests, cache, o, zsrc_int, zobs, grid; terms, kw...)
    bysrc = Dict{Int,Vector{Tuple{Int,Any}}}()
    for (i, j, slot) in requests
        push!(get!(() -> Tuple{Int,Any}[], bysrc, i), (j, slot))
    end
    for (i, list) in bysrc
        Ĝ = _ghat_planes(cache, o, zsrc_int[i], [zobs[j] for (j, _) in list], grid; terms=terms, kw...)
        for (n, (_, slot)) in enumerate(list)
            put!(slot, view(Ĝ, :, :, n, :, :))
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  Structure detection and check
# ─────────────────────────────────────────────────────────────────────────────

"""
Isotropic source layer (both modes share `Q`, so mode mixing at the interfaces is
harmless), or uniaxial with the axis along z in an azimuthally symmetric stack
(the interfaces never mix its ordinary and extraordinary modes).
"""
function _structured_by_media(cache)
    lay = cache.stack[cache.source_layer]
    μ = hasproperty(lay, :μ) ? lay.μ : nothing
    (μ === nothing || μ isa IsotropicMaterial) || return false
    lay.ϵ isa IsotropicMaterial && return true
    return lay.ϵ isa UniaxialMaterial && _is_azimuthally_symmetric_stack(cache.stack)
end

"""
Numerical check of `G_scattered(z,z') = T(z'−z) + H(z'+z)` on a few pixels:
`f(a,b) − f(a+δ,b+δ) − f(a−δ,b+δ) + f(a,b+2δ) = 0` on grid planes (cavity frame).
Returns the relative residual, or `NaN` if the grids are too small to test.
"""
function _check_structure(cache, zsrc_int, zobs, kw)
    Ns, No = length(zsrc_int), length(zobs)
    (Ns >= 3 && No >= 3) || return NaN
    i0, j0 = (Ns + 1) ÷ 2, min((No + 1) ÷ 2, No - 2)
    xs, ys = [0.0, 0.17, 0.4], [0.0, -0.11]
    kwd = cache isa CavityTransferCache2D ? merge(kw, (lateral=:direct,)) : kw
    f(i, j) = closed_form_G(cache, cache.source_layer, zsrc_int[i], [zobs[j]], xs, ys; terms=:scattered, kwd...)
    r = f(i0, j0) .- f(i0 + 1, j0 + 1) .- f(i0 - 1, j0 + 1) .+ f(i0, j0 + 2)
    return norm(r) / max(norm(f(i0, j0)), 1e-300)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Build
# ─────────────────────────────────────────────────────────────────────────────

_uniform_step(z::AbstractVector) = length(z) == 1 ? 0.0 :
    (Δ = (z[end] - z[1]) / (length(z) - 1);
     maximum(abs(z[k] - (z[1] + (k - 1) * Δ)) for k in eachindex(z)) <= 1e-9 * max(abs(Δ), 1.0) ||
         error("build_layer_operator: z grids must be uniform"); Δ)

"Representative `(i, j)` with `j − i = m` (Toeplitz) on `1:Ns × 1:No`, preferring source planes 1 and Ns."
function _toeplitz_pair(m, Ns, No)
    m >= 0 && return (1, m + 1)
    (1 <= Ns + m <= No) && return (Ns, Ns + m)
    return (1 - m, 1)
end

"Representative `(i, j)` with `i + j − 2 = s` (Hankel) on `1:Ns × 1:No`."
function _hankel_pair(s, Ns, No)
    s + 1 <= No && return (1, s + 1)
    (1 <= s + 2 - Ns <= No) && return (Ns, s + 2 - Ns)
    return (s + 2 - No, No)
end

"""
    build_layer_operator(cache, grid, z_src, z_obs_by_layer; structure=:auto, check=true, kwargs...)
        -> LayerGreenOperator

`cache`: the source layer's `CavityTransferCache1D`/`2D`, built with every
observation layer among its `obs_layers`. `grid`: a `uniform_kxky_grid`.
`z_src`: the uniform source planes (local depths, as `Dipole.z`).
`z_obs_by_layer[o]`: the uniform observation planes of layer `o` (local depths).
For the structured same-layer form the source and observation steps must be
equal. `kwargs` go to `closed_form_G` (e.g. `box_lateral=(Δx,Δy)`, `Δz_box`,
`lateral=:nufft`, `nufft_eps`).

`structure`: `:auto` (media test + numerical check), `:structured` (force),
`:dense` (no z structure in the same-layer block: `D` for everything, as a reference).
"""
function build_layer_operator(cache::AbstractCavityTransferCache, grid::NamedTuple, z_src::AbstractVector{<:Real},
                              z_obs_by_layer::AbstractDict{Int,<:AbstractVector{<:Real}};
                              structure::Symbol=:auto, check::Bool=true, check_tol::Real=1e-8, kwargs...)
    structure in (:auto, :structured, :dense) || error("build_layer_operator: structure must be :auto, :structured or :dense")
    kw = NamedTuple(kwargs)
    s = cache.source_layer
    zs = collect(Float64, z_src)
    Δs = _uniform_step(zs)
    flip = s == 1                                   # layer 1: zpos = d − z, so reverse to get increasing zpos
    zs_int = flip ? reverse(zs) : zs
    Ns = length(zs)
    Px, Py = _padded_size(grid)
    blocks = ZBlock[]
    timings = Dict{Int,Float64}()

    for o in sort(collect(keys(z_obs_by_layer)))
        zo = collect(Float64, z_obs_by_layer[o])
        isempty(zo) && continue
        No = length(zo)
        Δo = _uniform_step(zo)
        (Δo >= 0) || error("build_layer_operator: z_obs for layer $o must be increasing")
        t = @elapsed begin
            if o == s
                same_step = Ns == 1 || No == 1 || isapprox(abs(Δs), Δo; rtol=1e-9)
                structured = structure === :structured ||
                             (structure === :auto && same_step && _structured_by_media(cache))
                if structured && check && structure === :auto
                    res = _check_structure(cache, zs_int, zo, kw)
                    if res > check_tol
                        @warn "build_layer_operator: Toeplitz+Hankel check failed on layer $o (residual $res) -- using the dense form"
                        structured = false
                    end
                end
                if structure === :dense || !same_step
                    push!(blocks, _dense_block(cache, o, zs_int, zo, grid, Ns, No; terms=:all, kw...))
                elseif structured
                    push!(blocks, _structured_block(cache, o, zs_int, zo, grid, Ns, No; kw...))
                else
                    push!(blocks, _toeplitz_dense_block(cache, o, zs_int, zo, grid, Ns, No; kw...))
                end
            else
                push!(blocks, _dense_block(cache, o, zs_int, zo, grid, Ns, No; terms=:all, kw...))
            end
        end
        timings[o] = t
    end
    return LayerGreenOperator(s, grid, zs, flip, blocks, kw, timings)
end

_zlen(Ns, No) = nextprod([2, 3, 5], Ns + No - 1)

"Toeplitz planes `m = −(Ns−1):(No−1)` at circular z-slots `mod(m, Lz)`, FFT'd along z, `/Lz`."
function _toeplitz_z(cache, o, zs, zo, grid, Ns, No, Lz; terms, kw...)
    Px, Py = _padded_size(grid)
    A = zeros(ComplexF64, Px, Py, Lz, 3, 3)
    reqs = [(_toeplitz_pair(m, Ns, No)..., mod(m, Lz) + 1) for m in -(Ns - 1):(No - 1)]
    _fill_by_source!((slot, v) -> (A[:, :, slot, :, :] .= v), reqs, cache, o, zs, zo, grid; terms=terms, kw...)
    _zfft!(A)
    A ./= Lz
    return A
end

function _zfft!(A::Array{ComplexF64,5})
    FFTW.set_num_threads(Threads.nthreads())
    fft!(A, 3)
end

function _structured_block(cache, o, zs, zo, grid, Ns, No; kw...)
    Lz = _zlen(Ns, No)
    Tz = _toeplitz_z(cache, o, zs, zo, grid, Ns, No, Lz; terms=:toeplitz, kw...)
    # Hankel planes s = 0:(Ns+No−2) at z-slot s; the reversed-source convolution lands at
    # s' = j0 + Ns − 1, shifted back by −(Ns−1): fold e^{+2πik(Ns−1)/Lz} into the spectrum.
    Px, Py = _padded_size(grid)
    Hz = zeros(ComplexF64, Px, Py, Lz, 3, 3)
    reqs = [(_hankel_pair(sidx, Ns, No)..., sidx + 1) for sidx in 0:(Ns + No - 2)]
    _fill_by_source!((slot, v) -> (Hz[:, :, slot, :, :] .= v), reqs, cache, o, zs, zo, grid; terms=:hankel, kw...)
    _zfft!(Hz)
    for k in 1:Lz
        Hz[:, :, k, :, :] .*= cis(2π * (k - 1) * (Ns - 1) / Lz) / Lz
    end
    return ZBlock(o, zo, :structured, Lz, Tz, Hz, nothing)
end

function _dense_planes(cache, o, zs, zo, grid, Ns, No; terms, kw...)
    Px, Py = _padded_size(grid)
    D = Array{ComplexF64,6}(undef, Px, Py, 3, 3, Ns, No)
    reqs = [(i, j, (i, j)) for i in 1:Ns for j in 1:No]
    _fill_by_source!((slot, v) -> (D[:, :, :, :, slot[1], slot[2]] .= v), reqs, cache, o, zs, zo, grid; terms=terms, kw...)
    return D
end

_dense_block(cache, o, zs, zo, grid, Ns, No; terms, kw...) =
    ZBlock(o, zo, :dense, 0, nothing, nothing, _dense_planes(cache, o, zs, zo, grid, Ns, No; terms=terms, kw...))

function _toeplitz_dense_block(cache, o, zs, zo, grid, Ns, No; kw...)
    Lz = _zlen(Ns, No)
    Tz = _toeplitz_z(cache, o, zs, zo, grid, Ns, No, Lz; terms=:direct, kw...)
    D = _dense_planes(cache, o, zs, zo, grid, Ns, No; terms=:scattered, kw...)
    return ZBlock(o, zo, :toeplitz_dense, Lz, Tz, nothing, D)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Apply
# ─────────────────────────────────────────────────────────────────────────────

"`Ê[.., k, r] += Σ_c K[.., k, r, c] · Ĵ[.., k, c]` over all padded nodes and z-slots."
function _zmul_add!(Ê::Array{ComplexF64,4}, K::Array{ComplexF64,5}, Ĵ::Array{ComplexF64,4})
    Px, Py, L = size(K, 1), size(K, 2), size(K, 3)
    @threads for t in 1:(Py * L)
        j = (t - 1) % Py + 1
        k = (t - 1) ÷ Py + 1
        @inbounds for i in 1:Px
            J1, J2, J3 = Ĵ[i, j, k, 1], Ĵ[i, j, k, 2], Ĵ[i, j, k, 3]
            for r in 1:3
                Ê[i, j, k, r] += K[i, j, k, r, 1] * J1 + K[i, j, k, r, 2] * J2 + K[i, j, k, r, 3] * J3
            end
        end
    end
    return Ê
end

"`Ê[.., j, r] += Σ_{i,c} D[.., r, c, i, j] · Ĵ[.., i, c]` (dense plane pairs)."
function _dense_add!(Ê::Array{ComplexF64,4}, D::Array{ComplexF64,6}, Ĵ::Array{ComplexF64,4})
    Px, Py, Ns, No = size(D, 1), size(D, 2), size(D, 5), size(D, 6)
    @threads for t in 1:(Py * No)
        y = (t - 1) % Py + 1
        j = (t - 1) ÷ Py + 1
        @inbounds for i in 1:Ns, x in 1:Px
            J1, J2, J3 = Ĵ[x, y, i, 1], Ĵ[x, y, i, 2], Ĵ[x, y, i, 3]
            for r in 1:3
                Ê[x, y, j, r] += D[x, y, r, 1, i, j] * J1 + D[x, y, r, 2, i, j] * J2 + D[x, y, r, 3, i, j] * J3
            end
        end
    end
    return Ê
end

"Zero-pad `Ĵ2[:, :, 1:Ns, :]` (or its plane-reversed copy) to `Lz` along z and FFT along z."
function _zpad_fft(Ĵ2::Array{ComplexF64,4}, Lz::Int; reversed::Bool=false)
    Px, Py, Ns, _ = size(Ĵ2)
    Z = zeros(ComplexF64, Px, Py, Lz, 3)
    @views Z[:, :, 1:Ns, :] .= reversed ? Ĵ2[:, :, Ns:-1:1, :] : Ĵ2
    FFTW.set_num_threads(Threads.nthreads())
    fft!(Z, 3)
    return Z
end

"""
    apply_operator(op, P) -> Dict{Int, Array{ComplexF64,4}}   # obs_layer => (Nx, Ny, No, component)

`P[i, j, k, q]`: dipole moment component `q` at pixel `(i, j)` of `op.grid` on
source plane `k` of `op.z_src` (caller's order). Only FFTs and pointwise products.
"""
function apply_operator(op::LayerGreenOperator, P::AbstractArray{<:Number,4})
    g = op.grid
    Ns = length(op.z_src)
    size(P) == (g.Nx, g.Ny, Ns, 3) || error("apply_operator: P must be (Nx, Ny, Ns, 3) = $((g.Nx, g.Ny, Ns, 3)), got $(size(P))")
    Px, Py = _padded_size(g)
    Ĵ2 = zeros(ComplexF64, Px, Py, Ns, 3)
    @views Ĵ2[1:g.Nx, 1:g.Ny, :, :] .= op.flip ? P[:, :, Ns:-1:1, :] : P
    _plan(:fft, Ĵ2) * Ĵ2

    zcache = Dict{Tuple{Int,Bool},Array{ComplexF64,4}}()
    zspec(L, rev) = get!(() -> _zpad_fft(Ĵ2, L; reversed=rev), zcache, (L, rev))

    out = Dict{Int,Array{ComplexF64,4}}()
    for b in op.blocks
        No = length(b.z_obs)
        Ê = zeros(ComplexF64, Px, Py, No, 3)
        if b.Tz !== nothing
            Ez = zeros(ComplexF64, Px, Py, b.Lz, 3)
            _zmul_add!(Ez, b.Tz, zspec(b.Lz, false))
            b.Hz === nothing || _zmul_add!(Ez, b.Hz, zspec(b.Lz, true))
            bfft!(Ez, 3)
            @views Ê .+= Ez[:, :, 1:No, :]
        end
        b.D === nothing || _dense_add!(Ê, b.D, Ĵ2)
        _plan(:bfft, Ê) * Ê
        out[b.obs_layer] = Ê[1:g.Nx, 1:g.Ny, :, :]
    end
    return out
end

"""
    dipole_moments(op, dips) -> P

Moments array for `apply_operator`: each dipole snapped to its nearest pixel
of `op.grid` and to its source plane, which must be one of `op.z_src`.
"""
function dipole_moments(op::LayerGreenOperator, dips::AbstractVector{<:Dipole})
    g = op.grid
    P = zeros(ComplexF64, g.Nx, g.Ny, length(op.z_src), 3)
    tol = 1e-9 * max(1.0, maximum(abs, op.z_src))
    for d in dips
        d.layer == op.source_layer || error("dipole_moments: dipole in layer $(d.layer), operator source layer $(op.source_layer)")
        k = findfirst(z -> abs(z - d.z) <= tol, op.z_src)
        k === nothing && error("dipole_moments: dipole depth $(d.z) is not one of the operator's source planes")
        i, j = _nearest_node_index(g, d.x, d.y)
        @views P[i, j, k, :] .+= d.p
    end
    return P
end

"Memory and build time per block."
function operator_summary(op::LayerGreenOperator; io::IO=stdout)
    tot = 0
    for b in op.blocks
        bytes = sum(x === nothing ? 0 : sizeof(x) for x in (b.Tz, b.Hz, b.D))
        tot += bytes
        @printf(io, "  source layer %d → obs layer %d: %-15s %3d obs planes  %8.1f MB  built in %.2f s\n",
                op.source_layer, b.obs_layer, b.kind, length(b.z_obs), bytes / 2^20, op.timings[b.obs_layer])
    end
    @printf(io, "  total %.1f MB\n", tot / 2^20)
end

end
