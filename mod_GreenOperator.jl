"""
mod_GreenOperator.jl -- precomputed, z-structured Green's operator of one
source layer onto a set of observation layers, on uniform z grids.

The user declares the source layer (its `CavityTransferCache`), the uniform
grid of source planes in it, and the uniform grid of observation planes in
each wanted observation layer. Everything grid-dependent is precomputed;
`apply_operator` then does only FFTs, pointwise 3×3 products and (for dense
and skeleton blocks) plane-by-plane products, all in k-space.

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
    Toeplitz (homogeneous layer); the scattered part has no z structure.
  - cross layer: no z structure either.

  Blocks with no z structure are stored on SKELETON planes (`skeleton=true`).
  In a homogeneous layer every node's z-dependence is a plane wave `e^{±iQz}`
  set by the layer alone (the stack only sets the amplitudes), so the field on
  all planes is interpolated from a few planes, `G(z'_j, ·) = Σ_p Po[j, p]·
  G(z'_{o_p}, ·)`, with the same `Po` for every node (an interpolative
  decomposition of the layer's exponential family, `_zskeleton`, checked on
  every node to `skeleton_tol`); likewise on the source side,
  `Σ_i G(·, z_i)·J_i = Σ_q G(·, z_{s_q})·(Psᵀ J)_q`. Only the skeleton pairs
  are evaluated and stored; the apply folds the source planes (`Psᵀ`, one GEMM),
  runs the pair loop on the skeleton planes and interpolates (`Po`, one GEMM).
  A side that cannot be compressed (≥ 80% of its planes, e.g. a hyperbolic
  layer, where high-kρ waves propagate) is kept whole; with both sides whole
  the block is the plain dense one.

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

export LayerGreenOperator, build_layer_operator, build_layer_operator_grouped, apply_operator, dipole_moments, operator_summary
export ZSkeleton, z_skeletons, z_skeletons_grouped

import ..ModeSolver: _is_azimuthally_symmetric_stack, _modes_layer, IsotropicMaterial, UniaxialMaterial, AbstractLayer
import ..CavityTransfer: AbstractCavityTransferCache, CavityTransferCache1D, CavityTransferCache2D, build_cavity_transfer_2d
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
`:structured` (`Tz`, `Hz` set), `:toeplitz_dense` (`Tz`, `D` set), `:dense` (`D` set),
`:skeleton` (`D` on skeleton planes, `Ps` and/or `Po` set), `:toeplitz_skeleton`
(`Tz`, and the scattered part as `:skeleton`).
`Tz`/`Hz` are `(Px, Py, Lz, 3, 3)`, already FFT'd along z (with `1/Lz` and, for
`Hz`, the output shift folded in). `D` is `(Px, Py, 3, 3, Ns, No)`, or
`(Px, Py, 3, 3, R_s, R_o)` on the skeleton planes, with `D_full[.., i, j] ≈
Σ_{q,p} Ps[i, q]·Po[j, p]·D[.., q, p]`; `Ps`/`Po` = `nothing` on a side kept whole.
"""
struct ZBlock
    obs_layer :: Int
    z_obs     :: Vector{Float64}
    kind      :: Symbol
    Lz        :: Int
    Tz        :: Union{Nothing,Array{ComplexF64,5}}
    Hz        :: Union{Nothing,Array{ComplexF64,5}}
    D         :: Union{Nothing,Array{ComplexF64,6}}
    Ps        :: Union{Nothing,Matrix{ComplexF64}}
    Po        :: Union{Nothing,Matrix{ComplexF64}}
end
ZBlock(o, zo, kind, Lz, Tz, Hz, D) = ZBlock(o, zo, kind, Lz, Tz, Hz, D, nothing, nothing)

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
#  z skeletons (cross-layer and biaxial-scattered blocks)
# ─────────────────────────────────────────────────────────────────────────────

"""
Interpolative decomposition of one layer's z-exponential family on its planes:
`F ≈ P·F[idx, :]` for the column of EVERY quadrature node and mode, with
`max |F − P·F[idx, :]| = err`. The entries of `F` are `e^{iQ·(distance into the
layer)}`, all of modulus ≤ 1, so `err` bounds each node term absolutely. `idx`
increasing, `P[idx, :] = I`, `pnorm = ‖P‖₂`. A side kept whole has `idx = 1:N`,
`P = I` (`whole = true`).
"""
struct ZSkeleton
    idx   :: Vector{Int}
    P     :: Matrix{ComplexF64}
    err   :: Float64
    pnorm :: Float64
    whole :: Bool
end

const _FAM_CHUNK = 8192

"Rows `e^{iQ_k(σ·z_j + c)}`, `k ∈ r`, as a `length(r) × N` block (node × plane); underflow to exact 0."
function _family_rows(Q::Vector{ComplexF64}, σ::Float64, c::Float64, z::Vector{Float64}, r)
    A = Matrix{ComplexF64}(undef, length(r), length(z))
    @inbounds for (j, zj) in enumerate(z), (m, k) in enumerate(r)
        x = im * Q[k] * (σ * zj + c)
        A[m, j] = real(x) < -700 ? zero(ComplexF64) : exp(x)
    end
    return A
end

"The `(term, node range)` chunks of the family `fam = [(Q, σ, c), ...]`."
_family_chunks(fam) = [(t, r) for (t, (Q, _, _)) in enumerate(fam) for r in Iterators.partition(eachindex(Q), _FAM_CHUNK)]

"""
`op(acc, A)` folded over every `(chunk × N)` row block `A` of the family, one
accumulator per task from `init()`, threaded over chunks; returns the accumulators.
"""
function _fold_chunks(op, init, fam, z)
    chunks = _family_chunks(fam)
    nt = min(Threads.nthreads(), max(length(chunks), 1))
    parts = collect(Iterators.partition(eachindex(chunks), cld(max(length(chunks), 1), nt)))
    accs = [init() for _ in parts]
    @sync for (a, part) in enumerate(parts)
        Threads.@spawn for i in part
            t, r = chunks[i]
            Q, σ, c = fam[t]
            accs[a] = op(accs[a], _family_rows(Q, σ, c, z, r))
        end
    end
    return accs
end

"Distinct finite non-zero values of node arrays (of numbers or 2-vectors); invalid nodes are 0."
function _qlist(xs...)
    v = ComplexF64[]
    for x in xs, q in x, e in q
        (iszero(e) || !isfinite(e)) || push!(v, e)
    end
    return unique!(v)
end

"""
    _zskeleton(z, fam; tol, whole_frac=0.8) -> ZSkeleton

`fam`: family terms `(Q, σ, c)`, one column `e^{iQ(σz + c)}` per entry of `Q`.
Streaming QR of `Fᵀ` (nodes × planes) down to its `N × N` factor `R` (`Fᵀ = Q₀R`:
a column ID of `R` is one of `Fᵀ`, same error), column-pivoted QR of `R`, then
the smallest rank `k` whose interpolation error, checked on every node, is
`≤ tol` (bisection: `k = N` is exact). `k ≥ whole_frac·N`: the side is kept whole.
"""
function _zskeleton(z::Vector{Float64}, fam; tol::Real, whole_frac::Real=0.8)
    nb = BLAS.get_num_threads()
    BLAS.set_num_threads(1)              # the chunk loops are threaded: nested BLAS threads are ≈ 10× slower
    try
        return _zskeleton_serial_blas(z, fam; tol=tol, whole_frac=whole_frac)
    finally
        BLAS.set_num_threads(nb)
    end
end

function _zskeleton_serial_blas(z::Vector{Float64}, fam; tol::Real, whole_frac::Real)
    N = length(z)
    whole = ZSkeleton(collect(1:N), Matrix{ComplexF64}(I, N, N), 0.0, 1.0, true)
    N <= 1 && return whole
    # TSQR: one triangular factor per task, then the factor of the stacked factors
    stackR(R, A) = Matrix(qr!(vcat(R, A)).R)
    Rs = _fold_chunks(stackR, () -> zeros(ComplexF64, 0, N), fam, z)
    R = foldl(stackR, Rs; init=zeros(ComplexF64, 0, N))
    size(R, 1) < N && (R = vcat(R, zeros(ComplexF64, N - size(R, 1), N)))
    F = qr(R, ColumnNorm())
    p, Rt = F.p, F.R
    function interp(k)
        P = zeros(ComplexF64, N, k)
        for q in 1:k
            P[p[q], q] = 1
        end
        k < N && (P[p[k+1:N], :] .= transpose(UpperTriangular(Rt[1:k, 1:k]) \ Rt[1:k, k+1:N]))
        perm = sortperm(p[1:k])                       # skeleton planes in increasing order
        return p[1:k][perm], P[:, perm]
    end
    maxerr(idx, P) = maximum(_fold_chunks((e, A) -> max(e, maximum(abs, mul!(A, A[:, idx], transpose(P), -1, 1))),
                                          () -> 0.0, fam, z))
    lo, hi = 0, N
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        maxerr(interp(mid)...) <= tol ? (hi = mid) : (lo = mid)
    end
    hi >= whole_frac * N && return whole
    idx, P = interp(hi)
    return ZSkeleton(idx, P, maxerr(idx, P), opnorm(P), false)
end

"""
Observation-side family of layer `o` (local `z'`, as `_obs_phases`): `e^{iKZf·z'}`
and `e^{iKZb·(z'−d)}`. Layer 1 carries the downward wave only and the last layer
the upward one only (the other rows of `T^o` are exactly 0 there).
"""
function _obs_family(KZf, KZb, d, o, nlayers)
    fam = Tuple{Vector{ComplexF64},Float64,Float64}[]
    o == 1 || push!(fam, (_qlist(KZf), 1.0, 0.0))
    o == nlayers || push!(fam, (_qlist(KZb), 1.0, -Float64(d)))
    return fam
end

"""
Source-side family of layer `s` (cavity-frame `zpos`, as `_seed_phases`):
`Ψf = e^{iQf(d−zpos)}`, `Ψb = e^{−iQb·zpos}`; a source in layer 1 feeds only through
`Ψf` (the `Ψb` columns of `T^o` are 0). The box factor `D(Q)` and the seeds are
per-node constants and do not enter.
"""
function _src_family(Qf, Qb, d, s)
    fam = [(_qlist(Qf), -1.0, Float64(d))]
    s == 1 || push!(fam, (_qlist(Qb), -1.0, 0.0))
    return fam
end

"""
    z_skeletons(Qsrc, Qobs, stack, s, zs_int, zobs, layers; tol) -> (src=ZSkeleton, obs=Dict)

`Qsrc = (Qf, Qb)` of source layer `s` and `Qobs[o] = (KZf, KZb)` for `o ∈ layers`,
on every quadrature node (any arrays of numbers or 2-vectors). `zs_int`: the
source planes in increasing `zpos` order (as the operator stores them).
"""
function z_skeletons(Qsrc, Qobs, stack, s, zs_int, zobs, layers; tol::Real)
    d = stack[s].d
    zpos = s == 1 ? d .- collect(Float64, zs_int) : collect(Float64, zs_int)
    src = _zskeleton(zpos, _src_family(Qsrc..., d, s); tol=tol)
    obs = Dict{Int,ZSkeleton}()
    for o in layers
        obs[o] = _zskeleton(collect(Float64, zobs[o]), _obs_family(Qobs[o]..., stack[o].d, o, length(stack)); tol=tol)
    end
    return (src=src, obs=obs)
end

_cache_skeletons(cache, zs_int, zobs; tol) =
    z_skeletons((cache.Qf_src, cache.Qb_src), Dict(o => (cache.basis[o].KZf, cache.basis[o].KZb) for o in keys(zobs)),
                cache.stack, cache.source_layer, zs_int, zobs, keys(zobs); tol=tol)

"`Qf`, `Qb` (2 × n) of one layer at in-plane wavevectors `kts`, from the layer's own mode solve (no stack solve)."
function _layer_Q(layer, ω, kts)
    Qf = zeros(ComplexF64, 2, length(kts)); Qb = zeros(ComplexF64, 2, length(kts))
    @threads for n in eachindex(kts)
        try
            mf, mb = _modes_layer(layer, ω, kts[n]...)
            Qf[:, n] .= mf[3]; Qb[:, n] .= mb[3]
        catch e
            e isa Union{LinearAlgebra.SingularException, ErrorException, DomainError} || rethrow()
        end
    end
    return Qf, Qb
end

"""
    z_skeletons_grouped(stack, ω, source_layer, groups, z_src, z_obs_by_layer; tol=1e-6)

The skeletons of `build_layer_operator_grouped`: one per side for ALL groups (the
blocks are summed over groups, so they must share their skeleton planes), from
each layer's `Q` at every `(kρ, φ)` node of every group (φ = 0 only on an
azimuthally symmetric stack, where `Q` depends on kρ alone).
"""
function z_skeletons_grouped(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int, groups::AbstractVector,
                             z_src::AbstractVector{<:Real}, z_obs_by_layer::AbstractDict; tol::Real=1e-6)
    sym = _is_azimuthally_symmetric_stack(stack)
    kts = [(kρ * cos(φ), kρ * sin(φ)) for g in groups for φ in (sym ? (0.0,) : g.φ) for kρ in g.kρ]
    sym && unique!(kts)
    zs = collect(Float64, z_src)
    zs_int = source_layer == 1 ? reverse(zs) : zs
    layers = collect(keys(z_obs_by_layer))
    Q = Dict(l => _layer_Q(stack[l], Float64(ω), kts) for l in union(layers, [source_layer]))
    return z_skeletons(Q[source_layer], Dict(o => Q[o] for o in layers), stack, source_layer, zs_int, z_obs_by_layer, layers; tol=tol)
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
`:dense` (no z structure anywhere: `D` for everything, as a reference).

`skeleton=true`: the blocks with no Toeplitz/Hankel structure (cross layer, and
the scattered part of a biaxial same-layer block) are stored on skeleton planes
(`:skeleton`, `:toeplitz_skeleton`), each side interpolated from them to
`skeleton_tol` per node term (`_zskeleton`). A side whose skeleton would hold
≥ 80% of its planes is kept whole; with both sides whole the block stays dense.
`skeletons`: precomputed `z_skeletons` (used by `build_layer_operator_grouped`, whose
groups must share them); by default they are built from this cache's own `Q`s.
"""
function build_layer_operator(cache::AbstractCavityTransferCache, grid::NamedTuple, z_src::AbstractVector{<:Real},
                              z_obs_by_layer::AbstractDict{Int,<:AbstractVector{<:Real}};
                              structure::Symbol=:auto, check::Bool=true, check_tol::Real=1e-8,
                              skeleton::Bool=true, skeleton_tol::Real=1e-6, skeletons=nothing, kwargs...)
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
    use_sk = skeleton && structure !== :dense
    skel() = (skeletons === nothing && (skeletons = _cache_skeletons(cache, zs_int, z_obs_by_layer; tol=skeleton_tol)); skeletons)

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
                    push!(blocks, _toeplitz_dense_block(cache, o, zs_int, zo, grid, Ns, No, use_sk ? skel() : nothing; kw...))
                end
            else
                push!(blocks, _dense_block(cache, o, zs_int, zo, grid, Ns, No, use_sk ? skel() : nothing; terms=:all, kw...))
            end
        end
        timings[o] = t
    end
    return LayerGreenOperator(s, grid, zs, flip, blocks, kw, timings)
end

"""
    build_layer_operator_grouped(stack, ω, source_layer, groups, grid, z_src, z_obs_by_layer;
                                 node_budget=900_000, kwargs...) -> (op, t_cache, t_build)

The operator on a φ-dependent kρ grid (`ClosedFormGrid.box_kρ_grid_grouped`):
one `CavityTransferCache2D` + `build_layer_operator` per group (`(kρ, wρ, φ, wφ)`),
summed block by block -- exact, since every block is linear in the quadrature
sum. A group with more than `node_budget` kρ × φ nodes is split into φ chunks
(the cache of a 256-φ biaxial group does not fit in 16 GB with the build).
`kwargs` go to `build_layer_operator`; every group must choose the same block
kinds (true for the media rule, the structure check is skipped after the first
group). The z skeletons are built once, on the nodes of all groups
(`z_skeletons_grouped`), before the first cache. `op.timings` holds the per-block
build time summed over groups.
"""
function build_layer_operator_grouped(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int, groups::AbstractVector,
                                      grid::NamedTuple, z_src::AbstractVector{<:Real},
                                      z_obs_by_layer::AbstractDict{Int,<:AbstractVector{<:Real}};
                                      node_budget::Int=900_000, kwargs...)
    obs = sort(collect(keys(z_obs_by_layer)))
    op = nothing; t_cache = 0.0; t_build = 0.0
    sk = get(kwargs, :skeletons, nothing)
    if sk === nothing && get(kwargs, :skeleton, true) && get(kwargs, :structure, :auto) !== :dense
        t_build += @elapsed sk = z_skeletons_grouped(stack, ω, source_layer, groups, z_src, z_obs_by_layer;
                                                     tol=get(kwargs, :skeleton_tol, 1e-6))
    end
    for g in groups
        chunk = clamp(node_budget ÷ max(length(g.kρ), 1), 1, length(g.φ))
        for r in Iterators.partition(eachindex(g.φ), chunk)
            t_cache += @elapsed c = build_cavity_transfer_2d(stack, ω, source_layer, g.kρ, g.wρ, g.φ[r], g.wφ[r]; obs_layers=obs)
            t_build += @elapsed opc = build_layer_operator(c, grid, z_src, z_obs_by_layer; kwargs..., skeletons=sk,
                                                           check=(op === nothing && get(kwargs, :check, true)))
            c = nothing
            if op === nothing
                op = opc
            else
                for (b, bc) in zip(op.blocks, opc.blocks)
                    b.kind == bc.kind || error("build_layer_operator_grouped: groups chose different block kinds ($(b.kind) vs $(bc.kind))")
                    (b.Ps === bc.Ps && b.Po === bc.Po) || error("build_layer_operator_grouped: groups have different z skeletons")
                    for f in (:Tz, :Hz, :D)
                        x = getfield(b, f)
                        x === nothing || (x .+= getfield(bc, f))
                    end
                    op.timings[b.obs_layer] += opc.timings[b.obs_layer]
                end
            end
            opc = nothing; GC.gc()
        end
    end
    return op, t_cache, t_build
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

"""
`D` of one block, on the skeleton planes of each side that is not kept whole:
`(D, Ps, Po, skeletal)`; `skeletal = false` (full `D`, both `nothing`) without
skeletons or with both sides whole.
"""
function _planes_maybe_skeleton(cache, o, zs, zo, grid, Ns, No, sk; terms, kw...)
    (sk === nothing || (sk.src.whole && sk.obs[o].whole)) &&
        return _dense_planes(cache, o, zs, zo, grid, Ns, No; terms=terms, kw...), nothing, nothing, false
    s_, o_ = sk.src, sk.obs[o]
    D = _dense_planes(cache, o, zs[s_.idx], zo[o_.idx], grid, length(s_.idx), length(o_.idx); terms=terms, kw...)
    return D, s_.whole ? nothing : s_.P, o_.whole ? nothing : o_.P, true
end

function _dense_block(cache, o, zs, zo, grid, Ns, No, sk=nothing; terms, kw...)
    D, Ps, Po, skl = _planes_maybe_skeleton(cache, o, zs, zo, grid, Ns, No, sk; terms=terms, kw...)
    return ZBlock(o, zo, skl ? :skeleton : :dense, 0, nothing, nothing, D, Ps, Po)
end

function _toeplitz_dense_block(cache, o, zs, zo, grid, Ns, No, sk=nothing; kw...)
    Lz = _zlen(Ns, No)
    Tz = _toeplitz_z(cache, o, zs, zo, grid, Ns, No, Lz; terms=:direct, kw...)
    D, Ps, Po, skl = _planes_maybe_skeleton(cache, o, zs, zo, grid, Ns, No, sk; terms=:scattered, kw...)
    return ZBlock(o, zo, skl ? :toeplitz_skeleton : :toeplitz_dense, Lz, Tz, nothing, D, Ps, Po)
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

"`X[.., q, c] = Σ_i Ĵ[.., i, c]·Ps[i, q]`: the source planes folded onto the skeleton planes (one GEMM per component)."
function _zfold(Ĵ::Array{ComplexF64,4}, Ps::Matrix{ComplexF64})
    Px, Py, Ns, _ = size(Ĵ)
    X = Array{ComplexF64,4}(undef, Px, Py, size(Ps, 2), 3)
    A, B = reshape(Ĵ, Px * Py, Ns, 3), reshape(X, Px * Py, size(Ps, 2), 3)
    for c in 1:3
        mul!(view(B, :, :, c), view(A, :, :, c), Ps)
    end
    return X
end

"`Ê[.., j, r] += Σ_p Po[j, p]·Ês[.., p, r]`: the skeleton observation planes interpolated to all planes."
function _zexpand_add!(Ê::Array{ComplexF64,4}, Ês::Array{ComplexF64,4}, Po::Matrix{ComplexF64})
    Px, Py, No, _ = size(Ê)
    A, B = reshape(Ês, Px * Py, size(Po, 2), 3), reshape(Ê, Px * Py, No, 3)
    for r in 1:3
        mul!(view(B, :, :, r), view(A, :, :, r), transpose(Po), true, true)
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
    folded = IdDict{Matrix{ComplexF64},Array{ComplexF64,4}}()      # one fold per source skeleton

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
        if b.D !== nothing
            Ĵs = b.Ps === nothing ? Ĵ2 : get!(() -> _zfold(Ĵ2, b.Ps), folded, b.Ps)
            if b.Po === nothing
                _dense_add!(Ê, b.D, Ĵs)
            else
                Ês = zeros(ComplexF64, Px, Py, size(b.Po, 2), 3)
                _dense_add!(Ês, b.D, Ĵs)
                _zexpand_add!(Ê, Ês, b.Po)
            end
        end
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
        bytes = sum(x === nothing ? 0 : sizeof(x) for x in (b.Tz, b.Hz, b.D, b.Ps, b.Po))
        tot += bytes
        sk = b.kind in (:skeleton, :toeplitz_skeleton) ?
             @sprintf("  skeleton %d × %d of %d × %d planes", size(b.D, 5), size(b.D, 6), length(op.z_src), length(b.z_obs)) : ""
        @printf(io, "  source layer %d → obs layer %d: %-17s %3d obs planes  %8.1f MB  built in %.2f s%s\n",
                op.source_layer, b.obs_layer, b.kind, length(b.z_obs), bytes / 2^20, op.timings[b.obs_layer], sk)
    end
    @printf(io, "  total %.1f MB\n", tot / 2^20)
end

end
