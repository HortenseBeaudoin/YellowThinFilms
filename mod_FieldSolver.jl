module FieldSolver

export ModalAmp, Modal_amp, Modal_amp_profile
export _cavity_transfer_columns

import ..ModeSolver: AbstractLayer, Modes
import ..ScatteringMatrix: ScatteringM, _SProp, _Redheffer
import ..SourceHandling: SourceDef, ModalSeed


using LinearAlgebra
using StaticArrays


# ─────────────────────────────────────────────────────────────────────────────
#  Constant and structures declaration
# ─────────────────────────────────────────────────────────────────────────────

struct ModalAmp{T}
    Ef::SVector{2,T}
    Eb::SVector{2,T}
end

const _DENORMAL_FLUSH_FLOOR = 1e-250
_flush_denormal(x::Complex) = abs(x) < _DENORMAL_FLUSH_FLOOR ? zero(x) : x
_flush_denormal(v::AbstractArray) = _flush_denormal.(v)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────


function _cond2x2(A::SMatrix{2,2})
    G = A' * A
    tr   = real(G[1,1] + G[2,2])
    det_ = real(G[1,1] * G[2,2] - G[1,2] * G[2,1])
    disc = sqrt(max(tr^2 - 4 * det_, 0.0))
    λ1 = (tr + disc) / 2
    λ2 = (tr - disc) / 2
    λ2 <= 0 && return Inf
    return sqrt(λ1 / λ2)
end


function _SProp(d::Real, Q::AbstractVector)
    φ1 = exp(im * Q[1] * d)
    φ2 = exp(im * Q[2] * d)
    Tc = promote_type(typeof(φ1), typeof(φ2))
    return SVector{2,Tc}(φ1, φ2)
end


"Extracts `(z, sif, sib)` from one source (`ModalSeed`: Modal_amp-convention seeds)."
_seed(s::ModalSeed) = (s.z, SVector{2,ComplexF64}(s.sif[1], s.sif[2]), SVector{2,ComplexF64}(s.sib[1], s.sib[2]))


"""
Reflection matrices for `stack[layer]` as seen from its own TRUE edges (z=0
and z=d).
"""
function _R_cavity(stack::Vector{<:AbstractLayer}, layer::Int,
                           modes::Vector{<:Modes}, Sall::Vector{<:ScatteringM})
    N  = length(stack)
    d  = stack[layer].d
    Qf, Qb = modes[layer].Qf, modes[layer].Qb

    RL = SMatrix{2,2,ComplexF64}(Sall[layer].SLeft[SOneTo(2), SVector(3,4)])

    if layer == N
        return RL, zero(SMatrix{2,2,ComplexF64}), nothing, Qf, Qb, d
    end

    SΦ = SMatrix{4,4,ComplexF64}(Diagonal(vcat(_SProp(-d, Qf), _SProp(d, Qb))))
    RR_right   = _Redheffer(SΦ, Sall[layer].SRight)
    RR = SMatrix{2,2,ComplexF64}(RR_right[SVector(3,4), SOneTo(2)])

    return RL, RR, RR_right, Qf, Qb, d
end



#TODO Sources are the arbitrary sources here, right ? If so, this could be simplify to have this as a default
"""
    _seed_from_sources(sources, layer, N, Qf, Qb, d) -> (S₊, S₋, seeds)

Collapses every source to the layer's own two true edges: `S₊` (forward seed,
referenced at z=d) and `S₋` (backward seed, referenced at z=0), plus the
per-source `(zpos, sif, sib)` list `Modal_amp_profile` needs for the direct
(within-layer) term. This is the ONLY place a source's `z` enters the solve:
each source contributes `sif .* _SProp(d-zpos,Qf)` to `S₊` and
`sib .* _SProp(-zpos,Qb)` to `S₋`; everything downstream
(`_cavity_resummation`, `_walk_stack`) is linear in `(S₊,S₋)` and
z-independent. Split out of `_cavity_solve` (2026-09-23) unchanged.
"""
function _seed_from_sources(sources::Vector{<:SourceDef}, layer::Int, N::Int,
                            Qf::AbstractVector, Qb::AbstractVector, d::Real)
    S₊  = zero(SVector{2,ComplexF64})   # forward seed, z=d
    S₋ = zero(SVector{2,ComplexF64})   # backward seed, z=0
    seeds = Vector{Tuple{Float64,SVector{2,ComplexF64},SVector{2,ComplexF64}}}(undef, length(sources))
    for (i, s) in enumerate(sources)
        z, sif, sib = _seed(s)
        @assert layer < N || iszero(sif) "layer==N (semi-infinite) sources cannot have a forward (sif) component -- nothing reflects it back from infinity"

        zpos = layer == 1 ? d - z : z
        seeds[i] = (zpos, sif, sib)

        if layer < N
            φf = _SProp(d-zpos, Qf)
            S₊ = S₊ + sif .* φf
        end

        φb = _SProp(-zpos, Qb)
        S₋ = S₋ + sib .* φb
    end
    return S₊, S₋, seeds
end


"""
    _cavity_resummation(S₊, S₋, RL, RR, Qf, Qb, d, N, layer) -> (F⁺, F⁻)

The linear, z-independent part of the cavity solve: resums the multiple
reflections between the layer's two edges,
`F⁺ = (I - ΦF·RL·ΦB·RR)⁻¹ (S₊ + ΦF·RL·S₋)`, `F⁻ = S₋ + ΦB·RR·F⁺`.

Generic over the seed's shape: `S₊,S₋` may be the physical `SVector{2}` seeds
(what `_cavity_solve` passes) or `SMatrix{2,n}` blocks of basis inputs (what
`_cavity_transfer_columns` passes, one column per unit basis vector) -- the
same arithmetic runs on every column, so the matrix form IS the linear map.
"""
function _cavity_resummation(S₊::StaticArray, S₋::StaticArray,
                             RL::SMatrix{2,2}, RR::SMatrix{2,2},
                             Qf::AbstractVector, Qb::AbstractVector, d::Real, N::Int, layer::Int)
    layer == N && return zero(S₋), S₋
    I_2x2 = SMatrix{2,2,ComplexF64}(I)
    ΦF = Diagonal(_SProp(d, Qf))
    ΦB = Diagonal(_SProp(-d, Qb))
    F⁺ = (I_2x2 - ΦF * RL * ΦB * RR) \ (S₊ + ΦF * (RL * S₋))
    F⁻ = S₋ + ΦB * (RR * F⁺)
    return F⁺, F⁻
end


"""
Derives the layer's two-sided cavity (F⁺, F⁻) from `sources`, plus everything needed to
either walk the rest of the stack (`Modal_amp`) or reconstruct the field
strictly WITHIN `layer` at an arbitrary point.

= `_seed_from_sources` (z-dependent seed) + `_cavity_resummation` (linear,
z-independent) -- factored 2026-09-23 so the linear part can be reused on unit
basis inputs by `_cavity_transfer_columns`; behavior unchanged.
"""
function _cavity_solve(stack::Vector{<:AbstractLayer}, sources::Vector{<:SourceDef}, layer::Int,
                                 modes::Vector{<:Modes}, Sall::Vector{<:ScatteringM})
    N = length(stack)
    @assert !isempty(sources) "sources must be non-empty"
    @assert 1 <= layer <= N "layer must be a valid stack index"

    RL, RR, RR_right, Qf, Qb, d = _R_cavity(stack, layer, modes, Sall)

    S₊, S₋, seeds = _seed_from_sources(sources, layer, N, Qf, Qb, d)
    F⁺, F⁻ = _cavity_resummation(S₊, S₋, RL, RR, Qf, Qb, d, N, layer)

    return F⁺, F⁻, RL, RR, RR_right, Qf, Qb, d, N, seeds
end


"""
    _walk_stack(stack, layer, F⁺, F⁻, RL, RR_right, modes, Sall, Sint; tol, layers)
        -> (Ef_tot, Eb_tot)

The Redheffer walk from the source layer's cavity amplitudes `(F⁺,F⁻)` to
every other layer, plus the final re-referencing of `Eb` to each layer's own
z=d. Extracted verbatim from `Modal_amp` (2026-09-23). Linear in `(F⁺,F⁻)`
(the `_cond2x2` guard and `_flush_denormal` depend on `Sall` / on magnitudes
at the denormal floor only), and generic over their shape like
`_cavity_resummation`: `SVector{2}` for a physical solve, `SMatrix{2,n}` for
basis columns.
"""
function _walk_stack(stack::Vector{<:AbstractLayer}, layer::Int, F⁺::StaticArray, F⁻::StaticArray,
                     RL::SMatrix{2,2}, RR_right, modes::Vector{<:Modes},
                     Sall::Vector{<:ScatteringM}, Sint::Vector{<:SMatrix{4,4}};
                     tol::Number = 1E-8, layers::Union{Nothing,AbstractSet{Int}} = nothing)
    N = length(stack)
    Z = zero(F⁺)

    Ef_tot = fill(Z, N)
    Eb_tot = fill(Z, N)
    Ef_left = RL * F⁻   # field at z=0 of `layer` itself

    # Required calculation
    want_layer_itself = layers === nothing || layer in layers
    need_right = layers === nothing || any(>(layer), layers) ||
                 (want_layer_itself && layer < N)
    need_left  = layers === nothing || any(<(layer), layers)

    # -------- Walk RIGHT, seeded by F⁺ ------------------
    if layer < N && need_right
        S11 = RR_right[SOneTo(2), SOneTo(2)]; S12 = RR_right[SOneTo(2), SVector(3,4)]

        c₋ = Z   # nothing incident from +infinity
        c₊ = S11 * F⁺ + S12 * c₋

        Ef_tot[end] = Ef_tot[end] + _flush_denormal(c₊)
        Eb_tot[end] = Eb_tot[end] + _flush_denormal(c₋)

        for k in (layer + 1):(N - 1)
            Sk = Sall[k].SRight
            S11k = Sk[SOneTo(2), SOneTo(2)]; S12k = Sk[SOneTo(2), SVector(3,4)]
            S21k = Sk[SVector(3,4), SOneTo(2)]; S22k = Sk[SVector(3,4), SVector(3,4)]

            if _cond2x2(S11k) > 1 / tol
                b₊ = Z
            else
                b₊ = S11k \ (c₊ - S12k * c₋)
            end
            b₋ = S21k * b₊ + S22k * c₋

            Ef_tot[k] = Ef_tot[k] + _flush_denormal(b₊)
            Eb_tot[k] = Eb_tot[k] + _flush_denormal(b₋)
        end
    end

    # -------- Walk LEFT, seeded by F⁻ -------------------

    if layer > 1 && need_left
        Sj = Sall[layer].SLeft
        S22 = Sj[SVector(3,4), SVector(3,4)]

        a₊ = Z   # nothing beyond layer 1
        a₋ = S22 * F⁻

        Ef_tot[1] = Ef_tot[1] + _flush_denormal(a₊)
        Eb_tot[1] = Eb_tot[1] + _flush_denormal(a₋)

        for k in (layer - 1):-1:2
            Sk = Sall[k].SLeft
            S11k = Sk[SOneTo(2), SOneTo(2)]; S12k = Sk[SOneTo(2), SVector(3,4)]
            S21k = Sk[SVector(3,4), SOneTo(2)]; S22k = Sk[SVector(3,4), SVector(3,4)]

            if _cond2x2(S22k) > 1 / tol
                b₋ = Z
            else
                b₋ = S22k \ (a₋ - S21k * a₊)
            end
            b₊ = S11k * a₊ + S12k * b₋

            Ef_tot[k] = Ef_tot[k] + _flush_denormal(b₊)
            Eb_tot[k] = Eb_tot[k] + _flush_denormal(b₋)
        end
    end

    # -------- Re-reference Eb to the file's global z=d-of-each-layer

    for k in 1:(N - 1)
        Ef_far = if k == layer
            F⁺
        else
            φf_k = _SProp(stack[k].d, modes[k].Qf)
            Ef_tot[k] .* φf_k
        end

        Eb_next = k == layer - 1 ? F⁻ : Eb_tot[k+1]

        Sk = Sint[k]
        S21k = Sk[SVector(3,4), SOneTo(2)]
        S22k = Sk[SVector(3,4), SVector(3,4)]

        Eb_tot[k] = _flush_denormal(S21k * Ef_far + S22k * Eb_next)
    end

    Ef_tot[layer] = _flush_denormal(Ef_left)

    if layer == N
        Eb_tot[N] = _flush_denormal(F⁻)
    end

    return Ef_tot, Eb_tot
end


# Unit basis inputs for `_cavity_transfer_columns`: column j of [S₊;S₋] is e_j.
const _E₊ = SMatrix{2,4,ComplexF64}(1,0, 0,1, 0,0, 0,0)
const _E₋ = SMatrix{2,4,ComplexF64}(0,0, 0,0, 1,0, 0,1)

"""
    _cavity_transfer_columns(stack, layer, modes, Sall, Sint; obs_layers, tol=1E-8)
        -> Dict{Int, SMatrix{4,4,ComplexF64,16}}

The z-independent linear map `T^o` from the source layer's edge seeds
`[S₊; S₋]` (see `_seed_from_sources`) to the modal amplitudes `[Ef; Eb]` of
each observation layer `o ∈ obs_layers`:

    [Ef_o; Eb_o] = T^o · [S₊; S₋]

obtained by running `_cavity_resummation` + `_walk_stack` on the 4 unit basis
inputs at once (exact, since both are linear in `(S₊,S₋)`). Reference
conventions match `Modal_amp`: `Ef_o` at z=0 of layer `o`, `Eb_o` at z=d.

For `o == layer` (same layer) the columns are the REFLECTED (cavity) part only,
`[RL·F⁻; RR·F⁺]` -- exactly `Modal_amp_profile`'s `Ef_j0`/`Eb_d`; the direct
term (`Σ sif·e^{-iQf zpos}` below / `sib·e^{iQb(d-zpos)}` above the source) is
not part of this map and is added by the caller. For `o ≠ layer` the columns
are `Modal_amp`'s `Ef_tot[o]`/`Eb_tot[o]`.

Added 2026-09-23 for the closed-form (z,z')-separable rewrite
(`PLAN_closed_form_point_source_rewrite.md` §3, option 1).
"""
function _cavity_transfer_columns(stack::Vector{<:AbstractLayer}, layer::Int,
                                  modes::Vector{<:Modes}, Sall::Vector{<:ScatteringM},
                                  Sint::Vector{<:SMatrix{4,4}};
                                  obs_layers = 1:length(stack), tol::Number = 1E-8)
    N = length(stack)
    @assert 1 <= layer <= N "layer must be a valid stack index"
    RL, RR, RR_right, Qf, Qb, d = _R_cavity(stack, layer, modes, Sall)

    S₊ = layer < N ? _E₊ : zero(_E₊)   # layer==N: no forward seed can exist
    F⁺, F⁻ = _cavity_resummation(S₊, _E₋, RL, RR, Qf, Qb, d, N, layer)

    others = Set(o for o in obs_layers if o != layer)
    Ef_tot, Eb_tot = isempty(others) ? (nothing, nothing) :
        _walk_stack(stack, layer, F⁺, F⁻, RL, RR_right, modes, Sall, Sint; tol=tol, layers=others)

    T = Dict{Int,SMatrix{4,4,ComplexF64,16}}()
    for o in obs_layers
        Ef, Eb = o == layer ? (RL * F⁻, RR * F⁺) : (Ef_tot[o], Eb_tot[o])
        T[o] = SMatrix{4,4,ComplexF64,16}(vcat(Ef, Eb))
    end
    return T
end


# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────


#TODO Is this truly adapted on sources confined to one layer ?
"""
    Modal_amp(stack, sources::Vector{<:SourceDef}, layer::Int, modes, Sall, Sint; tol=1E-8, layers=nothing)

Calculates the forward and backward field amplitude in each layer of a stack
of thin layers, for one or several sources confined to one layer. 
"""
function Modal_amp(stack::Vector{<:AbstractLayer}, sources::Vector{<:SourceDef}, layer::Int,
                   modes::Vector{<:Modes}, Sall::Vector{<:ScatteringM},
                   Sint::Vector{<:SMatrix{4,4}}; tol::Number = 1E-8,
                   layers::Union{Nothing,AbstractSet{Int}} = nothing)
    F⁺, F⁻, RL, _, RR_right, _, _, _, _, _ = _cavity_solve(stack, sources, layer, modes, Sall)
    Ef_tot, Eb_tot = _walk_stack(stack, layer, F⁺, F⁻, RL, RR_right, modes, Sall, Sint; tol=tol, layers=layers)
    return ModalAmp.(Ef_tot, Eb_tot)
end


"""
    Modal_amp_profile(stack, sources, layer, modes, Sall, Sint, zs; tol=1E-8) -> Vector{ModalAmp}

On-demand companion to `Modal_amp`: reconstructs the field at arbitrary
points within the layer of the stack containing a source.  Internal source only.

    Ef_eff(z) = Ef_j0 + Σ_{z_k < z} sif_k·exp(-i·Qf·z_k)
    Eb_eff(z) = Eb_d  + Σ_{z_k > z} sib_k·exp(+i·Qb·(d-z_k))
"""
function Modal_amp_profile(stack::Vector{<:AbstractLayer}, sources::Vector{<:SourceDef}, layer::Int,
                           modes::Vector{<:Modes}, Sall::Vector{<:ScatteringM},
                           Sint::Vector{<:SMatrix{4,4}}, zs::AbstractVector{<:Real}; tol::Number = 1E-8)
    N = length(stack)
    @assert layer < N "Modal_amp_profile does not support layer==N -- the semi-infinite substrate has no far edge to reference a within-layer position against"

    F⁺, F⁻, RL, RR, _, Qf, Qb, d, _, seeds = _cavity_solve(stack, sources, layer, modes, Sall)
    Ef_j0 = RL * F⁻
    Eb_d  = RR * F⁺

    result = Vector{ModalAmp{ComplexF64}}(undef, length(zs))
    for (i, z) in enumerate(zs)

        zf = Float64(z)
        @assert -sqrt(eps(Float64)) <= zf <= d + sqrt(eps(Float64)) "z=$zf out of bounds [0,$d] for layer $layer"
        Ef_eff = Ef_j0
        Eb_eff = Eb_d

        for (zpos, sif, sib) in seeds
            if zpos < zf
                Ef_eff = Ef_eff + sif .* _SProp(-zpos, Qf)
            end
            if zpos > zf
                Eb_eff = Eb_eff + sib .* _SProp(d-zpos, Qb)
            end
        end

        result[i] = ModalAmp(_flush_denormal(Ef_eff), _flush_denormal(Eb_eff))
    end

    return result
end


end
