module FieldSolver

export ModalAmp, plane_wave_amp
export _cavity_transfer_columns

import ..ModeSolver: AbstractLayer, Modes, Modes_Stack
import ..ScatteringMatrix: ScatteringM, _SProp, _Redheffer, S_all_ingredients, S_all_Matrix


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

# `_cond2x2` after scaling each column to unit norm. A transmission block through a
# layer whose two modes decay at very different rates (a hyperbolic slab: e^{-21}
# vs e^{-3.7} at 150 nm) has columns of very different size; that is a scaling,
# not a near-singularity, and the 2×2 solve handles it. Only an equilibrated
# condition number above 1/tol means the two modes are genuinely indistinct.
function _cond2x2_eq(A::SMatrix{2,2})
    n1 = sqrt(abs2(A[1,1]) + abs2(A[2,1])); n2 = sqrt(abs2(A[1,2]) + abs2(A[2,2]))
    (n1 > 0 && n2 > 0 && isfinite(n1) && isfinite(n2)) || return Inf
    return _cond2x2(A * Diagonal(SVector(1 / n1, 1 / n2)))
end


function _SProp(d::Real, Q::AbstractVector)
    φ1 = exp(im * Q[1] * d)
    φ2 = exp(im * Q[2] * d)
    Tc = promote_type(typeof(φ1), typeof(φ2))
    return SVector{2,Tc}(φ1, φ2)
end


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



"""
    _cavity_resummation(S₊, S₋, RL, RR, Qf, Qb, d, N, layer) -> (F⁺, F⁻)

The linear, z-independent part of the cavity solve: resums the multiple
reflections between the layer's two edges,
`F⁺ = (I - ΦF·RL·ΦB·RR)⁻¹ (S₊ + ΦF·RL·S₋)`, `F⁻ = S₋ + ΦB·RR·F⁺`.

Generic over the seed's shape: `S₊,S₋` may be the physical `SVector{2}` seeds
(one physical seed) or `SMatrix{2,n}` blocks of basis inputs (what
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
    _walk_stack(stack, layer, F⁺, F⁻, RL, RR_right, modes, Sall, Sint; tol, layers)
        -> (Ef_tot, Eb_tot)

The Redheffer walk from the source layer's cavity amplitudes `(F⁺,F⁻)` to
every other layer, plus the final re-referencing of `Eb` to each layer's own
z=d. Extracted verbatim from the former `Modal_amp` (2026-09-23). Linear in `(F⁺,F⁻)`
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

            if _cond2x2_eq(S11k) > 1 / tol
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

            if _cond2x2_eq(S22k) > 1 / tol
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
`[S₊; S₋]` to the modal amplitudes `[Ef; Eb]` of
each observation layer `o ∈ obs_layers`:

    [Ef_o; Eb_o] = T^o · [S₊; S₋]

obtained by running `_cavity_resummation` + `_walk_stack` on the 4 unit basis
inputs at once (exact, since both are linear in `(S₊,S₋)`). `S₊` is the forward
seed referenced at z=d of `layer`, `S₋` the backward seed at z=0: a source at
`zpos` with modal amplitudes `(sf, sb)` gives `S₊ = sf∘e^{iQf(d−zpos)}`,
`S₋ = sb∘e^{−iQb·zpos}` (`ClosedFormField._seed_phases`). Reference conventions
are `ModalAmp`'s: `Ef_o` at z=0 of layer `o`, `Eb_o` at z=d.

For `o == layer` (same layer) the columns are the REFLECTED (cavity) part only,
`[RL·F⁻; RR·F⁺]`; the direct term (`sf·e^{-iQf zpos}` below / `sb·e^{iQb(d-zpos)}`
above the source) is not part of this map and is added by the caller. For
`o ≠ layer` the columns are `_walk_stack`'s `Ef_tot[o]`/`Eb_tot[o]`.

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


"""
    plane_wave_amp(stack, modes, Sall, Sint, a; side=:top, obs_layers=1:N, include_incident=true, tol=1E-8)
        -> Dict{Int, ModalAmp{ComplexF64}}
    plane_wave_amp(stack, ω, kt, a; kwargs...)

Modal amplitudes in each layer `o ∈ obs_layers` for a single plane wave of in-plane
wavevector `kt` hitting the stack from one of its half-spaces.

`a` holds the incident wave's two modal amplitudes, referenced AT THE STACK FACE:
  - `side = :top`    -- forward modes of layer 1, at its z=d edge (the face with layer 2);
  - `side = :bottom` -- backward modes of layer N, at its z=0 edge (the face with layer N-1).
Referencing at the face keeps `a` independent of the nominal thickness given to the
half-space. `a = modes[1].Vf \\ [Ex, Ey, Hx, Hy]` converts a tangential field at the face.

This is the closed-form map with the edge seed `[S₊; S₋] = [a; 0]` (top) or `[0; a]`
(bottom): `[Ef_o; Eb_o] = T^o · [S₊; S₋]`, `T^o = _cavity_transfer_columns(...)`.

Output follows the `ModalAmp` convention in every layer: `Ef` at z=0, `Eb` at z=d. In the
incidence layer the incident wave is added (`include_incident=true`) after moving it to that
convention (`a·e^{-iQf·d₁}` top, `a·e^{+iQb·d_N}` bottom); with `include_incident=false`
only the scattered (reflected) wave is returned there. A plane wave fills the whole
incidence layer, so these amplitudes hold at every depth of it.
"""
function plane_wave_amp(stack::Vector{<:AbstractLayer}, modes::Vector{<:Modes},
                        Sall::Vector{<:ScatteringM}, Sint::Vector{<:SMatrix{4,4}},
                        a::AbstractVector; side::Symbol = :top, obs_layers = 1:length(stack),
                        include_incident::Bool = true, tol::Number = 1E-8)
    N = length(stack)
    length(a) == 2 || error("plane_wave_amp: `a` must hold the 2 incident modal amplitudes (got $(length(a)))")
    a2 = SVector{2,ComplexF64}(a[1], a[2])
    z2 = zero(a2)
    layer, u = side === :top    ? (1, vcat(a2, z2)) :
               side === :bottom ? (N, vcat(z2, a2)) :
               error("plane_wave_amp: side must be :top or :bottom (got :$side)")

    T = _cavity_transfer_columns(stack, layer, modes, Sall, Sint; obs_layers = obs_layers, tol = tol)

    out = Dict{Int,ModalAmp{ComplexF64}}()
    for (o, To) in T
        v = To * u
        Ef, Eb = v[SOneTo(2)], v[SVector(3, 4)]
        if o == layer && include_incident
            d = stack[layer].d
            side === :top ? (Ef = Ef + a2 .* _SProp(-d, modes[layer].Qf)) :
                            (Eb = Eb + a2 .* _SProp(d, modes[layer].Qb))
        end
        out[o] = ModalAmp(_flush_denormal(Ef), _flush_denormal(Eb))
    end
    return out
end

function plane_wave_amp(stack::Vector{<:AbstractLayer}, ω::Real, kt::AbstractVector, a::AbstractVector; kwargs...)
    modes = Modes_Stack(stack, ω, kt)
    Sint, Sprop = S_all_ingredients(stack, modes)
    Sall = S_all_Matrix(stack, modes, Sint, Sprop)
    return plane_wave_amp(stack, modes, Sall, Sint, a; kwargs...)
end

end
