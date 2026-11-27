module FieldSolver
# TODO Rename StackResponses, without the mod_
export ModalAmp, plane_wave_amp
export _cavity_transfer_columns

import ..Materials: AbstractLayer
import ..ModeSolver: Modes, Modes_Stack
import ..ScatteringMatrix: ScatteringM, _SProp, _Redheffer, S_all_ingredients, S_all_Matrix


using LinearAlgebra
using StaticArrays


# ─────────────────────────────────────────────────────────────────────────────
#  Constant and structures declaration
# ─────────────────────────────────────────────────────────────────────────────

"""
    ModalAmp(Ef, Eb)

Modal amplitudes resulting from the illumination of a single plane-wave,
after propagation in the stack using S-matrices. 

# Fields
- `Ef::SVector{2,T}`: forward-going modes amplitudes, referenced to the left, z=0.
- `Eb::SVector{2,T}`: backward-going modes amplitudes, referenced to the right, z=d.
"""
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

"""
    _cond2x2(A) -> Float64

Returns the conditionning of a 2x2 block `A`, after scaling each column to unit norm.
Scaling makes the safegard accurate even when the two modes are order of magnitudes apart.
"""
function _cond2x2(A::SMatrix{2,2})
    a1 = A[:, 1]; a2 = A[:, 2]
    n1 = norm(a1); n2 = norm(a2)

    (n1 > 0 && n2 > 0 && isfinite(n1) && isfinite(n2)) || return Inf
    c = abs(dot(a1, a2)) / (n1 * n2)          # |cos θ| between the columns
    c >= 1 && return Inf
    return sqrt((1 + c) / (1 - c))
end

"""
    _SProp(d, Q) -> SVector{2,ComplexF64}

Returns the 2x2 propagation matrix of a single direction, forward or backward.  
"""
function _SProp(d::Real, Q::AbstractVector)
    φ1 = exp(im * Q[1] * d)
    φ2 = exp(im * Q[2] * d)
    Tc = promote_type(typeof(φ1), typeof(φ2))
    return SVector{2,Tc}(φ1, φ2)
end

# Unit basis inputs for `_cavity_transfer_columns`: column j of [S₊;S₋] is e_j.
const _E₊ = SMatrix{2,4,ComplexF64}(1,0, 0,1, 0,0, 0,0)
const _E₋ = SMatrix{2,4,ComplexF64}(0,0, 0,0, 1,0, 0,1)

# ─────────────────────────────────────────────────────────────────────────────
#  Cavity solve functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    _R_cavity(stack, layer, modes, Sall, Sint) -> (RL, RR, RR_right, Qf, Qb, d)

Reflection matrices within a given layer, as seen from its own true edges,
z=0 and z=d; used in cavity solve.
Reflection matrix at right interface `RR` also returned; used in walk of 
the stack. 
"""
function _R_cavity(stack::Vector{<:AbstractLayer}, layer::Int,
                           modes::Vector{<:Modes}, Sall::Vector{<:ScatteringM},
                           Sint::Vector{<:SMatrix{4,4}})
    N = length(stack)
    d = stack[layer].d
    Qf, Qb = modes[layer].Qf, modes[layer].Qb

    RL = SMatrix{2,2,ComplexF64}(Sall[layer].SLeft[SOneTo(2), SVector(3,4)])

    if layer == N
        return RL, zero(SMatrix{2,2,ComplexF64}), nothing, Qf, Qb, d
    end

    RR_right = _Redheffer(Sint[layer], Sall[layer+1].SRight)
    RR = SMatrix{2,2,ComplexF64}(RR_right[SVector(3,4), SOneTo(2)])

    return RL, RR, RR_right, Qf, Qb, d
end

"""
    _cavity_resummation(S₊, S₋, RL, RR, Qf, Qb, d, N, layer) -> (F⁺, F⁻)

The linear, z-independent part of the cavity solve. All values are referenced
to the true layer's interface, or brought back to them. Solves;
    F⁺ = S₊ + ΦF·RL·(S₋ + ΦB·RR·F⁺)
    F⁻ = S₋ + ΦB·RR·F⁺

Returns (F⁺, F⁻), referenced at right and left interface respectively. 
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

# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    _walk_stack(stack, layer, F⁺, F⁻, RL, RR_right, modes, Sall, Sint; tol, layers) -> (Ef_tot, Eb_tot)

Walk through the stack to compute modal amplitudes in each layer. Comes after the 
cavity solve, works from its outputs. cond2x2 avoid inversion of near-singular 
transimission blocks, and flush turns numerically negligible amplitudes into 0s. 

Returns (Ef, Eb) referenced at the left (z=0) and right (z=d) interface respectively. 
"""
function _walk_stack(stack::Vector{<:AbstractLayer}, layer::Int, F⁺::StaticArray, F⁻::StaticArray,
                     RL::SMatrix{2,2}, RR_right, modes::Vector{<:Modes},
                     Sall::Vector{<:ScatteringM}, Sint::Vector{<:SMatrix{4,4}};
                     tol::Number = 1E-8, layers::Union{Nothing,AbstractSet{Int}} = nothing)
    N = length(stack)
    Z = zero(F⁺)

    Ef_tot = fill(Z, N)
    Eb_tot = fill(Z, N)
    Ef_left = RL * F⁻   # field at z=0

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

        a₊ = Z   # nothing incident from -infinity
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

"""
    _cavity_transfer_columns(stack, layer, modes, Sall, Sint; obs_layers, tol=1E-8) -> Dict{Int,SMatrix{4,4,ComplexF64,16}}
        
The z-independent linear map `T^o` from the source layer's (`layer`) edge seeds
`[S₊; S₋]` to the modal amplitudes `[Ef; Eb]`, for each observation 
layer `o ∈ obs_layers`. Related by:

    [Ef_o; Eb_o] = T^o · [S₊; S₋]

Obtained by running `_cavity_resummation` + `_walk_stack` on the 4 unit basis
inputs at once. `S₊` is referenced at z=d,, `S₋` at z=0. Seeds come from
`ClosedFormField._seed_phases`. For `o == layer` (same layer) the columns 
are the reflected (cavity) part only. For `o ≠ layer` the columns are 
`_walk_stack`'s `Ef_tot[o]`/`Eb_tot[o]`.

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, with N the number of layers of the stack + 2 (half-plane and/or substrate).
  Each entry contains a layer's description. 
- `layer::Int`: source layer id.
- `modes::Vector{<:Modes}`: vector of length N carrying the modes info in struct Modes.
- `Sall::Vector{<:ScatteringM}`: vector of length N carrying the composed, cumulative S-matrices of the stack.
- `Sint::Vector{<:SMatrix{4,4}}`: vector of length N carrying the single interface and propagation 
  S-matrices of the stack.
- `obs_layers`: observation layers targeted for precomputation. Defaults to all layers. 

# Returns
- `Dict{Int, SMatrix{4,4,ComplexF64,16}}`: keys are the observation layers id, and each id contains a 
  4x4 matrix linking source to answer in the stack. 
"""
function _cavity_transfer_columns(stack::Vector{<:AbstractLayer}, layer::Int,
                                  modes::Vector{<:Modes}, Sall::Vector{<:ScatteringM},
                                  Sint::Vector{<:SMatrix{4,4}};
                                  obs_layers = 1:length(stack), tol::Number = 1E-8)
    N = length(stack)
    @assert 1 <= layer <= N "layer must be a valid stack index"
    RL, RR, RR_right, Qf, Qb, d = _R_cavity(stack, layer, modes, Sall, Sint)

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

"""
    plane_wave_amp(stack, modes, Sall, Sint, a; side=:top, obs_layers=1:N, include_incident=true, tol=1E-8) -> Dict{Int,ModalAmp{ComplexF64}}
    plane_wave_amp(stack, ω, kt, a; kwargs...)

Helper for a single plane-wane illumination. Computes the modal amplitudes in each
`o ∈ obs_layers`. `a` holds the incident wave's two modal amplitudes, referenced at one
of the two edges of the stack:
  - `side = :top`, forward modes at z=d edge;
  - `side = :bottom`, backward modes at z=0 edge.
Closed-form map with the seed `[S₊; S₋] = [a; 0]` or `[0; a]`. The incident wave in the incidence 
layer is added by default. 

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, with N the number of layers of the stack + 2 (half-plane and/or substrate).
  Each entry contains a layer's description. 
- `modes::Vector{<:Modes}`: vector of length N carrying the modes info in struct Modes.
- `Sall::Vector{<:ScatteringM}`: vector of length N carrying the composed, cumulative S-matrices of the stack.
- `Sint::Vector{<:SMatrix{4,4}}`: vector of length N carrying the single interface and propagation 
  S-matrices of the stack.
- `a::AbstractVector`: incident wave two modal amplitudes. 
- `side::Symbol`: `top` or `bottom`, defaults to the first. 
- `obs_layers`: observation layers targeted for precomputation. Defaults to all layers. 

# Returns
- `Dict{Int, ModalAmp{ComplexF64}}`: keys are the observation layers id, and each id contains the 
  modal amplitude Ef and Eb.
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
