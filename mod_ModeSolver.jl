module ModeSolver

export R2s, Modes
export AbstractLayer, FlatOpticalLayer, FlatLayer
export AbstractMaterial, IsotropicMaterial, UniaxialMaterial, BiaxialMaterial, GyrotropicMaterial
export Modes_Stack

using LinearAlgebra, StaticArrays


# ─────────────────────────────────────────────────────────────────────────────
#  Constant and structures declaration
# ─────────────────────────────────────────────────────────────────────────────

const R2s = SMatrix{2,2,Int}(0, -1, 1, 0)

const R4s = SMatrix{4,4,Float64}(0,0,0,1, 0,0,-1,0, 0,-1,0,0, 1,0,0,0)


# Material related definition
abstract type AbstractMaterial end

Base.eltype(m::AbstractMaterial) = eltype(m.ϵ)
Base.getindex(m::AbstractMaterial, I...) = getindex(m.ϵ, I...)
Base.size(::AbstractMaterial) = (3, 3)
Base.size(::AbstractMaterial, d) = d in (1, 2) ? 3 : 1

# Isotropic
struct IsotropicMaterial{T<:Complex} <: AbstractMaterial
    ϵ::UniformScaling{T}
end

IsotropicMaterial(ϵ::Complex) = IsotropicMaterial(UniformScaling(ϵ))
IsotropicMaterial(ϵ::Real) = IsotropicMaterial(complex(ϵ))

# Uniaxial
struct UniaxialMaterial{T<:Complex} <: AbstractMaterial
    ϵ::Diagonal{T, SVector{3,T}}
end

function UniaxialMaterial(ϵt::Complex, ϵz::Complex)
    T = promote_type(typeof(ϵt), typeof(ϵz))
    return UniaxialMaterial(Diagonal(SVector{3,T}(ϵt, ϵt, ϵz)))
end

UniaxialMaterial(ϵt::Real, ϵz::Real) = UniaxialMaterial(complex(ϵt), complex(ϵz))

# Biaxial material
struct BiaxialMaterial{T<:Complex} <: AbstractMaterial
    ϵ::Diagonal{T, SVector{3,T}}
end

BiaxialMaterial(ϵ::SVector{3, <:Complex}) = BiaxialMaterial(Diagonal(ϵ))
BiaxialMaterial(ϵx::Complex, ϵy::Complex, ϵz::Complex) =
    BiaxialMaterial(SVector{3}(promote(ϵx, ϵy, ϵz)))
BiaxialMaterial(ϵx::Real, ϵy::Real, ϵz::Real) =
    BiaxialMaterial(complex(ϵx), complex(ϵy), complex(ϵz))

# Gyrotropic material
struct GyrotropicMaterial{T<:Complex} <: AbstractMaterial
    ϵ::SMatrix{3,3,T,9}
end

GyrotropicMaterial(ϵ::AbstractMatrix{<:Real})    = GyrotropicMaterial(complex.(float.(ϵ)))
GyrotropicMaterial(ϵ::AbstractMatrix{<:Complex}) = GyrotropicMaterial(SMatrix{3,3}(ϵ))


# Layer related definitions
abstract type AbstractLayer end

struct FlatOpticalLayer{Tϵ<:AbstractMaterial} <: AbstractLayer
    ϵ::Tϵ
    d::Float64
end

struct FlatLayer{Tϵ<:AbstractMaterial, Tμ<:AbstractMaterial} <: AbstractLayer
    ϵ::Tϵ
    μ::Tμ
    d::Float64
end

# Modes related definitions
const _Modes = Tuple{SMatrix{4,2,ComplexF64,8}, SMatrix{4,2,ComplexF64,8}, SVector{2,ComplexF64}}

struct Modes
    Vf::SMatrix{4,2,ComplexF64,8}
    Vb::SMatrix{4,2,ComplexF64,8}
    Wf::SMatrix{4,2,ComplexF64,8}
    Wb::SMatrix{4,2,ComplexF64,8}
    Qf::SVector{2,ComplexF64}
    Qb::SVector{2,ComplexF64}
end

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

# Wraps modes out of eigen into Static Arrays.
function Modes(Vf::AbstractMatrix, Vb::AbstractMatrix, Wf::AbstractMatrix, Wb::AbstractMatrix,
               Qf::AbstractVector, Qb::AbstractVector)
    Modes(SMatrix{4,2,ComplexF64}(Vf), SMatrix{4,2,ComplexF64}(Vb),
          SMatrix{4,2,ComplexF64}(Wf), SMatrix{4,2,ComplexF64}(Wb),
          SVector{2,ComplexF64}(Qf), SVector{2,ComplexF64}(Qb))
end


# One allocation-free instance of vacuum μ, for FlatOpticalLayer
const _μ_Identity = IsotropicMaterial(one(ComplexF64))
_layer_μ(stack::FlatOpticalLayer) = _μ_Identity
_layer_μ(stack::AbstractLayer) = stack.μ

# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    _berreman_core(ϵ, μ, ω, k)

Builds the 4x4 Berreman matrix from a layer's 3x3 ϵ/μ tensors.

"""
function _berreman_core(ϵ, μ, ω, k)

    T = promote_type(eltype(ϵ), eltype(μ), eltype(k), typeof(ω))

    ϵ11=T(ϵ[1,1]); ϵ12=T(ϵ[1,2]); ϵ21=T(ϵ[2,1]); ϵ22=T(ϵ[2,2])
    ϵ13=T(ϵ[1,3]); ϵ23=T(ϵ[2,3]); ϵ31=T(ϵ[3,1]); ϵ32=T(ϵ[3,2]); ϵ33=T(ϵ[3,3])

    μ11=T(μ[1,1]); μ12=T(μ[1,2]); μ21=T(μ[2,1]); μ22=T(μ[2,2])
    μ13=T(μ[1,3]); μ23=T(μ[2,3]); μ31=T(μ[3,1]); μ32=T(μ[3,2]); μ33=T(μ[3,3])

    k1=T(k[1]); k2=T(k[2])

    denomϵ = one(T) / ϵ33
    denomμ = one(T) / μ33
    iω = one(T) / ω

    ϵzt1 = ϵ31; ϵzt2 = ϵ32
    μzt1 = μ31; μzt2 = μ32

    Sϵ11 = ϵ11 - ϵ13 * denomϵ * ϵzt1
    Sϵ12 = ϵ12 - ϵ13 * denomϵ * ϵzt2
    Sϵ21 = ϵ21 - ϵ23 * denomϵ * ϵzt1
    Sϵ22 = ϵ22 - ϵ23 * denomϵ * ϵzt2

    Sμ11 = μ11 - μ13 * denomμ * μzt1
    Sμ12 = μ12 - μ13 * denomμ * μzt2
    Sμ21 = μ21 - μ23 * denomμ * μzt1
    Sμ22 = μ22 - μ23 * denomμ * μzt2

    ρϵ1 = denomϵ * ϵzt1; ρϵ2 = denomϵ * ϵzt2
    σϵ1 = ϵ13 * denomϵ;  σϵ2 = ϵ23 * denomϵ
    ρμ1 = denomμ * μzt1; ρμ2 = denomμ * μzt2
    σμ1 = μ13 * denomμ;  σμ2 = μ23 * denomμ

    t2_1 = -conj(k2); t2_2 = conj(k1)
    t3_1 = iω*k1; t3_2 = iω*k2

    Rσμ1 = σμ2; Rσμ2 = -σμ1
    Rσϵ1 = σϵ2; Rσϵ2 = -σϵ1

    B11_11 = Rσμ1 * t2_1 - k1 * ρϵ1
    B11_12 = Rσμ1 * t2_2 - k1 * ρϵ2
    B11_21 = Rσμ2 * t2_1 - k2 * ρϵ1
    B11_22 = Rσμ2 * t2_2 - k2 * ρϵ2

    t1Sμ11 = ω * Sμ21;  t1Sμ12 = ω * Sμ22 
    t1Sμ21 = -ω * Sμ11; t1Sμ22 = -ω * Sμ12
    B12_11 = t1Sμ11 - t3_1 * denomϵ * t2_1
    B12_12 = t1Sμ12 - t3_1 * denomϵ * t2_2
    B12_21 = t1Sμ21 - t3_2 * denomϵ * t2_1
    B12_22 = t1Sμ22 - t3_2 * denomϵ * t2_2

    t1Sϵ11 = ω * Sϵ21;  t1Sϵ12 = ω * Sϵ22
    t1Sϵ21 = -ω * Sϵ11; t1Sϵ22 = -ω * Sϵ12
    B21_11 = t3_1 * denomμ * t2_1 - t1Sϵ11
    B21_12 = t3_1 * denomμ * t2_2 - t1Sϵ12
    B21_21 = t3_2 * denomμ * t2_1 - t1Sϵ21
    B21_22 = t3_2 * denomμ * t2_2 - t1Sϵ22

    B22_11 = Rσϵ1 * t2_1 - k1 * ρμ1
    B22_12 = Rσϵ1 * t2_2 - k1 * ρμ2
    B22_21 = Rσϵ2 * t2_1 - k2 * ρμ1
    B22_22 = Rσϵ2 * t2_2 - k2 * ρμ2

    Bs = SMatrix{4,4,T}(
        B11_11, B11_21, B21_11, B21_21,
        B11_12, B11_22, B21_12, B21_22,
        B12_11, B12_21, B22_11, B22_21,
        B12_12, B12_22, B22_12, B22_22,
    )
    return Matrix(Bs)
end



"""
    _eigen_QV(layer, ω, kx, ky)

General dispatcher between closed-form Q,V calculation or mode solving through Berreman's matrix.

"""
_eigen_QV(layer::AbstractLayer, ω::Number, kx::Number, ky::Number) =
    _eigen_QV(layer.ϵ, _layer_μ(layer), ω, kx, ky)


function _eigen_QV(ϵ::AbstractMaterial, μ::AbstractMaterial, ω::Number, kx::Number, ky::Number)
    B = _berreman_core(ϵ, μ, ω, SVector{2}(kx, ky))
    Q, V = eigen(B)
    return SVector{4,ComplexF64}(Q), SMatrix{4,4,ComplexF64}(V)
end



# TODO Recheck and validate this properly, it works but still
"""
Closed-form (Q,V) for ϵ isotropic or uniaxial (c-axis along ẑ, i.e. the layer
normal -- the only geometry `UniaxialMaterial`'s own constructor can express)
with isotropic μ. Standard TE(ordinary)/TM(extraordinary) decomposition:
ŝ = ẑ×k̂_t (TE polarization direction, Ez≡0), p-direction in the (k̂_t,ẑ)
plane (TM, Hz≡0); `φ = atan(ky,kx)` (arbitrary at kρ=0, harmless -- the
construction stays exactly linearly independent there regardless of the
convention chosen, unlike a numerically degenerate `eigen()` call). The
prefactors `A = 1/(ωμ)`, `B = 1/(ωϵt)` and the exact sign pattern below were
fixed by directly verifying `Berreman_B*Ψ - Q*Ψ ≈ 0` (machine precision, not
merely `Identity_check`-level) against THIS file's own `_berreman_core`
across isotropic/uniaxial materials (including negative/hyperbolic ϵz),
on/off axis, and on/near/above each material's own light line -- see
`handoff_fixed_grid_green_apply.md`'s Module 4a discussion (2026-09-01) for
the validation. `kzo`/`kze` use `sqrt` of a `Complex` argument throughout
(never a bare real `sqrt`), so evanescent (imaginary kz), lossy, and
negative-ϵz branches are all handled by the same principal-branch formula --
correctness does not depend on which branch `sqrt` picks, since both signs of
each of `kzo`,`kze` are returned as separate eigenpairs and left to the
UNCHANGED forward/backward `Pz`/`imag(Q)` sort below to classify, exactly as
`eigen()`'s arbitrarily-ordered output already required.

Isotropic is the exact degenerate case `ϵt=ϵz`: `kzo≡kze`, TE/TM become the
same eigenvalue with two linearly independent eigenvectors (confirmed via
`Identity_check`-equivalent completeness, not just individually-valid
eigenvectors) -- correct, expected, and NOT specially cased.
"""
function _eigen_QV(ϵ::Union{IsotropicMaterial,UniaxialMaterial}, μ::IsotropicMaterial,
                    ω::Number, kx::Number, ky::Number)
    ϵt = ComplexF64(ϵ[1,1]); ϵz = ComplexF64(ϵ[3,3]); μr = ComplexF64(μ[1,1])
    ωc = ComplexF64(ω)
    kρ = hypot(Float64(kx), Float64(ky))
    φ  = kρ == 0 ? 0.0 : atan(Float64(ky), Float64(kx))
    cφ, sφ = cos(φ), sin(φ)

    kzo = sqrt(ωc^2 * ϵt * μr - kρ^2)
    kze = sqrt(ϵt * (ωc^2 * μr - kρ^2 / ϵz))

    A = 1 / (ωc * μr)
    Bn = 1 / (ωc * ϵt)

    ΨTEf = SVector{4,ComplexF64}(-sφ, cφ, -A * kzo * cφ, -A * kzo * sφ)
    ΨTEb = SVector{4,ComplexF64}(-sφ, cφ,  A * kzo * cφ,  A * kzo * sφ)
    ΨTMf = SVector{4,ComplexF64}(Bn * kze * cφ,  Bn * kze * sφ, -sφ, cφ)
    ΨTMb = SVector{4,ComplexF64}(-Bn * kze * cφ, -Bn * kze * sφ, -sφ, cφ)

    Qs = SVector{4,ComplexF64}(kzo, -kzo, kze, -kze)
    Vs = hcat(ΨTEf, ΨTEb, ΨTMf, ΨTMb)
    return Qs, Vs
end


"""
    _modes_layer(layer, ω, kx, ky; tol=sqrt(eps(Float64)))

Shared core for EVERY material type: obtain `(Q,V)` from `_eigen_QV`, 
then sort into forward/backward branches by power flow / decay direction. 
"""
function _modes_layer(layer::AbstractLayer, ω::Number, kx::Number, ky::Number;
                       tol::Real=sqrt(eps(Float64)))

    Qs, Vs = _eigen_QV(layer, ω, kx, ky)
    Ws = R4s * Vs

    Pz = SVector{4,Float64}(ntuple(j -> 0.25*real(dot(Vs[:,j], Ws[:,j])), 4))

    Vf = MMatrix{4,2,ComplexF64}(ntuple(_->ComplexF64(0),8))
    Vb = MMatrix{4,2,ComplexF64}(ntuple(_->ComplexF64(0),8))
    Wf = MMatrix{4,2,ComplexF64}(ntuple(_->ComplexF64(0),8))
    Wb = MMatrix{4,2,ComplexF64}(ntuple(_->ComplexF64(0),8))
    Qf = MVector{2,ComplexF64}(ComplexF64(0),ComplexF64(0))
    Qb = MVector{2,ComplexF64}(ComplexF64(0),ComplexF64(0))
    deg_idx = MVector{4,Int}(0,0,0,0)
    ndeg = 0
    idf = 1
    idb = 1

    # Sorts according to sign of power if propagating mode
    for j in 1:4
        if abs(Pz[j]) > tol
            if Pz[j] > 0
                Vf[:,idf]=Vs[:,j]; Wf[:,idf]=Ws[:,j]; Qf[idf]=Qs[j]; idf += 1
            else
                Vb[:,idb]=Vs[:,j]; Wb[:,idb]=Ws[:,j]; Qb[idb]=Qs[j]; idb += 1
            end
        elseif abs(imag(Qs[j])) > tol
            # Sorts according to direction of decay
            if imag(Qs[j]) > tol
                Vf[:,idf]=Vs[:,j]; Wf[:,idf]=Ws[:,j]; Qf[idf]=Qs[j]; idf += 1
            elseif imag(Qs[j]) < -(tol)
                Vb[:,idb]=Vs[:,j]; Wb[:,idb]=Ws[:,j]; Qb[idb]=Qs[j]; idb += 1
            end
        else
            # Stores for later assignment
            ndeg += 1
            deg_idx[ndeg] = j
        end
    end

    # Assign degenerate not power carrying modes to fill remaining slots
    for i in 1:ndeg
        j = deg_idx[i]
        if idf ≤ 2
            Vf[:,idf]=Vs[:,j]; Wf[:,idf]=Ws[:,j]; Qf[idf]=Qs[j]; idf += 1
        elseif idb ≤ 2
            Vb[:,idb]=Vs[:,j]; Wb[:,idb]=Ws[:,j]; Qb[idb]=Qs[j]; idb += 1
        end
    end

    return (SMatrix(Vf), SMatrix(Wf), SVector(Qf)), (SMatrix(Vb), SMatrix(Wb), SVector(Qb))
end


"""
    _match_modes(modes_A, modes_B; eigpenalty=0.0)

"""
function _match_modes(modes_A::_Modes, modes_B::_Modes; eigpenalty::Real = 0.0)
    _, Wa, Qa = modes_A
    Vb, Wb, Qb = modes_B

    C = Wa' * Vb
    score_id   = abs2(C[1,1]) + abs2(C[2,2])
    score_swap = abs2(C[1,2]) + abs2(C[2,1])
    if eigpenalty != 0.0
        score_id   -= eigpenalty*(abs2(Qa[1]-Qb[1]) + abs2(Qa[2]-Qb[2]))
        score_swap -= eigpenalty*(abs2(Qa[1]-Qb[2]) + abs2(Qa[2]-Qb[1]))
    end
    if score_swap > score_id
        return hcat(Vb[:,2], Vb[:,1]), hcat(Wb[:,2], Wb[:,1]), SVector{2,ComplexF64}(Qb[2], Qb[1])
    else
        return Vb, Wb, Qb
    end
end


"""
    _modes_stack_core(stack, ω, kx, ky)

Shared per-stack mode solver. Calls _modes_layer(layer, ω, kx, ky) per layer, 
then _match_modes aligns each subsequent layer's to layer 1's ordering.
"""
function _modes_stack_core(stack::Vector{<:AbstractLayer}, ω::Number, kx::Number, ky::Number)
    N = length(stack)

    # Keyed by object identity (objectid), not material value: layers that share
    # the same ϵ/μ INSTANCE hit the cache at the cost of two cheap pointer hashes.
    raw_cache = Dict{Tuple{UInt,UInt}, Tuple{_Modes, _Modes}}()
    solved(layer) = get!(raw_cache, (objectid(layer.ϵ), objectid(_layer_μ(layer)))) do
        _modes_layer(layer, ω, kx, ky)
    end

    M0f, M0b = solved(stack[1])
    V0f, W0f, Q0f = M0f
    V0b, W0b, Q0b = M0b

    smat = Vector{Modes}(undef, N)

    smat[1] = Modes(V0f, V0b, W0f, W0b, Q0f, Q0b)

    for j in 2:N
        Mjf, Mjb = solved(stack[j])

        Vjf, Wjf, Qjf = _match_modes(M0f, Mjf)
        Vjb, Wjb, Qjb = _match_modes(M0b, Mjb)

        smat[j] = Modes(Vjf, Vjb, Wjf, Wjb, Qjf, Qjb)

        V0f = Vjf; V0b = Vjb
        W0f = Wjf; W0b = Wjb
        Q0f = Qjf; Q0b = Qjb
        M0f = (V0f, W0f, Q0f)
        M0b = (V0b, W0b, Q0b)
    end

    return smat
end


"""
    Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, k∥)

Returns the ordered and aligned modes and k⟂ in each layer of a stack. 

# Arguments
- `k∥`: can be called as a single kₓ (::Number) value, as a vector [kx; ky] (::AbstractVector) 
  or as two arguments kρ and ϕ (::Number, ::Number).
"""
Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, kx::Number) =
    _modes_stack_core(stack, ω, kx, zero(kx))

function Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, k::AbstractVector)
    length(k) == 2 || throw(ArgumentError("k must be a 2-element vector [kx; ky]"))
    return _modes_stack_core(stack, ω, k[1], k[2])
end

Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, kρ::Number, ϕ::Number) =
    _modes_stack_core(stack, ω, kρ*cos(ϕ), kρ*sin(ϕ))


"""
Stronger than a `ky -> -ky` mirror: true iff every layer's `(ϵ,μ)` pair
dispatches to `_eigen_QV`'s CLOSED-FORM isotropic/uniaxial branch (isotropic
or uniaxial `ϵ`, c-axis along ẑ, AND isotropic `μ` -- exactly the
`_eigen_QV(ϵ::Union{IsotropicMaterial,UniaxialMaterial}, μ::IsotropicMaterial,
...)` method above, never the general Berreman-`eigen()` fallback). For such
a stack the WHOLE Berreman problem is invariant under an arbitrary in-plane
rotation about ẑ -- so a single stack solve at φ=0 suffices for every φ on a
2D `(kρ,φ)` grid: see `_rotate_modes`'s own docstring for the transform and
its empirical verification.
"""
function _is_azimuthally_symmetric_stack(stack::Vector{<:AbstractLayer})
    for lay in stack
        lay.ϵ isa Union{IsotropicMaterial,UniaxialMaterial} || return false
        μ = hasproperty(lay, :μ) ? lay.μ : nothing
        (μ === nothing || μ isa IsotropicMaterial) || return false
    end
    return true
end

"""
The exact transformation of a layer's Berreman eigenmodes under the in-plane
rotation `(kx,ky) = (kρcos(φ),kρsin(φ))` about ẑ, valid whenever
`_is_azimuthally_symmetric_stack(stack)` holds: `Vf/Vb/Wf/Wb -> Rot4(φ)*(...)`,
`Qf/Qb` unchanged, with `Rot4(φ) = blockdiag(R(φ),R(φ))`, `R(φ)` the ordinary
2x2 rotation -- the SAME `R(φ)` applied to the `(Ex,Ey)`-like row pair (1,2)
and, independently, to the `(Hx,Hy)`-like row pair (3,4). Read directly off
`_eigen_QV`'s own closed-form `(ΨTEf,ΨTEb,ΨTMf,ΨTMb)` formulas (each is
exactly `Rot4(φ)` applied to its own φ=0 value) and `Qs=(kzo,-kzo,kze,-kze)`
depending on kρ alone -- so `_modes_layer`'s forward/backward sort (driven by
`Pz=real(Vs'Ws)/4` and `imag(Qs)`, both invariant since `Rot4(φ)` is real
orthogonal, `Rot4(φ)'*Rot4(φ)=I`) picks the SAME columns at every φ, and
`_modes_stack_core`'s `_match_modes` overlap `Wa'*Vb` is likewise invariant
(`(Rot4 Wa)'*(Rot4 Vb) = Wa'*Vb`) -- so the WHOLE aligned `Vector{Modes}`
(every layer, already continuity-matched) transforms by one common `Rot4(φ)`
("uniform transform, matching survives").

Downstream: `S_all_ingredients`/`S_all_Matrix` applied to a uniformly-
`Rot4(φ)`-rotated mode set reproduce their true-φ values EXACTLY (same
bilinear-form cancellation) -- `Sint`/`Sall`
built at φ=0 are reused unchanged for every φ, not just mirror-paired ones.
`Source_projection_Berreman_xyz`, fed the ACTUAL `kt=(kρcosφ,kρsinφ)` (not a
rotated one) together with the rotated modes, reproduces the true per-φ
projection exactly too -- it is cheap enough (one 4x4 solve per layer, no
stack-wide Redheffer scan) that it is simply recomputed at every φ rather
than given its own mirror-style shortcut.

Verified empirically (not merely derived): isotropic, uniaxial, and a
5-layer high-contrast/lossy isotropic stack, 4 kρ x 7 φ spanning all four
quadrants -- max relative disagreement between a direct φ-solve and this
Rot4(φ)-transformed φ=0 solve was ~1e-15-1e-16 (floating-point noise) for
modes, `Sint`, `Sall`, and the source projection alike
(`verify_mode_rotation.jl`, 2026-09-21) -- the same "confirmed to machine
zero" standard the mirror trick itself was held to.
"""
function _Rot4(φ::Float64)
    c, s = cos(φ), sin(φ)
    return SMatrix{4,4,ComplexF64}(c, s, 0, 0,
                                    -s, c, 0, 0,
                                    0, 0, c, s,
                                    0, 0, -s, c)
end

_rotate_modes(modes::Vector{<:Modes}, φ::Float64) = begin
    R = _Rot4(φ)
    [Modes(R * m.Vf, R * m.Vb, R * m.Wf, R * m.Wb, m.Qf, m.Qb) for m in modes]
end


end
