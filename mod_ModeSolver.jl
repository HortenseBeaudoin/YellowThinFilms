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


"""
    IsotropicMaterial(ϵ)

Permittivity tensor ϵI of an isotropic material. ϵ may be real or complex;
real values are promoted to complex.

# Fields
- `ϵ::UniformScaling{T}`: 3x3 permittivity tensor. 
"""
struct IsotropicMaterial{T<:Complex} <: AbstractMaterial
    ϵ::UniformScaling{T}
end

IsotropicMaterial(ϵ::Complex) = IsotropicMaterial(UniformScaling(ϵ))
IsotropicMaterial(ϵ::Real) = IsotropicMaterial(complex(ϵ))

"""
    UniaxialMaterial(ϵt, ϵz)

Permittivity tensor of a uniaxial material, with principal axis along z. 
ϵt and ϵz may be real or complex; real values are promoted to complex.

# Fields
- `ϵ::Diagonal{T, SVector{3,T}}`: 3x3 permittivity tensor, diag(ϵt, ϵt, ϵz). 
"""
struct UniaxialMaterial{T<:Complex} <: AbstractMaterial
    ϵ::Diagonal{T, SVector{3,T}}
end

function UniaxialMaterial(ϵt::Complex, ϵz::Complex)
    T = promote_type(typeof(ϵt), typeof(ϵz))
    return UniaxialMaterial(Diagonal(SVector{3,T}(ϵt, ϵt, ϵz)))
end

UniaxialMaterial(ϵt::Real, ϵz::Real) = UniaxialMaterial(complex(ϵt), complex(ϵz))

"""
    BiaxialMaterial(ϵ::SVector{3,<:Complex})
    BiaxialMaterial(ϵx, ϵy, ϵz)

Permittivity tensor of a biaxial material, with principal axis along z. 
Can be called with real or complex values; real values are promoted to complex.

# Fields
- `ϵ::Diagonal{T, SVector{3,T}}`: 3x3 permittivity tensor, diag(ϵx, ϵy, ϵz). 
"""
struct BiaxialMaterial{T<:Complex} <: AbstractMaterial
    ϵ::Diagonal{T, SVector{3,T}}
end

BiaxialMaterial(ϵ::SVector{3, <:Complex}) = BiaxialMaterial(Diagonal(ϵ))
BiaxialMaterial(ϵx::Complex, ϵy::Complex, ϵz::Complex) =
    BiaxialMaterial(SVector{3}(promote(ϵx, ϵy, ϵz)))
BiaxialMaterial(ϵx::Real, ϵy::Real, ϵz::Real) =
    BiaxialMaterial(complex(ϵx), complex(ϵy), complex(ϵz))

"""
    GyrotropicMaterial(ϵ::AbstractMatrix)

Permittivity tensor of a gyrotropic material, with principal axis along z. 
ϵxy = -ϵyx =ig with the exp(-iωt) convention.
Can be called with real or complex values; real values are promoted to complex.

# Fields
- `ϵ::SMatrix{3, 3, T, 9}`: 3x3 permittivity tensor. 
"""
struct GyrotropicMaterial{T<:Complex} <: AbstractMaterial
    ϵ::SMatrix{3,3,T,9}
end

GyrotropicMaterial(ϵ::AbstractMatrix{<:Real})    = GyrotropicMaterial(complex.(float.(ϵ)))
GyrotropicMaterial(ϵ::AbstractMatrix{<:Complex}) = GyrotropicMaterial(SMatrix{3,3}(ϵ))


# Layer related definitions
abstract type AbstractLayer end

"""
    FlatOpticalLayer(ϵ, d)

Optical layer for which μ = I. 

# Fields
- `ϵ::Tϵ`: Permittivity tensor of the material, with type depending on the material's class. 
- `d::Float64`: Thickness of the layer, in λ₀. 
"""
struct FlatOpticalLayer{Tϵ<:AbstractMaterial} <: AbstractLayer
    ϵ::Tϵ
    d::Float64
end

"""
    FlatLayer(ϵ, d)

Optical layer for which μ ≠ I. 

# Fields
- `ϵ::Tϵ`: Permittivity tensor of the material, with type depending on the material's class. 
- `μ::Tμ`: Permeability tensor of the material. 
- `d::Float64`: Thickness of the layer, in λ₀. 
"""
struct FlatLayer{Tϵ<:AbstractMaterial, Tμ<:AbstractMaterial} <: AbstractLayer
    ϵ::Tϵ
    μ::Tμ
    d::Float64
end


# Modes related definitions
const _Modes = Tuple{SMatrix{4,2,ComplexF64,8}, SMatrix{4,2,ComplexF64,8}, SVector{2,ComplexF64}}

"""
    Modes(Vf, Vb, Wf, Wb, Qf, Qb)

Modes in a layer, described by their eigenvalues and eigenvectors. 
There is always 4 modes propagating in multilayers, so elements are static arrays. 

# Fields
- `Vf::SMatrix{4, 2, ComplexF64, 8}`: Forward-going eigenvectors
- `Vb::SMatrix{4, 2, ComplexF64, 8}`: Backward-going eigenvectors
- `Wf::SMatrix{4, 2, ComplexF64, 8}`: Left-eigenvector / power in z coupling, Wf = R4s * Vf
- `Wb::SMatrix{4, 2, ComplexF64, 8}`: Left-eigenvector / power in z coupling, Wb = R4s * Vb
- `Qf::SVector{2, ComplexF64}`: Eigenvalues, k⟂/kz or forward-going modes
- `Qb::SVector{2, ComplexF64}`: Eigenvalues, k⟂/kz or backward-going modes
"""
struct Modes
    Vf::SMatrix{4, 2, ComplexF64, 8}
    Vb::SMatrix{4, 2, ComplexF64, 8}
    Wf::SMatrix{4, 2, ComplexF64, 8}
    Wb::SMatrix{4, 2, ComplexF64, 8}
    Qf::SVector{2, ComplexF64}
    Qb::SVector{2, ComplexF64}
end

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    Modes(Vf, Vb, Wf, Wb, Qf, Qb)

Wraps modes out of eigen into Static Arrays.
"""
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


# The three following helper functions exist for azimuthally symmetric stack, with asymmetric sources
"""
    _is_azimuthally_symmetric_stack(stack::Vector{<:AbstractLayer})

Check the azimuthal symmetry of the whole stack, not just one layer. 
All layers have to be isotropic of uniaxial, axis has to be aligned, μ has to be diagonal.
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
    _Rot4(φ)

Standard rotation matrix for angle φ. 
"""
function _Rot4(φ::Float64)
    c, s = cos(φ), sin(φ)
    return SMatrix{4,4,ComplexF64}(c, s, 0, 0,
                                    -s, c, 0, 0,
                                    0, 0, c, s,
                                    0, 0, -s, c)
end

"""
    _rotate_modes(modes, φ)

Returns rotated modes and associated unchanged eigenvalues Qf, Qb.
Valid when _is_azimuthally_symmetric_stack(stack) holds
"""
_rotate_modes(modes::Vector{<:Modes}, φ::Float64) = begin
    R = _Rot4(φ)
    [Modes(R * m.Vf, R * m.Vb, R * m.Wf, R * m.Wb, m.Qf, m.Qb) for m in modes]
end

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
    t3_1 = iω * k1; t3_2 = iω * k2

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
    _eigen_QV(ϵ::AbstractMaterial, μ::AbstractMaterial, ω::Number, kx::Number, ky::Number)
    _eigen_QV(ϵ::Union{IsotropicMaterial,UniaxialMaterial}, μ::IsotropicMaterial,
                ω::Number, kx::Number, ky::Number)

Returns the eigenvalues and eigenvectors of a layer for a given kₜ. 
Dispatches between closed-form QV calculation or eigen solve of Berreman's matrix based on material.
"""
_eigen_QV(layer::AbstractLayer, ω::Number, kx::Number, ky::Number) =
    _eigen_QV(layer.ϵ, _layer_μ(layer), ω, kx, ky)

    
function _eigen_QV(ϵ::AbstractMaterial, μ::AbstractMaterial, ω::Number, kx::Number, ky::Number)
    B = _berreman_core(ϵ, μ, ω, SVector{2}(kx, ky))
    Q, V = eigen(B)
    return SVector{4,ComplexF64}(Q), SMatrix{4,4,ComplexF64}(V)
end

# TODO : Review the equations in here, changed Bn for C, make sure it didn't break everything 
function _eigen_QV(ϵ::Union{IsotropicMaterial,UniaxialMaterial}, μ::IsotropicMaterial,
                    ω::Number, kx::Number, ky::Number)
    ϵt = ComplexF64(ϵ[1,1]); ϵz = ComplexF64(ϵ[3,3]); μr = ComplexF64(μ[1,1])
    ωc = ComplexF64(ω)
    kρ = hypot(Float64(kx), Float64(ky))
    φ = kρ == 0 ? 0.0 : atan(Float64(ky), Float64(kx))
    cφ, sφ = cos(φ), sin(φ)

    kzo = sqrt(ωc^2 * ϵt * μr - kρ^2)
    kze = sqrt(ϵt * (ωc^2 * μr - kρ^2 / ϵz))

    A = 1 / (ωc * μr)
    C = 1 / (ωc * ϵt)

    ΨTEf = SVector{4,ComplexF64}(-sφ, cφ, -A * kzo * cφ, -A * kzo * sφ)
    ΨTEb = SVector{4,ComplexF64}(-sφ, cφ,  A * kzo * cφ,  A * kzo * sφ)
    ΨTMf = SVector{4,ComplexF64}(C * kze * cφ,  C * kze * sφ, -sφ, cφ)
    ΨTMb = SVector{4,ComplexF64}(-C * kze * cφ, -C * kze * sφ, -sφ, cφ)

    Qs = SVector{4,ComplexF64}(kzo, -kzo, kze, -kze)
    Vs = hcat(ΨTEf, ΨTEb, ΨTMf, ΨTMb)
    return Qs, Vs
end

"""
    _modes_layer(layer, ω, kx, ky; tol=sqrt(eps(Float64)))

Obtain (Q, V), the eigenmodes and eigenvectors of a layer from _eigen_QV,
then sort then into forward/backward based on power in z and/or decay direction. 
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
    
Matches modes between two layers by looking at the maximum overlap between the eigenvectors, Wa'*Vb.
"""
function _match_modes(modes_A::_Modes, modes_B::_Modes; eigpenalty::Real = 0.0)
    _, Wa, Qa = modes_A
    Vb, Wb, Qb = modes_B

    C = Wa' * Vb
    score_id = abs2(C[1,1]) + abs2(C[2,2])
    score_swap = abs2(C[1,2]) + abs2(C[2,1])
    if eigpenalty != 0.0
        score_id -= eigpenalty*(abs2(Qa[1]-Qb[1]) + abs2(Qa[2]-Qb[2]))
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

Computes modes in each layer of the stack, 
then _match_modes aligns them based on layer 1's ordering.
"""
function _modes_stack_core(stack::Vector{<:AbstractLayer}, ω::Number, kx::Number, ky::Number)
    N = length(stack)

    # Make sure that mode solving is run once per material, not per layer.
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
    Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, kx::Number)
    Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, k::AbstractVector)
    Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, kρ::Number, ϕ::Number)

Returns the ordered and aligned modes and k⟂ in each layer of a stack. 

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, with N the number of layers of the stack + 2 (half-plane and/or substrate).
  Each entry contains a layer's description. 
- `ω`: frequency. 
- `k∥`: transverse, conserved, wavevector. Can be called as a single kₓ (::Number) value, as a vector [kx; ky] (::AbstractVector) 
  or as two arguments kρ and ϕ (::Number, ::Number).  

# Returns
- `Vector{Modes}(undef, N)`: of length N containing Modes, the struct containing eigenmodes and eigenvectors.
"""
Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, kx::Number) =
    _modes_stack_core(stack, ω, kx, zero(kx))

function Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, k::AbstractVector)
    length(k) == 2 || throw(ArgumentError("k must be a 2-element vector [kx; ky]"))
    return _modes_stack_core(stack, ω, k[1], k[2])
end

Modes_Stack(stack::Vector{<:AbstractLayer}, ω::Number, kρ::Number, ϕ::Number) =
    _modes_stack_core(stack, ω, kρ*cos(ϕ), kρ*sin(ϕ))

end
