module LayeredMaterials

export AbstractMaterial, IsotropicMaterial, UniaxialMaterial, BiaxialMaterial, GyrotropicMaterial
export AbstractLayer, FlatOpticalLayer, FlatLayer

using LinearAlgebra, StaticArrays


# ─────────────────────────────────────────────────────────────────────────────
#  Materials
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Layers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

# One allocation-free instance of vacuum μ, for FlatOpticalLayer
const _μ_Identity = IsotropicMaterial(one(ComplexF64))
_layer_μ(stack::FlatOpticalLayer) = _μ_Identity
_layer_μ(stack::AbstractLayer) = stack.μ


# Azimuthal symmetry of the stack (used with asymmetric sources: rotate modes instead of re-solving)
"""
    _is_azimuthally_symmetric_stack(stack::Vector{<:AbstractLayer}) -> Bool

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

end
