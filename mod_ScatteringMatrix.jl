module ScatteringMatrix

export ScatteringM
export S_all_Matrix
export S_all_ingredients

import ..Materials: AbstractLayer
import ..ModeSolver: Modes
using LinearAlgebra
using StaticArrays


# ─────────────────────────────────────────────────────────────────────────────
#  Constant and structures declaration
# ─────────────────────────────────────────────────────────────────────────────

const SI = SMatrix{4,4,ComplexF64}(I)

"""
    ScatteringM(SLeft, SRight)

Left- and right-looking cumulated scattering matrix. 

# Fields
- `SLeft::SMatrix{4, 4, T, 16}`: Left-looking scattering matrix.
- `SRight::SMatrix{4, 4, T, 16}`: Right-looking scattering matrix.
"""
struct ScatteringM{T}
    SLeft::SMatrix{4, 4, T, 16}
    SRight::SMatrix{4, 4, T, 16}
end

# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    _S_Interface_Matrix(Vf_A, Vb_A, Vf_B, Vb_B)

Returns the S-matrix of a single interface between material A and B. 
"""
function _S_Interface_Matrix(Vf_A::AbstractMatrix, Vb_A::AbstractMatrix,
                             Vf_B::AbstractMatrix, Vb_B::AbstractMatrix)

    T = promote_type(eltype(Vf_A), eltype(Vb_A), eltype(Vf_B), eltype(Vb_B))

    LHSs = SMatrix{4,4,T}(hcat(-Vf_B, Vb_A))
    RHSs = SMatrix{4,4,T}(hcat(-Vf_A, Vb_B))

    S = try
        LHSs \ RHSs
    catch e
        if e isa SingularException
            Fqr = qr(Matrix{T}(LHSs), ColumnNorm())
            SMatrix{4,4,T}(Fqr \ Matrix{T}(RHSs))
        else
            rethrow()
        end
    end

    return SMatrix{4,4,T}(S)
end

"""
    _SProp(layer, Qf, Qb)

Returns the S-matrix of propagation within a layer, given its eigenmodes Qf, Qb. 
"""
function _SProp(layer::AbstractLayer, Qf::AbstractVector, Qb::AbstractVector)

    d = layer.d

    φf1 = exp(im * Qf[1] * d)
    φf2 = exp(im * Qf[2] * d)
    φb1 = exp(-im * Qb[1] * d)
    φb2 = exp(-im * Qb[2] * d)

    Tc = promote_type(typeof(φf1), typeof(φf2), typeof(φb1), typeof(φb2))

    return SMatrix{4,4,Tc}(Diagonal(SVector{4,Tc}(φf1, φf2, φb1, φb2)))
end

"""
    _Redheffer(S1, S2)

Composes systems S1 and S2 by 2x2 block, through the Redheffer star product. 
Implemented for static matrices of size 4x4. 
"""
function _Redheffer(S1::SMatrix{4,4}, S2::SMatrix{4,4})

    T = promote_type(eltype(S1), eltype(S2))

    S1_11 = SMatrix{2,2,T}(S1[SOneTo(2), SOneTo(2)])
    S1_12 = SMatrix{2,2,T}(S1[SOneTo(2), SVector(3,4)])
    S1_21 = SMatrix{2,2,T}(S1[SVector(3,4), SOneTo(2)])
    S1_22 = SMatrix{2,2,T}(S1[SVector(3,4), SVector(3,4)])

    S2_11 = SMatrix{2,2,T}(S2[SOneTo(2), SOneTo(2)])
    S2_12 = SMatrix{2,2,T}(S2[SOneTo(2), SVector(3,4)])
    S2_21 = SMatrix{2,2,T}(S2[SVector(3,4), SOneTo(2)])
    S2_22 = SMatrix{2,2,T}(S2[SVector(3,4), SVector(3,4)])

    S_11 = S2_11 * ((I - S1_12 * S2_21) \ S1_11)
    S_12 = S2_12 + S2_11 * ((I - S1_12 * S2_21) \ S1_12 * S2_22)
    S_21 = S1_21 + S1_22 * ((I - S2_21 * S1_12) \ S2_21 * S1_11)
    S_22 = S1_22 * ((I - S2_21 * S1_12) \ S2_22)

    return vcat(hcat(S_11, S_12), hcat(S_21, S_22))
end


"""
    S_all_ingredients(stack::Vector{<:AbstractLayer}, modes::Vector{<:Modes})

Returns a tuple containing the S-Matrices of all interface and all propagation
regions of the stack. 

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, each entry contains a layer's description. 
- `modes::Vector{<:Modes}`: of length N containing Modes, the struct containing eigenmodes and eigenvectors.

# Returns
- `(Sint, Sprop)`: where Sint is a vector of length N containing interface S-matrices, 
  and Sprop is a vector of length N containing propagation S-matrices.
"""
function S_all_ingredients(stack::Vector{<:AbstractLayer}, modes::Vector{<:Modes})
    N = length(stack)
    Tc = promote_type(ComplexF64, eltype(modes[1].Qf), eltype(modes[1].Qb))

    Sprop = Vector{SMatrix{4,4,Tc,16}}(undef, N)
    for j in 1:N
        Sprop[j] = _SProp(stack[j], modes[j].Qf, modes[j].Qb)
    end

    Sint = Vector{SMatrix{4,4,Tc,16}}(undef, N - 1)
    for j in 2:N
        Sint[j-1] = _S_Interface_Matrix(modes[j-1].Vf, modes[j-1].Vb, modes[j].Vf, modes[j].Vb)
    end

    return Sint, Sprop
end

"""
    S_all_Matrix(stack::Vector{<:AbstractLayer}, modes::Vector{<:Modes})
    S_all_Matrix(stack::Vector{<:AbstractLayer}, modes::Vector{<:Modes},
                 Sint::Vector{<:SMatrix{4,4}}, Sprop::Vector{<:SMatrix{4,4}})

Composes the cumulative S-Matrices through the stack, from the left and right,
using the Redheffer Star Product. Matrices are referenced at the left interface of each layer.
Requires precomputed modes. If ingredient Sint and Sprop are not provided, computes them. 

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length N, each entry contains a layer's description. 
- `modes::Vector{<:Modes}`: of length N containing Modes, the struct containing eigenmodes and eigenvectors.
- `Sint::Vector{<:SMatrix{4,4}}`: of length N containing interface S-matrices
- `Sprop::Vector{<:SMatrix{4,4}}`: of length N containing propagation S-matrices

# Returns
- `Vector{ScatteringM{Tc}}(undef, N)`: of length N containing ScatteringM, 
  the struct containing left and right cumulative S-matrices.
"""
function S_all_Matrix(stack::Vector{<:AbstractLayer}, modes::Vector{<:Modes},
                      Sint::Vector{<:SMatrix{4,4}}, Sprop::Vector{<:SMatrix{4,4}})
    
    N = length(stack)
    Tc = promote_type(ComplexF64, eltype(modes[1].Qf), eltype(modes[1].Qb))
    smat = Vector{ScatteringM{Tc}}(undef, N)

    # ---- SLeft: forward prefix scan, O(N) ----
    SLefts = Vector{SMatrix{4,4,Tc,16}}(undef, N)
    SLefts[1] = SI
    S0 = _Redheffer(SI, Sprop[1])
    for j in 2:N
        Stot = _Redheffer(S0, Sint[j-1])
        SLefts[j] = Stot
        S0 = _Redheffer(Stot, Sprop[j])
    end

    # ---- SRight: backward suffix scan, O(N) ----
    SRights = Vector{SMatrix{4,4,Tc,16}}(undef, N)
    SRights[N] = SI
    for k in (N-1):-1:1
        effect = _Redheffer(Sprop[k], Sint[k])
        SRights[k] = _Redheffer(effect, SRights[k+1])
    end

    for j in 1:N
        smat[j] = ScatteringM(SLefts[j], SRights[j])
    end

    return smat
end

function S_all_Matrix(stack::Vector{<:AbstractLayer}, modes::Vector{<:Modes})
    Sint, Sprop = S_all_ingredients(stack, modes)
    return S_all_Matrix(stack, modes, Sint, Sprop)
end

end
