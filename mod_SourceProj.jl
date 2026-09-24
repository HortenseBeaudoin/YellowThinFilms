module SourceProjection

export Source_projection_Berreman_xyz

import ..ModeSolver: Modes, R2s, AbstractMaterial
using StaticArrays


# TODO : Potential for simplification for isotropic materials (maybe uniaxial too)
# A symmetry seems to exist in the source projection for isotropic materials
# Also; for a non-mixed stack, ky=0, simplification of the full thing
# J could be split in a scalar way, similar to the Berreman function
# Lots of potential here.

# Unit dipole current templates (columns/entries = x̂, ŷ, ẑ), Berreman xyz basis.
# Constant, so hoisted out of the hot per-(layer,k-point) call below.
const _Jt_cols_unit = SMatrix{2,3,ComplexF64}(1,0, 0,1, 0,0)
const _Jz_row_unit  = SVector{3,ComplexF64}(0,0,1)

# ─────────────────────────────────────────────────────────────────────────────
#  Source projection and normalization
# ─────────────────────────────────────────────────────────────────────────────


#TODO Revise if this is still used for "general" sources or not. 
"""
Projects unit dipole currents along x, y and z onto the Berreman mode basis,
returning the forward/backward modal seed amplitudes for all three orientations
at once (one factorization, three right-hand sides).

Returns (sif, sib), each a 2x3 matrix whose columns correspond to a unit dipole
oriented along x, y and z respectively.
"""
function Source_projection_Berreman_xyz(kt::AbstractVector, ω::Real, ϵ::Union{AbstractMatrix, AbstractMaterial},
                           modes::Modes)

    ϵtz = SVector{2,ComplexF64}(ϵ[1,3], ϵ[2,3])
    ϵzz = ϵ[3,3]
    kts = SVector{2,ComplexF64}(kt[1], kt[2])

    # Columns = unit dipoles p = x̂, ŷ, ẑ
    Jt_cols = (-1im * ω) .* _Jt_cols_unit
    Jz_row  = (-1im * ω) .* _Jz_row_unit

    J_Et = (-1im * kts) * transpose(Jz_row) / (ϵzz * ω)
    J_Ht = (1im * R2s * ϵtz) * transpose(Jz_row) / ϵzz - 1im * R2s * Jt_cols
    J_Berreman = vcat(J_Et, J_Ht)

    V = hcat(modes.Vf, modes.Vb)
    W = hcat(modes.Wf, modes.Wb)

    T  = promote_type(eltype(V), eltype(W), eltype(J_Berreman))
    Ws = SMatrix{4,4,T}(W)
    Vs = SMatrix{4,4,T}(V)
    Js = SMatrix{4,3,T}(J_Berreman)

    s_all = (Ws' * Vs) \ (Ws' * Js)

    return SMatrix{2,3,ComplexF64}(s_all[1:2, :]), SMatrix{2,3,ComplexF64}(s_all[3:4, :])
end


end
