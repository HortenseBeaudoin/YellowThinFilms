module Dipoles

export Dipole_constructor
export Dipole

using StaticArrays

# ────────────────────────── POINT SOURCE ─────────────────────────────

"""
    Dipole(p, ω, x, z, layer, y)
    Dipole(p, ω, x, z, layer)

Source point definition.

# Fields
- `p::SVector{3,ComplexF64}`: dipole's moment p, [px, py, pz].
- `ω::Real`: frequency. 
- `x::Float64`: dipole's position in x. 
- `z::Float64`: source depth, following the convention
    For finite layers, n∈(`2:N-1`): distance from the layer's z = 0 edge, left interface. 
    For semi-infinite layers, n∈(`1, N`): distance from the stack, whether it sits on the left of right.
- `layer::Int`: indicates the layer, half-space/substrate included. The index in stack of length N. 
- `y::Float64`: dipole's position in y. 

See docs for the grid's position reference. 
"""
struct Dipole
    p::SVector{3,ComplexF64}
    ω::Real
    x::Float64
    z::Float64
    layer::Int
    y::Float64
end

Dipole(p, ω, x, z, layer) = Dipole(p, ω, x, z, layer, 0.0)

"""
    Dipole_constructor(p, ω, x, z, layer; y) -> Dipole

Returns a dipole struct from either a full moment input, of a single :x, :y or :z
direction. Position in y optinal, defaults to 0.
"""
function Dipole_constructor(p, ω::Real, x::Real, z::Real, layer::Int; y::Real=0.0)
    pvec = if p isa Symbol
        p === :x ? SVector(1.0, 0, 0) :
        p === :y ? SVector(0, 1.0, 0) :
        p === :z ? SVector(0, 0, 1.0) :
        error("Dipole_constructor: unknown orientation symbol $p (expected :x, :y or :z)")
    else
        SVector{3}(p)
    end
    return Dipole(SVector{3,ComplexF64}(pvec), ω, Float64(x), Float64(z), layer, Float64(y))
end

end
