module SourceHandling

export Dipole_constructor
export SourceDef, Dipole, ModalSeed
export Sheet

using StaticArrays

abstract type SourceDef end

# ────────────────────────── GENERAL SOURCE INPUT ───────────────────────────────────

# Source definition adapter to the solver, most source type should have translator to this type
# x-y wrapped into sif and sib (to confirm)
struct ModalSeed <: SourceDef
    layer :: Int
    z     :: Float64
    sif   :: Vector{ComplexF64}
    sib   :: Vector{ComplexF64}
end

# TODO Maybe reframe this as a "moment" constructor rather than dipole, adjust the sheet as well    
# ────────────────────────── DIPOLE ─────────────────────────────

struct Dipole <:SourceDef
    p    :: SVector{3,ComplexF64}
    ω    :: Real
    x   :: Float64
    z   :: Float64
    layer:: Int
    y   :: Float64
end

# Default constructor for Dipole with y=0.0
Dipole(p, ω, x, z, layer) = Dipole(p, ω, x, z, layer, 0.0)

# Dipole constructor function, that accepts either amplitude vector or a symbol for the dipole orientation
function Dipole_constructor(p, ω::Real, x::Real, z::Real, layer::Int; y::Real=0.0, amp::Number=1.0)
    pvec = if p isa Symbol
        p === :x ? SVector(amp, 0, 0) :
        p === :y ? SVector(0, amp, 0) :
        p === :z ? SVector(0, 0, amp) :
        error("Dipole_constructor: unknown orientation symbol $p (expected :x, :y or :z)")
    else
        SVector{3}(p)
    end
    return Dipole(SVector{3,ComplexF64}(pvec), ω, Float64(x), Float64(z), layer, Float64(y))
end


# ─────────────────────── ARBITRARY REAL-SPACE SOURCE ────────────────

"""
    Sheet(J, z; weight=1.0)

J[:, :, 1:3] = (Jx, Jy, Jz)`, shape `(Nx, Ny, 3)` = (x, y, component), sampled
on the source's uniform lateral grid, sitting at LOCAL depth `z` (top-
referenced within the host layer, same convention as `Dipole.z`).
"""
struct Sheet
    J      :: Array{ComplexF64,3}       # Nx x Ny x 3 array of (Jx, Jy, Jz) on the lateral grid
    z      :: Float64
    weight :: Float64
end

Sheet(J::AbstractArray, z::Real; weight::Real=1.0) =
    Sheet(ComplexF64.(J), Float64(z), Float64(weight))


end
