module Sheets

export Sheet, uniform_kxky_grid, point_current_sheet

# ─────────────────────────────────────────────────────────────────────────────
#  Constant and structures declaration
# ─────────────────────────────────────────────────────────────────────────────

"""
    Sheet(J, z)

`J[:, :, 1:3] = (Jx, Jy, Jz)`, shape `(Nx, Ny, 3)` = (x, y, component): a current
density sampled on a `uniform_kxky_grid`, at local depth `z` (top-referenced
within the host layer, same convention as `Dipole.z`).

# Fields
- `J::Array{ComplexF64,3}`: Nx x Ny x 3 array of (Jx, Jy, Jz) on the lateral grid.
- `z::Float64`: local depth `z` of the sheet.
"""
struct Sheet
    J::Array{ComplexF64,3}
    z::Float64
end

Sheet(J::AbstractArray{<:Number,3}, z::Real) =
    Sheet(ComplexF64.(J), Float64(z))

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    _nearest_node_index(grid, x, y) -> (i, j)

Pixel nearest to the physical position `(x,y)`: `i = round(Int, x/Δx) + midx`.
Exact when `(x,y)` is on a node, otherwise moves the point by up to `Δx/2`
(`Δy/2`). Warns and clamps if the point falls outside `[-L,L)×[-Ly,Ly)`.
"""
function _nearest_node_index(grid::NamedTuple, x::Real, y::Real)
    i = round(Int, x / grid.Δx) + grid.midx
    j = round(Int, y / grid.Δy) + grid.midy
    if !(1 <= i <= grid.Nx) || !(1 <= j <= grid.Ny)
        @warn "point_current_sheet: position (x=$x, y=$y) snaps to pixel ($i,$j), outside this grid's Nx=$(grid.Nx)/Ny=$(grid.Ny) -- clamping into range. Enlarge L/Ly if this point should be included."
        i = clamp(i, 1, grid.Nx)
        j = clamp(j, 1, grid.Ny)
    end
    return i, j
end

# Makes sure that source moment is complex
_point_moment_vec(p::AbstractVector) = ComplexF64.(p)
_point_moment_vec(p::Symbol) =
    p === :x ? ComplexF64[1, 0, 0] :
    p === :y ? ComplexF64[0, 1, 0] :
    p === :z ? ComplexF64[0, 0, 1] :
    error("point_current_sheet: unknown moment symbol $p (expected :x, :y or :z)")

# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    uniform_kxky_grid(; L, Δx, Ly=L, Δy=Δx) -> NamedTuple

Lateral pixel grid, set by `L` and `Δx`. By default, set for x and copied
by y. Option to set `Ly` and `Δy` independently.

# Arguments
- `L::Real`: computation windows half-width in λ₀.
- `Δx::Real`: smallest feature to resolve/desired pixel size in λ₀.

# Returns
- `(Nx, Ny, Δx, Δy, midx, midy, L, Ly)`: Nx and Ny are the number of pixel in 
  x and y, Δx and Δx the computed resolution 2L/N, midx and midy the center's id, 
  and L and Ly the half-width of the window. 
"""
function uniform_kxky_grid(; L::Real, Δx::Real, Ly::Real=L, Δy::Real=Δx)
    Lf, Lyf  = Float64(L), Float64(Ly)
    Δxf, Δyf = Float64(Δx), Float64(Δy)
    (Lf > 0 && Lyf > 0 && Δxf > 0 && Δyf > 0) ||
        error("uniform_kxky_grid: L, Δx, Ly, Δy must all be positive.")

    Nx = ceil(Int, 2Lf / Δxf)
    Ny = ceil(Int, 2Lyf / Δyf)
    (Nx >= 2 && Ny >= 2) ||
        error("uniform_kxky_grid: derived grid too small (Nx=$Nx, Ny=$Ny) -- increase L or decrease Δx (and the y analogues).")

    return (Nx=Nx, Ny=Ny, Δx=2Lf / Nx, Δy=2Lyf / Ny,
            midx=1 + fld(Nx, 2), midy=1 + fld(Ny, 2), L=Lf, Ly=Lyf)
end

"""
    point_current_sheet(grid, z, points) -> Sheet

Build a current `Sheet` from point moments. Each point is snapped 
to the nearest pixel of `grid`.

# Arguments
- `grid::NamedTuple`: a uniform in-plane grid.
- `z::Real`: local depth `z` of the sheet.
- `points::AbstractVector`: a vector of a single of multiple `(x, y, p)`, 
  where `p` is a 3-vector or `:x`/`:y`/`:z`. Points on the same pixel are summed.

# Returns
- `Sheet`: current density sheet at depth `z`.
"""
function point_current_sheet(grid::NamedTuple, z::Real, points::AbstractVector)
    J = zeros(ComplexF64, grid.Nx, grid.Ny, 3)
    for (x, y, p) in points
        i, j = _nearest_node_index(grid, x, y)
        J[i, j, :] .+= _point_moment_vec(p) ./ (grid.Δx * grid.Δy)
    end
    return Sheet(J, z)
end


end
