"""
mod_ClosedFormAssembly.jl -- multi-source lateral assembly on the uniform DFT
grid (`PLAN_closed_form_point_source_rewrite.md` §4.4).

For one source depth `zsrc_local` (every source at that depth shares one `G`),
`closed_form_G` (`mod_ClosedFormField.jl`) materializes `G(x,y,z')` on the
rigid `uniform_kxky_grid`; it is forward-FFT'd once to `Ĝ(kx,ky,z')` and any
number of sources at that depth are applied at once,

    E(x,y,z') = sheet.weight · F⁻¹{ Ĝ(kx,ky,z') · F{sheet.J} } ,

through `apply_ghat`. The grid/transform primitives (`uniform_kxky_grid`,
`_centering_ramp`, `_forward_transform`, `_inverse_transform`, the FFT plan
cache) and `point_current_sheet` are used directly from `mod_UniformSource.jl`
(kept as-is -- nothing in them is shape- or z-specific).

Several source DEPTHS each get their own `Ĝ` from the SAME
`CavityTransferCache` (no re-solve, just `amplitude_at` at a new `z`) and are
summed: `apply_closed_form`.
"""
module ClosedFormAssembly

export ClosedFormGreen, build_closed_form_green, apply_ghat, apply_closed_form, grid_offsets
export closed_form_field

import ..SourceHandling: Sheet, Dipole
import ..UniformSource
import ..UniformSource: point_current_sheet
import ..CavityTransfer: AbstractCavityTransferCache, CavityTransferCache1D, CavityTransferCache2D
import ..ClosedFormField: closed_form_G

using StaticArrays
using Base.Threads

"""
    ClosedFormGreen

`G`/`Ĝ` of one `(source_layer, zsrc_local, obs_layer)` on `grid`, at local
observation depths `z_obs`; shape `(Nx, Ny, Nz, 3, 3)` = (x, y, z, field
component, moment).
"""
struct ClosedFormGreen
    source_layer :: Int
    obs_layer    :: Int
    zsrc_local   :: Float64
    grid         :: NamedTuple
    z_obs        :: Vector{Float64}
    G            :: Array{ComplexF64,5}
    Ghat         :: Array{ComplexF64,5}
end

"Physical pixel offsets `x_i = (i − midx)·Δx`, `y_j = (j − midy)·Δy` of a `uniform_kxky_grid`."
grid_offsets(grid::NamedTuple) = (Float64[(i - grid.midx) * grid.Δx for i in 1:grid.Nx],
                                  Float64[(j - grid.midy) * grid.Δy for j in 1:grid.Ny])

"""
    build_closed_form_green(cache, obs_layer, zsrc_local, grid, z_obs; kwargs...) -> ClosedFormGreen

Materializes `G` on `grid`'s own pixel offsets and FFTs each of the 9
`(row, col)` slices, with the same centering ramp `apply_ghat` uses. `kwargs`
go to `closed_form_G` (2D cache: `box_lateral=(Δx,Δy)`, `lateral=:direct|:nufft`).
"""
function build_closed_form_green(cache::AbstractCavityTransferCache, obs_layer::Int, zsrc_local::Real,
                                 grid::NamedTuple, z_obs::AbstractVector{<:Real}; kwargs...)
    xs, ys = grid_offsets(grid)
    zo = collect(Float64, z_obs)
    G = closed_form_G(cache, obs_layer, zsrc_local, zo, xs, ys; kwargs...)
    Ghat = _forward_G(G, grid)
    return ClosedFormGreen(cache.source_layer, obs_layer, Float64(zsrc_local), grid, zo, G, Ghat)
end

function _forward_G(G::Array{ComplexF64,5}, grid::NamedTuple)
    Nx, Ny, Nz = size(G, 1), size(G, 2), size(G, 3)
    Ghat = Array{ComplexF64,5}(undef, Nx, Ny, Nz, 3, 3)
    ramp = UniformSource._centering_ramp(grid.kx, grid.ky, grid.L, grid.Ly)
    for zi in 1:Nz, r in 1:3, c in 1:3
        Ghat[:, :, zi, r, c] .= UniformSource._forward_transform(view(G, :, :, zi, r, c), grid.Δx, grid.Δy, ramp)
    end
    return Ghat
end

"""
    apply_ghat(grid, Ghat, sheet) -> Array{ComplexF64,4}   # (Nx, Ny, Nz, component)

`E = sheet.weight · F⁻¹{ Ĝ·F{sheet.J} }`, the 3×3 `Ĝ` genuinely varying node to
node. `sheet` must live on the same `grid`.
"""
function apply_ghat(grid::NamedTuple, Ghat::AbstractArray{ComplexF64,5}, sheet::Sheet, who::String="apply_ghat")
    Nx, Ny = grid.Nx, grid.Ny
    Δx, Δy, L, Ly = grid.Δx, grid.Δy, grid.L, grid.Ly
    Δkx, Δky = grid.Δkx, grid.Δky
    Nz = size(Ghat, 3)

    size(sheet.J) == (Nx, Ny, 3) || error(
        "$who: sheet.J has shape $(size(sheet.J)), but this grid is (Nx,Ny,3) = " *
        "$((Nx, Ny, 3)) -- build the sheet against this SAME grid (e.g. `point_current_sheet(grid, ...)`).")

    ramp = UniformSource._centering_ramp(grid.kx, grid.ky, L, Ly)

    Ĵ = Array{ComplexF64,3}(undef, Nx, Ny, 3)
    for q in 1:3
        Ĵ[:, :, q] .= UniformSource._forward_transform(view(sheet.J, :, :, q), Δx, Δy, ramp)
    end

    Ê = Array{ComplexF64,4}(undef, Nx, Ny, Nz, 3)
    @inbounds @threads for zi in 1:Nz
        for j in 1:Ny, i in 1:Nx
            Ghat_ij = SMatrix{3,3,ComplexF64}(view(Ghat, i, j, zi, :, :))
            v = Ghat_ij * SVector{3,ComplexF64}(Ĵ[i, j, 1], Ĵ[i, j, 2], Ĵ[i, j, 3])
            Ê[i, j, zi, 1] = v[1]; Ê[i, j, zi, 2] = v[2]; Ê[i, j, zi, 3] = v[3]
        end
    end

    E = Array{ComplexF64,4}(undef, Nx, Ny, Nz, 3)
    for zi in 1:Nz, q in 1:3
        E[:, :, zi, q] .= sheet.weight .* UniformSource._inverse_transform(view(Ê, :, :, zi, q), Δkx, Δky, Nx, Ny, ramp)
    end
    return E
end

apply_ghat(g::ClosedFormGreen, sheet::Sheet) = apply_ghat(g.grid, g.Ghat, sheet, "apply_ghat(ClosedFormGreen)")

"""
    apply_closed_form(cache, obs_layer, grid, z_obs, sheets) -> Array{ComplexF64,4}

Field in `obs_layer` at depths `z_obs` from any number of `Sheet`s in
`cache.source_layer`. Sheets are grouped by depth (`sheet.z`, local to the
source layer): one `Ĝ` per distinct depth, built from the same cache with no
re-solve; sources sharing a depth superpose through their sheets.
"""
function apply_closed_form(cache::AbstractCavityTransferCache, obs_layer::Int, grid::NamedTuple,
                           z_obs::AbstractVector{<:Real}, sheets::AbstractVector{Sheet}; kwargs...)
    E = zeros(ComplexF64, grid.Nx, grid.Ny, length(z_obs), 3)
    for zs in unique(s.z for s in sheets)
        g = build_closed_form_green(cache, obs_layer, zs, grid, z_obs; kwargs...)
        for s in sheets
            s.z == zs && (E .+= apply_ghat(g, s))
        end
    end
    return E
end


"""
    closed_form_field(caches, grid, z_obs_by_layer, dips; ghat_cache=nothing)
        -> Dict{Int, Array{ComplexF64,4}}     # obs_layer => (Nx, Ny, Nz_obs, component)

Multi-dipole, multi-layer field on `grid`. `caches[l]` is the
`CavityTransferCache1D`/`2D` of source layer `l` (one per layer that hosts a
dipole, each built with every layer of `z_obs_by_layer` among its
`obs_layers`); `z_obs_by_layer[o]` lists the local observation depths wanted
in layer `o`.

Dipoles are grouped by exact `(layer, z)`: each group becomes ONE sheet
(`point_current_sheet`, all of the group's lateral positions/moments) and one
`Ĝ` per obs layer, so co-planar dipoles cost a single FFT/multiply/iFFT.
Distinct depths reuse the same cache (no re-solve). Pass a `Dict` as
`ghat_cache` to keep the `Ĝ`s, keyed `(source_layer, z, obs_layer)`, across calls
(only with identical `kwargs`). `kwargs` go to `closed_form_G` -- e.g.
`box_lateral=(Δx,Δy)` makes every dipole a laterally-extended box on a 2D cache.
"""
function closed_form_field(caches::AbstractDict{Int,<:AbstractCavityTransferCache}, grid::NamedTuple,
                           z_obs_by_layer::AbstractDict{Int,<:AbstractVector{<:Real}},
                           dips::AbstractVector{<:Dipole};
                           ghat_cache::Union{Nothing,AbstractDict}=nothing, kwargs...)
    isempty(dips) && error("closed_form_field: need at least one source")
    groups = Dict{Tuple{Int,Float64},Vector{Tuple{Float64,Float64,SVector{3,ComplexF64}}}}()
    for dip in dips
        haskey(caches, dip.layer) || error("closed_form_field: no CavityTransferCache for source layer $(dip.layer)")
        push!(get!(() -> Tuple{Float64,Float64,SVector{3,ComplexF64}}[], groups, (dip.layer, dip.z)),
              (dip.x, dip.y, dip.p))
    end

    out = Dict(o => zeros(ComplexF64, grid.Nx, grid.Ny, length(zs), 3) for (o, zs) in z_obs_by_layer)
    for ((layer, z), points) in groups
        sheet = point_current_sheet(grid, z, points)
        for (o, zs) in z_obs_by_layer
            isempty(zs) && continue
            build() = build_closed_form_green(caches[layer], o, z, grid, zs; kwargs...)
            g = ghat_cache === nothing ? build() : get!(build, ghat_cache, (layer, z, o))
            out[o] .+= apply_ghat(g, sheet)
        end
    end
    return out
end

end
