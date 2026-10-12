"""
mod_ClosedFormAssembly.jl -- multi-source lateral assembly on the uniform DFT
grid (`PLAN_closed_form_point_source_rewrite.md` §4.4).

For one source depth `zsrc_local` (every source at that depth shares one `G`),
`closed_form_G` (`mod_ClosedFormField.jl`) materializes `G` at every lateral
OFFSET the grid can produce, and it is FFT'd once to `Ĝ`. Any number of
sources at that depth are then applied at once as a LINEAR convolution,

    E(x_i, z') = Σ_j G(x_i − x_j, z') · J(x_j) · Δx·Δy ,

computed as a zero-padded circular convolution: on an `Nx×Ny` grid the offsets
`x_i − x_j` span `−(Nx−1)…(Nx−1)` pixels, so `G` is sampled on a `2Nx×2Ny`
offset grid (FFT-native order: index 1 is offset 0) and `J` is zero-padded to
the same size. With `Px ≥ 2Nx−1` no offset wraps onto another, so the result
on the original `Nx×Ny` pixels is exact (no wraparound from the opposite edge).
The normalisation `Δx·Δy/(Px·Py)` is folded into `Ĝ`, so an apply is just

    E = bfft( Σ_depths Ĝ(z_s, z') · fft(J_pad(z_s)) )   cropped to Nx×Ny.

Contributions of all source depths are summed in k-space, so each observation
plane is inverse-transformed once, whatever the number of source planes.

Workflow: `precompute_ghats!` builds every `Ĝ` for declared source depths and
observation layers up front; `closed_form_field` with that `ghat_cache` then
only does FFT · multiply · iFFT. Several source DEPTHS each get their own `Ĝ`
from the SAME `CavityTransferCache` (no re-solve, just `amplitude_at` at a new `z`).
"""
module ClosedFormAssembly

export ClosedFormGreen, build_closed_form_green, apply_ghat, apply_closed_form, grid_offsets
export closed_form_field, precompute_ghats!, padded_offsets

import ..SourceHandling: Sheet, Dipole
import ..UniformSource: point_current_sheet
import ..CavityTransfer: AbstractCavityTransferCache, CavityTransferCache1D, CavityTransferCache2D
import ..ClosedFormField: closed_form_G

using StaticArrays
using Base.Threads
using FFTW

"""
    ClosedFormGreen

`Ĝ` of one `(source_layer, zsrc_local, obs_layer)` for `grid`, at local
observation depths `z_obs`: shape `(Px, Py, Nz, 3, 3)` = (padded kx, padded ky,
z, field component, moment), `Px = 2Nx`, `Py = 2Ny`, with `Δx·Δy/(Px·Py)`
folded in (see the module docstring). The real-space `G` is not kept.
"""
struct ClosedFormGreen
    source_layer :: Int
    obs_layer    :: Int
    zsrc_local   :: Float64
    grid         :: NamedTuple
    z_obs        :: Vector{Float64}
    Ghat         :: Array{ComplexF64,5}
end

"Physical pixel offsets `x_i = (i − midx)·Δx`, `y_j = (j − midy)·Δy` of a `uniform_kxky_grid`."
grid_offsets(grid::NamedTuple) = (Float64[(i - grid.midx) * grid.Δx for i in 1:grid.Nx],
                                  Float64[(j - grid.midy) * grid.Δy for j in 1:grid.Ny])

"Padded transform size `(Px, Py) = (2Nx, 2Ny)` of `grid`."
_padded_size(grid::NamedTuple) = (2 * grid.Nx, 2 * grid.Ny)

"""
    padded_offsets(grid) -> (xs, ys)

The `2Nx × 2Ny` lateral offsets `G` is sampled on, centred and ascending
(`xs[1 + Nx] = 0`), as `closed_form_G` (and its NUFFT path) want them. They
cover every source–observation offset of `grid`, `−(Nx−1)…(Nx−1)` pixels,
plus one unused row `−Nx`.
"""
function padded_offsets(grid::NamedTuple)
    Px, Py = _padded_size(grid)
    return (Float64[(i - (1 + Px ÷ 2)) * grid.Δx for i in 1:Px],
            Float64[(j - (1 + Py ÷ 2)) * grid.Δy for j in 1:Py])
end

# ─────────────────────────────────────────────────────────────────────────────
#  FFT plans: in place, over dims (1,2), threaded, cached per array size
# ─────────────────────────────────────────────────────────────────────────────
const _PLANS = Dict{Tuple{Symbol,Tuple},Any}()
const _PLAN_LOCK = ReentrantLock()

function _plan(kind::Symbol, A::Array{ComplexF64})
    lock(_PLAN_LOCK) do
        get!(_PLANS, (kind, size(A))) do
            FFTW.set_num_threads(Threads.nthreads())
            scratch = similar(A)            # MEASURE overwrites its input
            kind === :fft ? plan_fft!(scratch, (1, 2); flags=FFTW.MEASURE) :
                            plan_bfft!(scratch, (1, 2); flags=FFTW.MEASURE)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  Ĝ construction (precompute)
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_closed_form_green(cache, obs_layer, zsrc_local, grid, z_obs; kwargs...) -> ClosedFormGreen

Materializes `G` on the padded offset grid (`padded_offsets`) and FFTs it in
place. The centred → FFT-native shift by `(Px/2, Py/2)` is applied in k-space as
the sign `(−1)^(i+j)` (exact, `Px`, `Py` even). `kwargs` go to `closed_form_G`
(2D cache: `box_lateral=(Δx,Δy)`, `Δz_box`, `lateral=:direct|:nufft`).
"""
function build_closed_form_green(cache::AbstractCavityTransferCache, obs_layer::Int, zsrc_local::Real,
                                 grid::NamedTuple, z_obs::AbstractVector{<:Real}; kwargs...)
    xs, ys = padded_offsets(grid)
    zo = collect(Float64, z_obs)
    G = closed_form_G(cache, obs_layer, zsrc_local, zo, xs, ys; kwargs...)
    _plan(:fft, G) * G
    Px, Py = size(G, 1), size(G, 2)
    scale = grid.Δx * grid.Δy / (Px * Py)
    @threads for j in 1:Py
        @inbounds for c in axes(G, 5), r in axes(G, 4), zi in axes(G, 3), i in 1:Px
            G[i, j, zi, r, c] *= iseven(i + j) ? scale : -scale
        end
    end
    return ClosedFormGreen(cache.source_layer, obs_layer, Float64(zsrc_local), grid, zo, G)
end

"""
    precompute_ghats!(ghat_cache, caches, grid, z_src_by_layer, z_obs_by_layer; kwargs...) -> ghat_cache

Builds `Ĝ` for every declared source depth `z_src_by_layer[l]` of every source
layer `l` (with `caches[l]`) and every observation layer of `z_obs_by_layer`,
keyed `(source_layer, z, obs_layer)` -- the keys `closed_form_field` looks up.
Already-present keys are kept. Pass the same `kwargs` to `closed_form_field`.
"""
function precompute_ghats!(ghat_cache::AbstractDict, caches::AbstractDict{Int,<:AbstractCavityTransferCache},
                           grid::NamedTuple, z_src_by_layer::AbstractDict{Int,<:AbstractVector{<:Real}},
                           z_obs_by_layer::AbstractDict{Int,<:AbstractVector{<:Real}}; kwargs...)
    for (l, zss) in z_src_by_layer, z in zss, (o, zos) in z_obs_by_layer
        isempty(zos) && continue
        haskey(ghat_cache, (l, Float64(z), o)) && continue
        ghat_cache[(l, Float64(z), o)] = build_closed_form_green(caches[l], o, z, grid, zos; kwargs...)
    end
    return ghat_cache
end

# ─────────────────────────────────────────────────────────────────────────────
#  Apply: FFT · multiply · iFFT
# ─────────────────────────────────────────────────────────────────────────────

function _check_sheet(grid::NamedTuple, sheet::Sheet, who::String)
    size(sheet.J) == (grid.Nx, grid.Ny, 3) || error(
        "$who: sheet.J has shape $(size(sheet.J)), but this grid is (Nx,Ny,3) = " *
        "$((grid.Nx, grid.Ny, 3)) -- build the sheet against this SAME grid (e.g. `point_current_sheet(grid, ...)`).")
end

"`fft` of `weight·J`, zero-padded to `(Px, Py, 3)`."
function _source_spectrum(grid::NamedTuple, sheet::Sheet)
    Px, Py = _padded_size(grid)
    Ĵ = zeros(ComplexF64, Px, Py, 3)
    @views Ĵ[1:grid.Nx, 1:grid.Ny, :] .= sheet.weight .* sheet.J
    _plan(:fft, Ĵ) * Ĵ
    return Ĵ
end

"`Ê[:, :, zi, :] += Ĝ[:, :, zi, :, :] · Ĵ` at every padded node (3×3 times 3-vector)."
function _accumulate!(Ê::Array{ComplexF64,4}, Ghat::Array{ComplexF64,5}, Ĵ::Array{ComplexF64,3})
    Px, Py, Nz = size(Ghat, 1), size(Ghat, 2), size(Ghat, 3)
    @threads for t in 1:(Py * Nz)
        j  = (t - 1) % Py + 1
        zi = (t - 1) ÷ Py + 1
        @inbounds for i in 1:Px
            J1, J2, J3 = Ĵ[i, j, 1], Ĵ[i, j, 2], Ĵ[i, j, 3]
            for r in 1:3
                Ê[i, j, zi, r] += Ghat[i, j, zi, r, 1] * J1 + Ghat[i, j, zi, r, 2] * J2 + Ghat[i, j, zi, r, 3] * J3
            end
        end
    end
    return Ê
end

"Inverse transform of the accumulated `Ê` (in place) and crop to the `Nx×Ny` pixels."
function _field_from_spectrum!(Ê::Array{ComplexF64,4}, grid::NamedTuple)
    _plan(:bfft, Ê) * Ê
    return Ê[1:grid.Nx, 1:grid.Ny, :, :]
end

"""
    apply_ghat(g::ClosedFormGreen, sheet) -> Array{ComplexF64,4}   # (Nx, Ny, Nz, component)

Field of one sheet (at `g`'s source depth) on `g.grid`. `sheet` must live on the same grid.
"""
function apply_ghat(g::ClosedFormGreen, sheet::Sheet)
    _check_sheet(g.grid, sheet, "apply_ghat")
    Ê = zeros(ComplexF64, size(g.Ghat, 1), size(g.Ghat, 2), size(g.Ghat, 3), 3)
    _accumulate!(Ê, g.Ghat, _source_spectrum(g.grid, sheet))
    return _field_from_spectrum!(Ê, g.grid)
end

"""
    apply_closed_form(cache, obs_layer, grid, z_obs, sheets) -> Array{ComplexF64,4}

Field in `obs_layer` at depths `z_obs` from any number of `Sheet`s in
`cache.source_layer`. Sheets are grouped by depth (`sheet.z`, local to the
source layer): one `Ĝ` per distinct depth, built from the same cache with no
re-solve; everything is summed in k-space before the single inverse transform.
"""
function apply_closed_form(cache::AbstractCavityTransferCache, obs_layer::Int, grid::NamedTuple,
                           z_obs::AbstractVector{<:Real}, sheets::AbstractVector{Sheet}; kwargs...)
    Px, Py = _padded_size(grid)
    Ê = zeros(ComplexF64, Px, Py, length(z_obs), 3)
    for zs in unique(s.z for s in sheets)
        g = build_closed_form_green(cache, obs_layer, zs, grid, z_obs; kwargs...)
        for s in sheets
            s.z == zs || continue
            _check_sheet(grid, s, "apply_closed_form")
            _accumulate!(Ê, g.Ghat, _source_spectrum(grid, s))
        end
    end
    return _field_from_spectrum!(Ê, grid)
end


"""
    closed_form_field(caches, grid, z_obs_by_layer, dips; ghat_cache=nothing, kwargs...)
        -> Dict{Int, Array{ComplexF64,4}}     # obs_layer => (Nx, Ny, Nz_obs, component)

Multi-dipole, multi-layer field on `grid`. `caches[l]` is the
`CavityTransferCache1D`/`2D` of source layer `l` (one per layer that hosts a
dipole, each built with every layer of `z_obs_by_layer` among its
`obs_layers`); `z_obs_by_layer[o]` lists the local observation depths wanted
in layer `o`.

Dipoles are grouped by exact `(layer, z)`: each group becomes ONE sheet
(`point_current_sheet`, all of the group's lateral positions/moments), FFT'd
once; each obs layer accumulates `Ĝ·Ĵ` over all groups in k-space and is
inverse-transformed once. Sources may sit anywhere on the grid (linear, not
circular, convolution). Pass a `Dict` as `ghat_cache` to keep the `Ĝ`s, keyed
`(source_layer, z, obs_layer)`, across calls -- fill it beforehand with
`precompute_ghats!` so that this call does no grid work at all (only with
identical `kwargs` and `z_obs_by_layer`). `kwargs` go to `closed_form_G` --
e.g. `box_lateral=(Δx,Δy)` makes every dipole a laterally-extended box on a 2D cache.
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

    Px, Py = _padded_size(grid)
    obs = [(o, zs) for (o, zs) in z_obs_by_layer if !isempty(zs)]
    Ê = Dict(o => zeros(ComplexF64, Px, Py, length(zs), 3) for (o, zs) in obs)
    for ((layer, z), points) in groups
        Ĵ = _source_spectrum(grid, point_current_sheet(grid, z, points))
        for (o, zs) in obs
            build() = build_closed_form_green(caches[layer], o, z, grid, zs; kwargs...)
            g = ghat_cache === nothing ? build() : get!(build, ghat_cache, (layer, z, o))
            _accumulate!(Ê[o], g.Ghat, Ĵ)
        end
    end
    out = Dict(o => zeros(ComplexF64, grid.Nx, grid.Ny, length(zs), 3) for (o, zs) in z_obs_by_layer)
    for (o, _) in obs
        out[o] = _field_from_spectrum!(Ê[o], grid)
    end
    return out
end

end
