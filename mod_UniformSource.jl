"""
mod_UniformSource.jl -- the rigid uniform-DFT lateral grid + FFT transform
primitives of the multi-source assembly (`mod_ClosedFormAssembly.jl`, plan §4.4).

For a real-space quantity `J(x,y)` on a rigid uniform lateral grid,
`E(x) = F⁻¹{ G(kx,ky)·F{J} }` where `F`/`F⁻¹` are the SAME uniform DFT -- so
`F⁻¹F = identity` holds to machine precision. `uniform_kxky_grid` (the grid
itself), `_fft_wavenumbers`/`_centering_ramp`/`_nearest_node_index`/
`_point_moment_vec` (its supporting conventions), `_forward_transform`/
`_inverse_transform` (the `F`/`F⁻¹` pair) and `point_current_sheet`.
"""
module UniformSource

export uniform_kxky_grid, point_current_sheet

import ..SourceHandling: Sheet

using FFTW

# ─────────────────────────────────────────────────────────────────────────────
#  FFT plan cache (2026-09-08, voxel-path speedup pass): bare `fft`/`ifft`
#  replan from scratch EVERY call (FFTW.jl's `fft(x)` is sugar for
#  `plan_fft(x) * x`). `_forward_transform`/`_inverse_transform` below are
#  called MANY times against the SAME `(Nx,Ny)` grid shape -- once per
#  `(z_obs,row,col)` in the zpos-grid build's inner FFT loop, and
#  `3 + 3·Nz` times per `_apply_ghat` call -- so replanning every time is pure
#  waste. Cached per `(Nx,Ny)`, guarded by a lock since plan CREATION touches
#  FFTW's global planner state; applying an already-built plan (`p * x`) is
#  safe without the lock and works on any array of matching size/strides
#  (that's the whole point of planning -- see FFTW.jl's own `plan_fft` docs),
#  so a view like `view(G, :, :, zi, r, c)` (contiguous in its leading two
#  dims) applies against a plan built from a plain `Matrix` just fine.
# ─────────────────────────────────────────────────────────────────────────────
const _FFT_PLAN_CACHE = Dict{Tuple{Int,Int},Any}()
const _IFFT_PLAN_CACHE = Dict{Tuple{Int,Int},Any}()
const _FFT_PLAN_LOCK = ReentrantLock()

_cached_fft_plan(Nx::Int, Ny::Int) =
    lock(_FFT_PLAN_LOCK) do
        get!(() -> plan_fft(Array{ComplexF64}(undef, Nx, Ny)), _FFT_PLAN_CACHE, (Nx, Ny))
    end

_cached_ifft_plan(Nx::Int, Ny::Int) =
    lock(_FFT_PLAN_LOCK) do
        get!(() -> plan_ifft(Array{ComplexF64}(undef, Nx, Ny)), _IFFT_PLAN_CACHE, (Nx, Ny))
    end

# ─────────────────────────────────────────────────────────────────────────────
#  The rigid uniform (kx,ky) grid -- single source of truth
# ─────────────────────────────────────────────────────────────────────────────

"""
    uniform_kxky_grid(; L, Δx, Ly=L, Δy=Δx)
        -> (kx, ky, Nx, Ny, Δkx, Δky, kx_max, ky_max, midx, midy, L, Ly, Δx, Δy)

The single source of truth for the rigid uniform-DFT grid this whole path
rests on -- fully determined by two user knobs. `L`/`Δx`
(lateral field-of-view half-width / finest resolved feature, per axis; `Ly`,
`Δy` independent for `y`) are the only two physical knobs -- everything else
is DERIVED:

  - `Nx = ceil(2L/Δx)` samples over the real-space window `[-L,L)` (at least
    as fine as requested; the `y` analogue gives `Ny`). `Δx` is then
    RE-DERIVED as `2L/Nx` (the effective pitch an integer `Nx` actually
    realizes) -- this is what makes `Δkx·Δx·Nx == 2π` hold to machine
    precision, not merely approximately, which the round-trip identity
    below needs exactly.
  - `Δkx = π/L` -- the reciprocal spacing (anti-aliasing / periodic-
    wraparound control);
  - `kx_max = π/Δx` -- the cutoff, i.e. the evanescent-content cutoff: asking
    for `Δx = λ/8` resolution IS asking for `kx_max ≈ 4·k0` -- the same
    statement, not independent knobs. `Δx = λ/8` reaches only ~4 evanescent
    units past k0; for a source very close to an interface, reflected
    evanescent content past that carries real weight and gets truncated by a
    grid this coarse. This is not a bug -- near-field resolution and
    evanescent content ARE the same statement.

`kx[i]`/`ky[j]` are returned in FFTW/AbstractFFTs' own native output ordering
(`0, Δk, 2Δk, ..., -⌈N/2⌉·Δk, ..., -Δk`) -- so `kx[i]` IS the physical
wavenumber at array index `i` of an `fft`/`ifft` call on this grid, no
reindexing needed downstream. `midx = 1 + fld(Nx,2)` is the real-space
pixel-centering constant such that physical `x_i = (i - midx)·Δx` places the
window symmetrically on `[-L, L)` -- "the same centering algebra applied in
reverse" relative to `_nufft_nodes_2d`'s `a_x`/`a_y` (`mod_ClosedFormField.jl`).

Unit-test this directly against the DFT relations it encodes
(`Δkx·Δx·Nx == 2π`, etc.) before trusting anything built on top of it.
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
    Δxf = 2Lf / Nx
    Δyf = 2Lyf / Ny

    Δkx = π / Lf
    Δky = π / Lyf
    kx_max = π / Δxf
    ky_max = π / Δyf

    kx = _fft_wavenumbers(Nx, Δkx)
    ky = _fft_wavenumbers(Ny, Δky)

    midx = 1 + fld(Nx, 2)
    midy = 1 + fld(Ny, 2)

    return (kx=kx, ky=ky, Nx=Nx, Ny=Ny, Δkx=Δkx, Δky=Δky,
            kx_max=kx_max, ky_max=ky_max, midx=midx, midy=midy,
            L=Lf, Ly=Lyf, Δx=Δxf, Δy=Δyf)
end

"""
FFTW/AbstractFFTs-native wavenumber ordering for an `N`-point uniform grid
with reciprocal spacing `Δk`: index `i` (1-based) holds `k = 0, Δk, 2Δk, ...`
for `i = 1:fld(N,2)`, then wraps to negative frequencies `-⌈N/2⌉·Δk, ..., -Δk`
for the rest -- exactly `fft`'s own output index-to-frequency mapping (matches
`AbstractFFTs.fftfreq(N, 1) .* (2π/(N·Δx))`, written out explicitly here so
the convention is auditable rather than dependent on that function's exact
semantics for odd `N`).
"""
function _fft_wavenumbers(N::Int, Δk::Float64)
    half = fld(N, 2)
    return Float64[Δk * (i <= half ? i - 1 : i - 1 - N) for i in 1:N]
end

"""
The real-space-window centering ramp (the FFT-input analogue of
`_nufft_nodes_2d`'s `ramp`, mod_ClosedFormField.jl): `fft`/`ifft` assume the array's
pixel `i` sits at `x' = (i-1)·Δx ∈ [0, 2L)`, but this path's physical window is
`[-L, L)` (pixel `i` at `x = x' - L`, `midx`-centered). By the shift theorem
this is exactly a linear phase `cis(kx·L + ky·Ly)` per node, applied once on
the way into `fft` (`rampfwd`) and its conjugate once on the way out of `ifft`
(`rampfwd` is not applied to `ifft`'s OUTPUT -- see `_forward_transform`/
`_inverse_transform`). Purely a bookkeeping convention: cancels exactly in a
round-trip (`_inverse_transform ∘ identity ∘ _forward_transform == identity`
regardless of its value), so it plays no role in an identity-kernel
round-trip; it matters once a real `G` sits between forward and inverse.
"""
function _centering_ramp(kx::Vector{Float64}, ky::Vector{Float64}, L::Float64, Ly::Float64)
    Nx, Ny = length(kx), length(ky)
    ramp = Matrix{ComplexF64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        ramp[i, j] = cis(kx[i] * L + ky[j] * Ly)
    end
    return ramp
end

"""
    _nearest_node_index(grid, x, y) -> (i, j)

Snaps a physical position `(x,y)` to its nearest pixel on `grid` (a
`uniform_kxky_grid`-shaped NamedTuple): `i = round(Int, x/Δx) +
midx`. There is no sub-pixel placement in this path (`Sheet` is a
pre-sampled array, not a callable/analytic position -- see `mod_Sources.jl`'s
`Sheet` docstring), so this snap is exact only when `(x,y)` already sits on
a grid node (as it does at the origin, since `midx`/`midy` are integers by
construction); otherwise it introduces a position error of up to `Δx/2`
(`Δy/2`). WARNS and clamps into range if `(x,y)` falls outside the window
`[-L,L)×[-Ly,Ly)` rather than erroring, since placing points near a small
validation grid's edge is a normal, non-fatal situation -- enlarge `L`/`Ly`
if the point should actually be included.
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

_point_moment_vec(p::AbstractVector) = ComplexF64.(p)
_point_moment_vec(p::Symbol) = Dict(:x => ComplexF64[1, 0, 0], :y => ComplexF64[0, 1, 0], :z => ComplexF64[0, 0, 1])[p]

"""
    point_current_sheet(grid, z, points; weight=1.0) -> Sheet

Convenience constructor for the common case: a `Sheet` built from one or
more POINT moments at given physical `(x,y)` positions, rather than a
hand-built density array. This is what makes declaring a point-like source
on this path as straightforward as declaring a `Dipole` -- you supply a
moment, not a density.

`grid` is a `uniform_kxky_grid`-shaped NamedTuple -- pass the SAME grid every
downstream consumer of this sheet was built against, so it lands on the
correct pixels. `points` is a vector of
`(x, y, p)` tuples: `(x,y)` the point's physical position (snapped to the
nearest grid node, see `_nearest_node_index`), `p` either a 3-vector
(`ComplexF64[px,py,pz]`) or a `Symbol` (`:x`/`:y`/`:z`, unit moment along
that axis -- same shorthand `Dipole_constructor` offers).

Handles the density conversion for you: `Sheet.J` is a current DENSITY
(moment per unit area, see `Sheet`'s own docstring and `_forward_transform`'s
`Δx·Δy` factor), so a point of moment `p` is written into its pixel as
`p/(Δx·Δy)`, NOT `p` -- get this wrong by hand and the field comes out
scaled by `Δx·Δy` (silently, no error). This helper is exactly there so you
never have to remember that.

Two points landing on the SAME pixel are summed in place (the physical
superposition of coincident dipoles); points at different pixels are
independent nonzero entries in the one returned `Sheet`, summed
automatically by whichever forward DFT consumes it.
"""
function point_current_sheet(grid::NamedTuple, z::Real, points::AbstractVector; weight::Real=1.0)
    J = zeros(ComplexF64, grid.Nx, grid.Ny, 3)
    for (x, y, p) in points
        i, j = _nearest_node_index(grid, x, y)
        J[i, j, :] .+= _point_moment_vec(p) ./ (grid.Δx * grid.Δy)
    end
    return Sheet(J, z; weight=weight)
end

"""
    _forward_transform(J, Δx, Δy, ramp) -> Ĵ

`Ĵ(kx,ky) = ∫J(x,y)e^{-i(kx·x+ky·y)}dxdy ≈ Δx·Δy·ramp .* fft(J)` -- the `F`
half of the round-trip. FFTW's `fft` is the `e^{-i}` convention, matching this
forward transform directly (no `isign` flip needed here; only the SYNTHESIS
step, `_inverse_transform`, needs the `+i` convention, hence `ifft`).
"""
_forward_transform(J::AbstractMatrix{ComplexF64}, Δx::Float64, Δy::Float64, ramp::Matrix{ComplexF64}) =
    (Δx * Δy) .* ramp .* (_cached_fft_plan(size(J, 1), size(J, 2)) * J)

"""
    _inverse_transform(S, Δkx, Δky, Nx, Ny, ramp) -> E

`E(x,y) = (1/(2π)²)∫S(kx,ky)e^{i(kx·x+ky·y)}dkxdky ≈
(Δkx·Δky·Nx·Ny/(2π)²)·ifft(S .* conj(ramp))` -- the `F⁻¹` half, the `e^{+i}`
convention (FFTW's `ifft`), matching the field-synthesis sign the dipole
path's own FINUFFT call uses (`isign=+1`, see `_nufft_nodes_2d`'s docstring).
The prefactor `Δkx·Δky·Nx·Ny/(2π)²` is, by construction, EXACTLY `1/(Δx·Δy)`
(since `Δkx·Δx·Nx == 2π` and `Δky·Δy·Ny == 2π` hold to machine precision by
`uniform_kxky_grid`'s own re-derivation of `Δx`/`Δy` -- see its docstring) --
this is the algebraic identity that makes `_inverse_transform(_forward_transform(J,...),...)`
reduce to `ifft(fft(J)) == J` exactly (`ramp .* conj(ramp) == 1`), independent
of any physics.
"""
_inverse_transform(S::AbstractMatrix{ComplexF64}, Δkx::Float64, Δky::Float64,
                    Nx::Int, Ny::Int, ramp::Matrix{ComplexF64}) =
    (Δkx * Δky * Nx * Ny / (2π)^2) .* (_cached_ifft_plan(Nx, Ny) * (S .* conj.(ramp)))

end
