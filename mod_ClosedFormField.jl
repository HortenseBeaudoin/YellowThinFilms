"""
mod_ClosedFormField.jl -- per-pixel synthesis of the closed-form rewrite
(`PLAN_closed_form_point_source_rewrite.md` §1.2-1.3, §4.3).

Given a `CavityTransferCache1D` (`mod_CavityTransfer.jl`), the Green's
function of a point source at local depth `z` in `source_layer`, observed at
local depth `z'` in `obs_layer` and lateral offset `(Δx,Δy)`, is assembled
with NO stack solve:

 1. `amplitude_at(cache, o, z)` -- per node, `[Ef;Eb] = T^o·[sf∘Ψf(z); sb∘Ψb(z)]`:
    `O(Nk)` exponentials + 4×4 matvecs, once per `(o, z)`, before any pixel loop.
 2. same layer only: `direct_amplitude(cache, z, z')` -- the source's own
    radiation, `sf∘e^{iQf·(z'−zpos)}` if `z' > zpos` (forward branch) or
    `sb∘e^{iQb·(z'−zpos)}` if `z' < zpos` (backward branch), evaluated with the
    combined (always decaying) exponent. This Heaviside branch is the one
    non-separable piece and is kept as an explicit branch (§1.3). Strict
    inequalities, exactly as `FieldSolver.Modal_amp_profile`: at `z' == zpos`
    exactly (a point source's own singular plane) neither branch is added.
 3. `Φf = e^{iQf,obs·z'}`, `Φb = e^{iQb,obs·(z'−d_obs)}` and the lateral
    Jacobi–Anger (J0,J1,J2) reduction (exact for a point source on an
    azimuthally symmetric stack).

`closed_form_G` materializes the full `(Nx,Ny,Nz,3,3)` Green's tensor on a
lateral grid; `mod_ClosedFormAssembly.jl` FFTs it for multi-source assembly.

Two synthesis paths, dispatched on the cache type:
  - `CavityTransferCache1D`: point source, isotropic/uniaxial -- Jacobi–Anger
    (J0,J1,J2) reduction, `_integrate_pixel_1d`.
  - `CavityTransferCache2D`: box (lateral factor `S(kt)`, `box_lateral=(Δx,Δy)`)
    on any medium, or point on biaxial/gyrotropic -- per z-row, one strength
    per `(kρ,φ)` node and the sum `Σ_n s_n e^{i(kx_n x + ky_n y)}` onto the
    pixel grid, either exactly (separable in x,y → one matrix product,
    `lateral=:direct`) or by a type-1 NUFFT (`lateral=:nufft`, `_nufft_nodes_2d`).
The axial box factor `D` is applied in `amplitude_at`/`direct_amplitude`
(`Δz_box`, face-difference form, Phase 6), exterior of the box's own z-slab.
The 1D cache accepts `Δz_box` too (the seed factor alone, `S ≡ 1`, for checks):
a physical box always goes through the 2D path with `box_lateral`. Inside the box's own slab (Phase 7) only the 2D
path is supported: face ramps + local Ez, §5.3.
"""
module ClosedFormField

export amplitude_at, direct_amplitude, closed_form_G, closed_form_pixel, box_lateral_factor
export interior_slab_at, ez_local_density

import ..CavityTransfer: CavityTransferCache1D, CavityTransferCache2D

using LinearAlgebra
using StaticArrays
using Base.Threads
using Bessels
using FINUFFT

@inline _isfinitec(z::Complex) = isfinite(real(z)) && isfinite(imag(z))
@inline _finite_or_zero(z::ComplexF64) = _isfinitec(z) ? z : 0.0 + 0.0im

"""
Clamped `exp(i·kz·z)`: returns 0 outside the `|Re exponent| ≤ 700` window instead of `Inf`/denormals.
"""
function _phase(kz::ComplexF64, z::Float64)
    a = -imag(kz) * z
    b =  real(kz) * z
    if !isfinite(a) || !isfinite(b) || a > 700
        return 0.0 + 0.0im
    elseif a < -700
        return 0.0 + 0.0im
    else
        return exp(complex(a, b))
    end
end

# `zpos` convention of FieldSolver._seed_from_sources (layer 1 is flipped).
@inline _zpos(cache, zsrc_local::Real) =
    cache.source_layer == 1 ? cache.d_src - Float64(zsrc_local) : Float64(zsrc_local)

"""
    _boxphase(Q, ℓ, Δz) = sinc_phys(Q·Δz/2)·e^{iQℓ}

The box-averaged exponential `(1/Δz)∫_{-Δz/2}^{Δz/2} e^{iQ(ℓ+u)} du`, evaluated in
the face-difference form of §5.2, `[e^{iQ(ℓ+Δz/2)} − e^{iQ(ℓ−Δz/2)}] / (iQΔz)`:
the box seen as two sheets at its faces with weights `±1/(iQΔz)`. Both
exponentials go through the clamped `_phase`, so `D = sinc(QΔz/2)` -- which grows
like `e^{|Im Q|Δz/2}` -- is never formed on its own (no overflow, no `Inf·0`).
Small-`|QΔz|` guard: 4th-order Taylor of `sinc` times `e^{iQℓ}`.
`Δz = 0` is exactly `_phase(Q, ℓ)`.
"""
function _boxphase(Q::ComplexF64, ℓ::Float64, Δz::Float64)
    Δz == 0 && return _phase(Q, ℓ)
    y = Q * Δz
    abs(y) < 1e-3 && return _phase(Q, ℓ) * (1 - y^2 / 24 + y^4 / 1920)
    return (_phase(Q, ℓ + Δz / 2) - _phase(Q, ℓ - Δz / 2)) / (im * y)
end

function _check_box_in_layer(zpos::Float64, Δz::Float64, d::Float64)
    Δz >= 0 || error("Δz_box = $Δz must be ≥ 0")
    tol = sqrt(eps()) * max(d, 1.0)
    (zpos - Δz / 2 >= -tol && zpos + Δz / 2 <= d + tol) ||
        error("box [$(zpos - Δz/2), $(zpos + Δz/2)] (cavity frame) does not fit inside its layer [0, $d]")
end

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1-2: modal amplitudes (no solve)
# ─────────────────────────────────────────────────────────────────────────────

# Ψf = e^{iQf(d−zpos)} (= _SProp(d-zpos,Qf)), Ψb = e^{−iQb·zpos} (= _SProp(-zpos,Qb)); for a
# box, D·Ψ via `_boxphase`. The Δz = 0 branch keeps the original `exp` evaluation.
@inline function _seed_phases(Qf, Qb, d::Float64, zpos::Float64, Δz::Float64)
    Δz == 0 && return (SVector(exp(im * Qf[1] * (d - zpos)), exp(im * Qf[2] * (d - zpos))),
                       SVector(exp(-im * Qb[1] * zpos), exp(-im * Qb[2] * zpos)))
    return (SVector(_boxphase(ComplexF64(Qf[1]), d - zpos, Δz), _boxphase(ComplexF64(Qf[2]), d - zpos, Δz)),
            SVector(_boxphase(ComplexF64(Qb[1]), -zpos, Δz), _boxphase(ComplexF64(Qb[2]), -zpos, Δz)))
end

"""
    amplitude_at(cache, obs_layer, zsrc_local; Δz_box=0.0) -> (Af, Ab)

Unweighted modal amplitudes, `Nk×2×3` each (node, mode, canonical moment
x̂/ŷ/ẑ), referenced like `Modal_amp`: `Af` at z=0 of `obs_layer`, `Ab` at
z=d_obs. For `obs_layer == source_layer` this is the REFLECTED part only; add
`direct_amplitude` for the observation plane of interest.

`Δz_box > 0`: box of thickness `Δz_box` centred at `zsrc_local` (§5.1-5.2):
the seed exponentials `Ψ` become `D·Ψ` in face-difference form (`_boxphase`),
exact for every reflected/transmitted term. `Δz_box = 0` is the point source,
bit-identical to the pre-box code path.
"""
function amplitude_at(cache::CavityTransferCache1D, obs_layer::Int, zsrc_local::Real; Δz_box::Real=0.0)
    haskey(cache.T, obs_layer) || error("amplitude_at: obs_layer=$obs_layer not in this cache's obs_layers $(cache.obs_layers)")
    zpos = _zpos(cache, zsrc_local)
    d    = cache.d_src
    Δz   = Float64(Δz_box)
    (-sqrt(eps()) <= zpos <= d + sqrt(eps())) || error("amplitude_at: zsrc_local=$zsrc_local outside [0,$d]")
    Δz > 0 && _check_box_in_layer(zpos, Δz, d)

    Tv = cache.T[obs_layer]
    Tok = cache.Tvalid[obs_layer]
    Nk = length(cache.kx_nodes)
    Af = zeros(ComplexF64, Nk, 2, 3)
    Ab = zeros(ComplexF64, Nk, 2, 3)
    @inbounds for k in 1:Nk
        (cache.valid[k] && Tok[k]) || continue
        Qf, Qb = cache.Qf_src[k], cache.Qb_src[k]
        Ψf, Ψb = _seed_phases(Qf, Qb, d, zpos, Δz)
        for q in 1:3
            u = SVector(cache.sf[k][1, q] * Ψf[1], cache.sf[k][2, q] * Ψf[2],
                        cache.sb[k][1, q] * Ψb[1], cache.sb[k][2, q] * Ψb[2])
            v = Tv[k] * u
            Af[k, 1, q] = v[1]; Af[k, 2, q] = v[2]
            Ab[k, 1, q] = v[3]; Ab[k, 2, q] = v[4]
        end
    end
    return Af, Ab
end

# Same-layer direct-term branch (§1.3): point -> strict sign of ζ (as Modal_amp_profile);
# box -> forward if ζ ≥ Δz/2, backward if ζ ≤ −Δz/2, and an error inside the slab,
# which needs the interior routine (§5.3).
@inline function _direct_branch(ζ::Float64, Δz::Float64)
    Δz == 0 && return (ζ > 0, ζ < 0)
    ζ >= Δz / 2 && return (true, false)
    ζ <= -Δz / 2 && return (false, true)
    error("direct_amplitude: z_obs is inside the box's own z-slab (|ζ|=$(abs(ζ)) < Δz/2=$(Δz/2)) -- " *
          "that needs the interior-slab term (§5.3), not the exterior seed.")
end

"""
    direct_amplitude(cache, zsrc_local, z_obs; Δz_box=0.0) -> (Af_dir, Ab_dir)

Same-layer direct term (§1.3), `Nk×2×3` each, ALREADY PROPAGATED to the
observation plane `z_obs` (not z=0/z=d-referenced like `amplitude_at`):
forward branch (`z_obs > zpos`) `sf∘e^{iQf·(z_obs−zpos)}`, backward branch
(`z_obs < zpos`) `sb∘e^{iQb·(z_obs−zpos)}`. Both exponentials decay, so this is
free of the overflow the split form `sf∘e^{−iQf·zpos} × e^{iQf·z_obs}` hits at
deep-evanescent nodes (`Im Qf·zpos > 709` → Inf·0 = NaN, i.e. a silently
dropped direct contribution -- see §5.2 of the plan for the same range
argument). Uses the clamped `_phase`. `z_obs` is local, z'=0 at the layer's
z=0 edge; strict inequalities as in `Modal_amp_profile`.

Box (`Δz_box > 0`): exterior only, `|ζ| ≥ Δz_box/2`, where the direct term is the
point one times `D` (`_boxphase(Q, ζ, Δz)`, face-difference form).
"""
function direct_amplitude(cache::CavityTransferCache1D, zsrc_local::Real, z_obs::Real; Δz_box::Real=0.0)
    zpos = _zpos(cache, zsrc_local)
    zo   = Float64(z_obs)
    ζ    = zo - zpos
    Δz   = Float64(Δz_box)
    Nk   = length(cache.kx_nodes)
    Af = zeros(ComplexF64, Nk, 2, 3)
    Ab = zeros(ComplexF64, Nk, 2, 3)
    fwd, bwd = _direct_branch(ζ, Δz)
    (fwd || bwd) || return Af, Ab
    @inbounds for k in 1:Nk
        cache.valid[k] || continue
        if fwd
            Qf = cache.Qf_src[k]
            e = SVector(_boxphase(ComplexF64(Qf[1]), ζ, Δz), _boxphase(ComplexF64(Qf[2]), ζ, Δz))
            for q in 1:3, m in 1:2
                Af[k, m, q] = cache.sf[k][m, q] * e[m]
            end
        else
            Qb = cache.Qb_src[k]
            e = SVector(_boxphase(ComplexF64(Qb[1]), ζ, Δz), _boxphase(ComplexF64(Qb[2]), ζ, Δz))
            for q in 1:3, m in 1:2
                Ab[k, m, q] = cache.sb[k][m, q] * e[m]
            end
        end
    end
    return Af, Ab
end

"""
Same-layer amplitudes pre-multiplied by their observation phase at `z_obs`:
`Ãf = Af_refl·Φf(z_obs) + Af_dir`, `Ãb = Ab_refl·Φb(z_obs) + Ab_dir` (the
direct parts already propagated, `direct_amplitude`). The pixel kernel is
linear in the amplitude of each `(k,m)`, so evaluating it with `Ã` and unit
phases equals the sum of the reflected and direct contributions exactly.
"""
function _phased_same_layer(cache::CavityTransferCache1D, Af_r::Array{ComplexF64,3}, Ab_r::Array{ComplexF64,3},
                            zsrc_local::Real, z_obs::Float64; Δz_box::Real=0.0)
    B = cache.basis[cache.source_layer]
    d = cache.d_obs[cache.source_layer]
    Afd, Abd = direct_amplitude(cache, zsrc_local, z_obs; Δz_box=Δz_box)
    Nk = size(Af_r, 1)
    @inbounds for q in 1:3, m in 1:2, k in 1:Nk
        Afd[k, m, q] += Af_r[k, m, q] * _phase(B.KZf[k, m], z_obs)
        Abd[k, m, q] += Ab_r[k, m, q] * _phase(B.KZb[k, m], z_obs - d)
    end
    return Afd, Abd
end

_unit_phases(Nk::Int) = ones(ComplexF64, Nk, 2)

"""
Quadrature-weighted amplitudes + obs-layer basis, as one NamedTuple the pixel
kernel reads. Weight `wk = −wt·ω/(4π²)`.
"""
function _weighted_mcache(cache::CavityTransferCache1D, obs_layer::Int, Af::Array{ComplexF64,3}, Ab::Array{ComplexF64,3})
    Nk = length(cache.kx_nodes)
    W  = [(-cache.wt_nodes[k] * cache.ω) / (4π^2) for k in 1:Nk]
    wA(A, q) = [_finite_or_zero(W[k] * A[k, m, q]) for k in 1:Nk, m in 1:2]
    B = cache.basis[obs_layer]
    return merge(B, (d=cache.d_obs[obs_layer],
                     Axf=wA(Af, 1), Axb=wA(Ab, 1), Ayf=wA(Af, 2), Ayb=wA(Ab, 2), Azf=wA(Af, 3), Azb=wA(Ab, 3)))
end

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: lateral (J0,J1,J2) synthesis
# ─────────────────────────────────────────────────────────────────────────────

"""
`(J0,J1,J2)(kx·ρ)` for every node and every DISTINCT radius `ρs[r]`, `Nk×Nρ`
(column `r` contiguous), shared by every z-row and moment.
"""
function _build_bessel_row_cache_vol(kx_nodes::Vector{Float64}, ρs::Vector{Float64})
    Nk, Nρ = length(kx_nodes), length(ρs)
    J0 = Matrix{Float64}(undef, Nk, Nρ)
    J1 = Matrix{Float64}(undef, Nk, Nρ)
    J2 = Matrix{Float64}(undef, Nk, Nρ)
    @threads for r in 1:Nρ
        ρ = ρs[r]
        @inbounds for k in 1:Nk
            kρ = kx_nodes[k] * ρ
            J0[k, r] = besselj0(kρ)
            J1[k, r] = besselj1(kρ)
            J2[k, r] = besselj(2, kρ)
        end
    end
    return J0, J1, J2
end

"""
Observation-plane phases `(Φf, Φb)`, `Nk×2`: `Φf = e^{iQf,obs·z'}`,
`Φb = e^{iQb,obs·(z'−d_obs)}` (Eb is z=d-referenced; always decays).
Depends on `(obs_layer, z')` only -- built once per z-row.
"""
function _obs_phases(mc, z_obs::Float64)
    Nk = size(mc.KZf, 1)
    Φf = Matrix{ComplexF64}(undef, Nk, 2)
    Φb = Matrix{ComplexF64}(undef, Nk, 2)
    @inbounds for m in 1:2, k in 1:Nk
        Φf[k, m] = _phase(mc.KZf[k, m], z_obs)
        Φb[k, m] = _phase(mc.KZb[k, m], z_obs - mc.d)
    end
    return Φf, Φb
end

"""
Point-source pixel integral along the local x-axis (`ψ ∈ {0,π}`), the
Jacobi–Anger reduction of the azimuthal integral: the `cos(2ψ)=1`, `sin(2ψ)=0`
values are built in, and `_integrate_pixel_rotated_1d` rotates a general offset
into this frame. Observation phases come precomputed (`_obs_phases`); `mc` holds
one obs layer.
"""
function _integrate_pixel_1d(mc, ρ::Float64, x::Float64,
                             Φf::AbstractMatrix{ComplexF64}, Φb::AbstractMatrix{ComplexF64},
                             kx_nodes::Vector{Float64},
                             J0row::AbstractVector{Float64}, J1row::AbstractVector{Float64},
                             J2row::AbstractVector{Float64},
                             px::ComplexF64, py::ComplexF64, pz::ComplexF64)
    val_x = 0.0 + 0.0im
    val_y = 0.0 + 0.0im
    val_z = 0.0 + 0.0im
    cosψ = ρ == 0.0 ? 0.0 : x / ρ

    @inbounds for k in eachindex(kx_nodes)
        kx = kx_nodes[k]
        J0 = J0row[k]; J1 = J1row[k]; J2 = J2row[k]
        (isfinite(J0) && isfinite(J1) && isfinite(J2)) || continue

        for m in 1:2
            exf = mc.Exf[k, m]; exb = mc.Exb[k, m]
            eyf = mc.Eyf[k, m]; eyb = mc.Eyb[k, m]
            ezf = mc.Ezf[k, m]; ezb = mc.Ezb[k, m]

            φf = Φf[k, m]
            φb = Φb[k, m]

            axf = mc.Axf[k, m]; ayf = mc.Ayf[k, m]; azf = mc.Azf[k, m]
            Af = px * axf + py * ayf
            Bf = px * ayf - py * axf
            Cf = pz * azf
            Ix_f = (1im/2) * ((J2 - J0) * Af * exf - (J0 + J2) * Bf * eyf) + J1 * cosψ * Cf * exf
            Iy_f = (1im/2) * ((J2 - J0) * Af * eyf + (J0 + J2) * Bf * exf) + J1 * cosψ * Cf * eyf
            Iz_f = J1 * cosψ * Af * ezf - 1im * J0 * Cf * ezf

            axb = mc.Axb[k, m]; ayb = mc.Ayb[k, m]; azb = mc.Azb[k, m]
            Ab = px * axb + py * ayb
            Bb = px * ayb - py * axb
            Cb = pz * azb
            Ix_b = (1im/2) * ((J2 - J0) * Ab * exb - (J0 + J2) * Bb * eyb) + J1 * cosψ * Cb * exb
            Iy_b = (1im/2) * ((J2 - J0) * Ab * eyb + (J0 + J2) * Bb * exb) + J1 * cosψ * Cb * eyb
            Iz_b = J1 * cosψ * Ab * ezb - 1im * J0 * Cb * ezb

            val_x += (Ix_f * φf + Ix_b * φb) * kx
            val_y += (Iy_f * φf + Iy_b * φb) * kx
            val_z += (Iz_f * φf + Iz_b * φb) * kx
        end
    end
    return val_x, val_y, val_z
end

"""
General in-plane offset `(Δx,Δy)` by rotating the moment into the frame where
the pixel lies on the local x-axis, evaluating `_integrate_pixel_1d`, and
rotating the in-plane field back -- ported from `_integrate_pixel_rotated`.
"""
function _integrate_pixel_rotated_1d(mc, Δx::Float64, Δy::Float64,
                                     Φf::AbstractMatrix{ComplexF64}, Φb::AbstractMatrix{ComplexF64},
                                     kx_nodes::Vector{Float64},
                                     J0row::AbstractVector{Float64}, J1row::AbstractVector{Float64},
                                     J2row::AbstractVector{Float64},
                                     px::ComplexF64, py::ComplexF64, pz::ComplexF64)
    ρ = hypot(Δx, Δy)
    ρ == 0.0 && return _integrate_pixel_1d(mc, 0.0, 0.0, Φf, Φb, kx_nodes, J0row, J1row, J2row, px, py, pz)
    cψ = Δx / ρ
    sψ = Δy / ρ
    p_par  =  px * cψ + py * sψ
    p_perp = -px * sψ + py * cψ
    val_par, val_perp, val_z = _integrate_pixel_1d(mc, ρ, ρ, Φf, Φb, kx_nodes, J0row, J1row, J2row, p_par, p_perp, pz)
    return val_par * cψ - val_perp * sψ, val_par * sψ + val_perp * cψ, val_z
end

# ─────────────────────────────────────────────────────────────────────────────
#  Materialization
# ─────────────────────────────────────────────────────────────────────────────

const _MOMENTS = (ComplexF64[1, 0, 0], ComplexF64[0, 1, 0], ComplexF64[0, 0, 1])

"""
    closed_form_G(cache::CavityTransferCache1D, obs_layer, zsrc_local, z_obs, xs, ys)
        -> Array{ComplexF64,5}    # (Nx, Ny, Nz, field component, moment)

The point-source Green's tensor at lateral offsets `(xs[i], ys[j])` (offsets
from the source) and local observation depths `z_obs` in `obs_layer`, for a
source at local depth `zsrc_local` in `cache.source_layer`. Column `q` of the
trailing 3×3 is the field of a unit canonical moment `q ∈ {x̂,ŷ,ẑ}` --
`(x, y, z, field component, moment)`.

`amplitude_at` runs once; in the same layer, `direct_amplitude` adds the
branch per z-row (cheap, `O(Nk)`); the Bessel row cache is shared by every
z-row and moment.
"""
function closed_form_G(cache::CavityTransferCache1D, obs_layer::Int, zsrc_local::Real,
                       z_obs::AbstractVector{<:Real}, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real};
                       Δz_box::Real=0.0)
    zo = collect(Float64, z_obs)
    xv = collect(Float64, xs); yv = collect(Float64, ys)
    Nx, Ny, Nz = length(xv), length(yv), length(zo)
    kx = cache.kx_nodes

    Af_r, Ab_r = amplitude_at(cache, obs_layer, zsrc_local; Δz_box=Δz_box)
    same = obs_layer == cache.source_layer
    mc_refl = same ? nothing : _weighted_mcache(cache, obs_layer, Af_r, Ab_r)

    # The pixel integral depends on the offset only through ρ and an in-plane rotation:
    # G(ρ,ψ) = Rot(ψ)·R(ρ)·Rot(ψ)ᵀ, with R the 3×3 response in the frame where the pixel
    # lies on the local x-axis (columns: unit moments along par, perp, z). R is computed
    # once per DISTINCT radius (square grids: ~8× fewer than pixels), then rotated per pixel.
    ρpix = [hypot(xv[i], yv[j]) for i in 1:Nx, j in 1:Ny]
    uρ = unique(ρpix)
    ridx = Dict(ρ => r for (r, ρ) in enumerate(uρ))
    rmap = [ridx[ρpix[i, j]] for i in 1:Nx, j in 1:Ny]
    Nρ = length(uρ)
    J0, J1, J2 = _build_bessel_row_cache_vol(kx, uρ)

    G = Array{ComplexF64,5}(undef, Nx, Ny, Nz, 3, 3)
    R = Array{ComplexF64,3}(undef, 3, 3, Nρ)
    for zi in 1:Nz
        mc, Φf, Φb = if same
            Ãf, Ãb = _phased_same_layer(cache, Af_r, Ab_r, zsrc_local, zo[zi]; Δz_box=Δz_box)
            (_weighted_mcache(cache, obs_layer, Ãf, Ãb), _unit_phases(length(kx)), _unit_phases(length(kx)))
        else
            (mc_refl, _obs_phases(mc_refl, zo[zi])...)
        end
        @threads for r in 1:Nρ
            ρ = uρ[r]
            J0r = @view J0[:, r]; J1r = @view J1[:, r]; J2r = @view J2[:, r]
            for (q, p) in enumerate(_MOMENTS)
                v = _integrate_pixel_1d(mc, ρ, ρ, Φf, Φb, kx, J0r, J1r, J2r, p[1], p[2], p[3])
                R[1, q, r] = v[1]; R[2, q, r] = v[2]; R[3, q, r] = v[3]
            end
        end
        @threads for j in 1:Ny
            @inbounds for i in 1:Nx
                ρ = ρpix[i, j]
                cψ, sψ = ρ == 0.0 ? (1.0, 0.0) : (xv[i] / ρ, yv[j] / ρ)
                Rot = SMatrix{3,3,Float64}(cψ, sψ, 0.0, -sψ, cψ, 0.0, 0.0, 0.0, 1.0)
                M = Rot * SMatrix{3,3,ComplexF64}(view(R, :, :, rmap[i, j])) * transpose(Rot)
                for q in 1:3, c in 1:3
                    G[i, j, zi, c, q] = M[c, q]
                end
            end
        end
    end
    return G
end

"""
    closed_form_pixel(cache, obs_layer, zsrc_local, z_obs, Δx, Δy, p) -> SVector{3,ComplexF64}

One-off field query at a single `(Δx, Δy, z_obs)` for moment `p`, no grid.
"""
function closed_form_pixel(cache::CavityTransferCache1D, obs_layer::Int, zsrc_local::Real,
                           z_obs::Real, Δx::Real, Δy::Real, p::AbstractVector)
    Af, Ab = amplitude_at(cache, obs_layer, zsrc_local)
    mc, Φf, Φb = if obs_layer == cache.source_layer
        Ãf, Ãb = _phased_same_layer(cache, Af, Ab, zsrc_local, Float64(z_obs))
        (_weighted_mcache(cache, obs_layer, Ãf, Ãb), _unit_phases(length(cache.kx_nodes)), _unit_phases(length(cache.kx_nodes)))
    else
        mc0 = _weighted_mcache(cache, obs_layer, Af, Ab)
        (mc0, _obs_phases(mc0, Float64(z_obs))...)
    end
    ρ = hypot(Float64(Δx), Float64(Δy))
    kx = cache.kx_nodes
    J0 = besselj0.(kx .* ρ); J1 = besselj1.(kx .* ρ); J2 = besselj.(2, kx .* ρ)
    pc = ComplexF64.(p)
    return SVector(_integrate_pixel_rotated_1d(mc, Float64(Δx), Float64(Δy), Φf, Φb, kx, J0, J1, J2, pc[1], pc[2], pc[3]))
end


# ─────────────────────────────────────────────────────────────────────────────
#  2D (kρ, φ) path
# ─────────────────────────────────────────────────────────────────────────────

sinc_phys(u::Float64) = sinc(u / π)

"""
    box_lateral_factor(kx, ky, Δx_box, Δy_box) = sinc_phys(kx·Δx_box/2)·sinc_phys(ky·Δy_box/2)

The box footprint's own 2D Fourier transform, normalised so `S(0,0) = 1`
(total moment preserved). First zero at `kx = 2π/Δx_box`. §1.1.
"""
box_lateral_factor(kx::Float64, ky::Float64, Δx_box::Real, Δy_box::Real) =
    sinc_phys(kx * Δx_box / 2) * sinc_phys(ky * Δy_box / 2)

_node_kxky(cache::CavityTransferCache2D) =
    (vec([kρ * cos(φ) for kρ in cache.kρ_nodes, φ in cache.phi_nodes]),
     vec([kρ * sin(φ) for kρ in cache.kρ_nodes, φ in cache.phi_nodes]))

"""
    amplitude_at(cache::CavityTransferCache2D, obs_layer, zsrc_local; Δz_box=0.0) -> (Af, Ab)

As the 1D method, per `(kρ,φ)` node: `Nk×Np×2×3` each (node k, node p, mode,
canonical moment), reflected/transmitted part only. `Δz_box` as in the 1D method:
the per-mode `D` is sampled at each node's own `Qf,src[n]`/`Qb,src[n]`, so it is
exact for biaxial/gyrotropic layers too (`Qf₁ ≠ Qf₂ ≠ −Qb`).
"""
function amplitude_at(cache::CavityTransferCache2D, obs_layer::Int, zsrc_local::Real; Δz_box::Real=0.0)
    haskey(cache.T, obs_layer) || error("amplitude_at: obs_layer=$obs_layer not in this cache's obs_layers $(cache.obs_layers)")
    zpos = _zpos(cache, zsrc_local)
    d    = cache.d_src
    Δz   = Float64(Δz_box)
    (-sqrt(eps()) <= zpos <= d + sqrt(eps())) || error("amplitude_at: zsrc_local=$zsrc_local outside [0,$d]")
    Δz > 0 && _check_box_in_layer(zpos, Δz, d)
    Nk, Np = size(cache.valid)
    Af = zeros(ComplexF64, Nk, Np, 2, 3)
    Ab = zeros(ComplexF64, Nk, Np, 2, 3)
    Tv, Tok = cache.T[obs_layer], cache.Tvalid[obs_layer]
    @threads for p in 1:Np
    @inbounds for k in 1:Nk
        (cache.valid[k, p] && Tok[k, p]) || continue
        Qf, Qb = cache.Qf_src[k, p], cache.Qb_src[k, p]
        Ψf, Ψb = _seed_phases(Qf, Qb, d, zpos, Δz)
        sfk, sbk, Tk = cache.sf[k, p], cache.sb[k, p], Tv[k, p]
        for q in 1:3
            v = Tk * SVector(sfk[1, q] * Ψf[1], sfk[2, q] * Ψf[2], sbk[1, q] * Ψb[1], sbk[2, q] * Ψb[2])
            Af[k, p, 1, q] = v[1]; Af[k, p, 2, q] = v[2]
            Ab[k, p, 1, q] = v[3]; Ab[k, p, 2, q] = v[4]
        end
    end
    end
    return Af, Ab
end

"""
    direct_amplitude(cache::CavityTransferCache2D, zsrc_local, z_obs) -> (Af_dir, Ab_dir)

As the 1D method: the same-layer direct term already propagated to `z_obs`,
with the combined, always-decaying exponent. `Nk×Np×2×3` each. For a box with
`z_obs` inside its own slab (`|ζ| < Δz_box/2`) this returns the interior face
ramps, `interior_slab_at` (§5.3); the local `Ez` term is added separately, in
`closed_form_G` (`ez_local_density`).
"""
function direct_amplitude(cache::CavityTransferCache2D, zsrc_local::Real, z_obs::Real; Δz_box::Real=0.0)
    zpos = _zpos(cache, zsrc_local)
    ζ = Float64(z_obs) - zpos
    Δz = Float64(Δz_box)
    Δz > 0 && abs(ζ) < Δz / 2 && return interior_slab_at(cache, zsrc_local, z_obs, Δz)
    Nk, Np = size(cache.valid)
    Af = zeros(ComplexF64, Nk, Np, 2, 3)
    Ab = zeros(ComplexF64, Nk, Np, 2, 3)
    fwd, bwd = _direct_branch(ζ, Δz)
    (fwd || bwd) || return Af, Ab
    A = fwd ? Af : Ab
    @threads for p in 1:Np
    @inbounds for k in 1:Nk
        cache.valid[k, p] || continue
        Q = fwd ? cache.Qf_src[k, p] : cache.Qb_src[k, p]
        s = fwd ? cache.sf[k, p] : cache.sb[k, p]
        for m in 1:2
            e = _boxphase(ComplexF64(Q[m]), ζ, Δz)
            for q in 1:3
                A[k, p, m, q] = s[m, q] * e
            end
        end
    end
    end
    return Af, Ab
end

"""
Per-node strengths for one z-row, written into `S` (`Nn×3×3`):
`S[n, c, q] = Σ_m (E_c,f · w_n Ãf + E_c,b · w_n Ãb)[n, m, q]`, `n` the flattened
`(k,p)` node (k fastest), `c ∈ x,y,z`, `q` the canonical moment, where
`Ãf = Af_r·e^{iKZf·z'} (+ Afd)` and `Ãb = Ab_r·e^{iKZb·(z'−d)} (+ Abd)` are the
reflected amplitudes moved to the observation plane plus, in the source layer,
the direct ones (`Afd/Abd`, `nothing` otherwise), and `w_n` the node weight
(quadrature × box lateral factor, × the NUFFT ramp on that path). Non-finite
weighted amplitudes are zeroed. `ezl` (or `nothing`) adds the local `Ez`
`w_n·ezl[q]` on valid nodes. One fused pass, threaded over φ columns; each
observation phase is evaluated once per (node, mode).
"""
function _strengths_row!(S::Array{ComplexF64,3}, cache::CavityTransferCache2D, B, Af_r, Ab_r, Afd, Abd,
                         W::Vector{ComplexF64}, z::Float64, d::Float64, ezl)
    Nk, Np = size(cache.valid)
    @threads for p in 1:Np
    @inbounds for k in 1:Nk
        n = k + (p - 1) * Nk
        w = W[n]
        s11 = s21 = s31 = s12 = s22 = s32 = s13 = s23 = s33 = zero(ComplexF64)
        for m in 1:2
            φf = _phase(B.KZf[k, p, m], z)
            φb = _phase(B.KZb[k, p, m], z - d)
            exf, eyf, ezf = B.Exf[k, p, m], B.Eyf[k, p, m], B.Ezf[k, p, m]
            exb, eyb, ezb = B.Exb[k, p, m], B.Eyb[k, p, m], B.Ezb[k, p, m]
            for q in 1:3
                ãf = Af_r[k, p, m, q] * φf
                ãb = Ab_r[k, p, m, q] * φb
                if Afd !== nothing
                    ãf += Afd[k, p, m, q]
                    ãb += Abd[k, p, m, q]
                end
                af = _finite_or_zero(w * ãf)
                ab = _finite_or_zero(w * ãb)
                vx = exf * af + exb * ab
                vy = eyf * af + eyb * ab
                vz = ezf * af + ezb * ab
                if q == 1
                    s11 += vx; s21 += vy; s31 += vz
                elseif q == 2
                    s12 += vx; s22 += vy; s32 += vz
                else
                    s13 += vx; s23 += vy; s33 += vz
                end
            end
        end
        if ezl !== nothing && cache.valid[k, p]
            s31 += w * ezl[1]; s32 += w * ezl[2]; s33 += w * ezl[3]
        end
        S[n, 1, 1] = s11; S[n, 2, 1] = s21; S[n, 3, 1] = s31
        S[n, 1, 2] = s12; S[n, 2, 2] = s22; S[n, 3, 2] = s32
        S[n, 1, 3] = s13; S[n, 2, 3] = s23; S[n, 3, 3] = s33
    end
    end
    return S
end

"""
Type-1 NUFFT nodes/ramp mapping the flattened `(kρ,φ)` nodes onto a uniform
pixel grid. `kx·px`, `ky·py` are folded into FINUFFT's `[-π,π)` domain with
`rem2pi` (exact: the integer-mode sum only sees them mod 2π, so deep-evanescent
nodes are fine), and `ramp` aligns FINUFFT's `k=0`-centred output with the pixel
centring (`a_x = (1+fld(Nx,2)) − midx`) plus any constant offset `(x0,y0)`.
"""
function _nufft_nodes_2d(kρ_nodes::Vector{Float64}, phi_nodes::Vector{Float64},
                         px::Float64, py::Float64, Nx::Int, Ny::Int,
                         midx::Float64, midy::Float64; x0::Float64=0.0, y0::Float64=0.0)
    Nk, Np = length(kρ_nodes), length(phi_nodes)
    n = Nk * Np
    kx_flat = Vector{Float64}(undef, n)
    ky_flat = Vector{Float64}(undef, n)
    ramp    = Vector{ComplexF64}(undef, n)
    a_x = (1 + fld(Nx, 2)) - midx
    a_y = (1 + fld(Ny, 2)) - midy
    idx = 1
    for p in 1:Np, k in 1:Nk
        kρ = kρ_nodes[k]; φ = phi_nodes[p]
        kx = kρ * cos(φ); ky = kρ * sin(φ)
        kx_flat[idx] = rem2pi(kx * px, RoundNearest)
        ky_flat[idx] = rem2pi(ky * py, RoundNearest)
        ramp[idx]    = cis(kx * (a_x * px - x0) + ky * (a_y * py - y0))
        idx += 1
    end
    return kx_flat, ky_flat, ramp
end

"""
    closed_form_G(cache::CavityTransferCache2D, obs_layer, zsrc_local, z_obs, xs, ys;
                  box_lateral=nothing, lateral=:direct, nufft_eps=1e-9)
        -> Array{ComplexF64,5}   # (Nx, Ny, Nz, field component, moment)

2D-path Green's tensor, same layout as the 1D method. A box source is
`box_lateral=(Δx_box, Δy_box)` (footprint factor `S(kt)` on every node, §1.1)
plus `Δz_box` (axial factor `D` on the seed, §5.1-5.2); for observation rows
inside the box's own slab (same layer, `|ζ| < Δz_box/2`) the direct term is the
interior face ramps (`interior_slab_at`) plus the local `Ez` (`ez_local_density`,
`ez_local=false` drops it, for validation only), §5.3.
`box_lateral=nothing, Δz_box=0` is a point source.

`lateral=:direct` evaluates `Σ_n s_n e^{i(kx_n x_i + ky_n y_j)}` exactly, as
`Pxᵀ·diag(s)·Py` (separable phases, one matrix product per (component, moment));
`lateral=:nufft` uses the ported type-1 NUFFT (uniform `xs`/`ys` required),
accuracy `nufft_eps` (default 1e-9: ~5e-10 relative, well below the quadrature
`rtol`; 1e-12 costs ~25% more).
"""
function closed_form_G(cache::CavityTransferCache2D, obs_layer::Int, zsrc_local::Real,
                       z_obs::AbstractVector{<:Real}, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real};
                       box_lateral::Union{Nothing,Tuple{<:Real,<:Real}}=nothing, Δz_box::Real=0.0,
                       ez_local::Bool=true, lateral::Symbol=:direct, nufft_eps::Float64=1e-9)
    lateral in (:direct, :nufft) || error("closed_form_G: lateral must be :direct or :nufft")
    zo = collect(Float64, z_obs)
    xv = collect(Float64, xs); yv = collect(Float64, ys)
    Nx, Ny, Nz = length(xv), length(yv), length(zo)
    Nk, Np = size(cache.valid)
    kxn, kyn = _node_kxky(cache)

    W = vec([(1im * kρ * wρ * wφ * cache.ω) / (8π^3)
             for (kρ, wρ) in zip(cache.kρ_nodes, cache.kρ_weights), wφ in cache.phi_weights])
    if box_lateral !== nothing
        W .*= box_lateral_factor.(kxn, kyn, Float64(box_lateral[1]), Float64(box_lateral[2]))
    end

    Af_r, Ab_r = amplitude_at(cache, obs_layer, zsrc_local; Δz_box=Δz_box)
    same = obs_layer == cache.source_layer
    B = cache.basis[obs_layer]
    d = cache.d_obs[obs_layer]
    zpos = _zpos(cache, zsrc_local)
    ezl = ez_local_density(cache, Δz_box)

    G = Array{ComplexF64,5}(undef, Nx, Ny, Nz, 3, 3)
    S = Array{ComplexF64,3}(undef, Nk * Np, 3, 3)
    row_strengths!(z) = begin
        Afd, Abd = same ? direct_amplitude(cache, zsrc_local, z; Δz_box=Δz_box) : (nothing, nothing)
        inside = same && ez_local && Δz_box > 0 && abs(z - zpos) < Δz_box / 2
        _strengths_row!(S, cache, B, Af_r, Ab_r, Afd, Abd, W, z, d, inside ? ezl : nothing)
    end

    if lateral === :direct
        Px = [cis(kxn[n] * xv[i]) for n in eachindex(kxn), i in 1:Nx]
        Py = [cis(kyn[n] * yv[j]) for n in eachindex(kyn), j in 1:Ny]
        PxT = transpose(Px)
        for zi in 1:Nz
            row_strengths!(zo[zi])
            @threads for cq in 1:9
                c, q = (cq - 1) % 3 + 1, (cq - 1) ÷ 3 + 1
                G[:, :, zi, c, q] .= PxT * (view(S, :, c, q) .* Py)
            end
        end
    else
        # One FINUFFT plan for every row: the nodes are z-independent, so they are
        # sorted/spread-prepared once (setpts) and each row is one 9-vector execute.
        # The z-independent ramp is folded into the node weights.
        cx, cy = 1 + fld(Nx, 2), 1 + fld(Ny, 2)
        pxp = Nx > 1 ? xv[2] - xv[1] : 1.0
        pyp = Ny > 1 ? yv[2] - yv[1] : 1.0
        kx_flat, ky_flat, ramp = _nufft_nodes_2d(cache.kρ_nodes, cache.phi_nodes, pxp, pyp, Nx, Ny,
                                                 Float64(cx), Float64(cy); x0=-xv[cx], y0=-yv[cy])
        W .*= ramp
        F = Array{ComplexF64,3}(undef, Nx, Ny, 9)
        plan = finufft_makeplan(1, [Nx, Ny], 1, 9, nufft_eps)
        try
            finufft_setpts!(plan, kx_flat, ky_flat)
            Sr = reshape(S, :, 9)
            for zi in 1:Nz
                row_strengths!(zo[zi])
                finufft_exec!(plan, Sr, F)
                for cq in 1:9
                    c, q = (cq - 1) % 3 + 1, (cq - 1) ÷ 3 + 1
                    G[:, :, zi, c, q] .= view(F, :, :, cq)
                end
            end
        finally
            finufft_destroy!(plan)
        end
    end
    return G
end


# ─────────────────────────────────────────────────────────────────────────────
#  Inside the source slab (§5.3): face ramps + local Ez
# ─────────────────────────────────────────────────────────────────────────────

# (e^{x} − 1)/x·(1/Δz)·a-type ramps, regular at Q → 0: returns expm1(iQ·a)/(iQ·Δz).
@inline function _ramp(Q::ComplexF64, a::Float64, Δz::Float64)
    y = im * Q * a
    abs(Q) * a < 1e-12 && return complex(a / Δz)
    real(y) < -700 && return -1 / (im * Q * Δz)     # e^{iQa} underflows: (0 − 1)/(iQΔz)
    return expm1(y) / (im * Q * Δz)
end

"""
    interior_slab_at(cache::CavityTransferCache2D, zsrc_local, z_obs, Δz_box) -> (Af_dir, Ab_dir)

Same-layer DIRECT term for an observation plane inside the box's own z-slab,
`z_lo < z_obs < z_hi` (`z_lo,hi = zpos ∓ Δz/2`, cavity frame), §5.3. The sheets
below `z_obs` radiate forward, those above radiate backward; averaging each set
over its part of the slab gives the face ramps, already propagated to `z_obs`:

    Af = sf ∘ (e^{iQf·a} − 1)/(iQf·Δz),   a = z_obs − z_lo
    Ab = sb ∘ (1 − e^{−iQb·b})/(iQb·Δz),  b = z_hi − z_obs

(`expm1` form, regular as `Q → 0`, both exponentials bounded). Each ramp is a
face wave plus a constant; with the cached `sb = −sib_raw` (i.e. `σ_b = −1`) the
constants add up to the particular solution `Ψ_p = −(iM)⁻¹Js/Δz` of
`dΨ/dz = iMΨ + Js/Δz`. Continuous with the exterior formula at both faces
(`(e^{iy}−1)/(iy) = e^{iy/2}·sinc(y/2)`). The reflected part inside the slab is
the ordinary `amplitude_at(...; Δz_box)` -- all its sheets are on one side.
"""
function interior_slab_at(cache::CavityTransferCache2D, zsrc_local::Real, z_obs::Real, Δz_box::Real)
    zpos = _zpos(cache, zsrc_local)
    Δz = Float64(Δz_box)
    zo = Float64(z_obs)
    a, b = zo - (zpos - Δz / 2), (zpos + Δz / 2) - zo
    # a plane on a face can round to |ζ| < Δz/2 (the caller's branch test) with a or b = 0 (or −eps):
    # accept it, the ramp formula is continuous there (one ramp vanishes, the other is the exterior form)
    tol = 8eps(max(abs(zo), abs(zpos), Δz))
    (a >= -tol && b >= -tol) || error("interior_slab_at: z_obs=$zo is outside the slab around zpos=$zpos (Δz=$Δz)")
    a, b = clamp(a, 0.0, Δz), clamp(b, 0.0, Δz)
    Nk, Np = size(cache.valid)
    Af = zeros(ComplexF64, Nk, Np, 2, 3)
    Ab = zeros(ComplexF64, Nk, Np, 2, 3)
    @threads for p in 1:Np
    @inbounds for k in 1:Nk
        cache.valid[k, p] || continue
        Qf, Qb = cache.Qf_src[k, p], cache.Qb_src[k, p]
        for m in 1:2
            rf = _ramp(ComplexF64(Qf[m]), a, Δz)
            rb = _ramp(-ComplexF64(Qb[m]), b, Δz)      # (e^{−iQb·b} − 1)/(−iQb·Δz) = (1 − e^{−iQb·b})/(iQb·Δz)
            for q in 1:3
                Af[k, p, m, q] = cache.sf[k, p][m, q] * rf
                Ab[k, p, m, q] = cache.sb[k, p][m, q] * rb
            end
        end
    end
    end
    return Af, Ab
end

"""
    ez_local_density(cache, Δz_box) -> SVector{3,ComplexF64}

The local (non-modal) `Ez` per unit canonical moment `x̂,ŷ,ẑ` inside a box's own
slab, §5.3. In the Berreman formalism `Ez` is algebraic; the ẑ row of ∇×H gives
`Ez = (mode part) + Ez_local`, and the mode field basis (`_mode_field_core`)
carries only the mode part. `Ez_local` follows from the same relation that
produced `J_Et = −i·kt·Jz/(ϵzz·ω)` in `Source_projection_Berreman_xyz`: the
Berreman Et rows couple to Ez as `+i·kt·Ez` (the `B12` block of
`_berreman_core`), so a source `Ez_local` produces exactly that Et jump iff
`Ez_local = −Jz/(ϵzz·ω)` per unit area. For the box density `1/Δz` and the
unit dipole current `Jz = −iω·p_z`: `Ez_local = i·p_z/(ϵzz·Δz)` -- only the
`p_z` column, no `Et`/`H` local terms. Uniform in `kt` (flat); only its sum
with the modal particular part decays (like `1/kρ²`). Added per node, times
the same quadrature weight and `S(kt)` as everything else, never in real space.
"""
ez_local_density(cache::CavityTransferCache2D, Δz_box::Real) =
    SVector{3,ComplexF64}(0, 0, im / (cache.ϵ_src[3, 3] * Δz_box))

end
