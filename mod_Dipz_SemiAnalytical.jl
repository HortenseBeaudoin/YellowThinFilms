module ZSemiAnalytical

export build_semi_external, build_semi_internal, build_semi_external_x
export eval_semi, eval_semi_external, eval_semi_internal, eval_semi_external_x
export eval_semi_cartesian
export build_kx_grid

import ..KxGrid: branch_points, evanescent_umax, _evanescent_band_nodes
using LinearAlgebra
using Bessels
using FastGaussQuadrature

@inline _isfinitec(z::Real)    = isfinite(z)
@inline _isfinitec(z::Complex) = isfinite(real(z)) && isfinite(imag(z))


# ─────────────────────────────────────────────────────────────────────────────
#  Fixed, non-uniform kx grid (2026-09-18: moved here from `mod_KxGrid.jl`).
#  `adaptive_kx_grid` (`mod_KxGrid.jl`) superseded this everywhere else in the
#  live pipeline -- see that module's own docstring note on this move. This
#  module keeps it as its own ground-truth quadrature: `_semi_common` below
#  is the only real caller left, plus `Gz_ShapedPixelKernel_Validation.jl`.
# ─────────────────────────────────────────────────────────────────────────────

# On a segment between two singularities, cluster point at the extremities
function _endpoint_cluster(a::Float64, b::Float64, N::Int; δ::Float64=1e-6)
    ξ, w = gausslegendre(N)

    θmax = π/2 - δ
    θ    = @. θmax * ξ
    wθ   = @. w * θmax

    c = (a + b) / 2
    h = (b - a) / 2

    kx = @. c + h * sin(θ)
    wt = @. wθ * h * cos(θ)

    return kx, wt
end


# Fixed-grid, non uniform.
function build_kx_grid(kbreaks::Vector{Float64},
                       pz_min::Float64,
                       z_dip_min::Float64;
                       N_prop::Int=60,
                       N_band::Int=125,
                       N_eva::Int=250,
                       target_decay::Float64=25.0,
                       kx_ratio_max::Float64=80.0,
                       δ::Float64=1e-6)

    isempty(kbreaks) && error("kbreaks cannot be empty")

    kx_all = Float64[]
    wt_all = Float64[]

    # first band: 0 -> k1 with sin substitution
    k1 = kbreaks[1]
    θmax = π/2 - δ
    ξ, w = gausslegendre(N_prop)
    θ = @. (θmax/2) * (ξ + 1)
    wθ = @. w * (θmax/2)

    append!(kx_all, @. k1 * sin(θ))
    append!(wt_all, @. wθ * (k1 * cos(θ)))

    # intermediate finite bands, clustered at BOTH endpoints
    for m in 1:length(kbreaks)-1
        a = kbreaks[m]
        b = kbreaks[m+1]

        kx_band, wt_band = _endpoint_cluster(a, b, N_band; δ=δ)

        append!(kx_all, kx_band)
        append!(wt_all, wt_band)
    end

    # evanescent band: kmax -> ∞
    kmax = kbreaks[end]
    Leff = max(z_dip_min, 0.5 * pz_min)
    umax = evanescent_umax(kmax, Leff; target_decay=target_decay, kx_ratio_max=kx_ratio_max)
    kx_eva, wt_eva = _evanescent_band_nodes(kmax, umax, N_eva; δ=δ)
    append!(kx_all, kx_eva)
    append!(wt_all, wt_eva)

    return kx_all, wt_all
end


function _kz_branch(k::ComplexF64, kx::Float64)
            kz = sqrt(k^2 - kx^2)
            return imag(kz) < 0 ? -kz : kz
        end


function _finite_or_zero(z::ComplexF64)
    return _isfinitec(z) ? z : 0.0 + 0.0im
end


"""
    _A_coeffs(kx, ϵ1, ϵ2, ϵ3, k1z, k2z, k3z, d2) -> (A1, A2, A3, A4, exp2)

Shared TM reflection/transmission coefficients (z-oriented-dipole basis) for
the 3-layer external-dipole models -- used identically by `eval_semi_external`
and `eval_semi_external_x` (the latter additionally builds B1..B4/C1..C4 on
top, reusing the returned `exp2 = exp(2im*k2z*d2)` rather than recomputing it).
"""
function _A_coeffs(kx, ϵ1, ϵ2, ϵ3, k1z, k2z, k3z, d2)
    exp2 = exp(2im * k2z * d2)
    f1 = ϵ2*k1z - ϵ1*k2z;  f2 = ϵ2*k1z + ϵ1*k2z
    f3 = ϵ3*k2z - ϵ2*k3z;  f4 = ϵ3*k2z + ϵ2*k3z
    den = f2*f4 + f1*f3*exp2
    A1 = 1im*kx*(f1*f4 + f2*f3*exp2) / (k1z * den)
    A2 = 1im*(2*ϵ1*kx*f4) / den
    A3 = 1im*(2*ϵ1*kx*f3*exp2) / den
    A4 = 1im*(4*ϵ1*ϵ2*kx*k2z*exp(1im*(k2z - k3z)*d2)) / den
    return A1, A2, A3, A4, exp2
end


function _slab_internal_coeffs(kx::Float64, ω::Float64,
                                                   ϵ1, ϵ2, ϵ3,
                                                   d2::Float64, zs::Float64)

    ε1 = ComplexF64(ϵ1)
    ε2 = ComplexF64(ϵ2)
    ε3 = ComplexF64(ϵ3)

    k1  = ComplexF64(ω) * sqrt(ε1)
    k2  = ComplexF64(ω) * sqrt(ε2)
    k3  = ComplexF64(ω) * sqrt(ε3)

    k1z = _kz_branch(k1, kx)
    k2z = _kz_branch(k2, kx)
    k3z = _kz_branch(k3, kx)

    # TM Fresnel coefficients for H_y (equivalently for Ψ = ε E_z)
    r21 = (ε1 * k2z - ε2 * k1z) / (ε1 * k2z + ε2 * k1z)
    r23 = (ε3 * k2z - ε2 * k3z) / (ε3 * k2z + ε2 * k3z)

    t21 = 1.0 + 0.0im + r21
    t23 = 1.0 + 0.0im + r23

    den = 1.0 + 0.0im - r21 * r23 * exp(2im * k2z * d2)

    # cavity amplitudes at the source plane (book-style 2.4.6 structure)
    U = (1.0 + 0.0im + r23 * exp(2im * k2z * (d2 - zs))) / den
    W = (1.0 + 0.0im + r21 * exp(2im * k2z * zs))        / den

    # transmitted scalar amplitudes Ψ = ε E_z
    Tup = t21 * U * exp(1im * k2z * zs)
    Tdn = t23 * W * exp(1im * k2z * (d2 - zs))

    return (k1z=k1z, k2z=k2z, k3z=k3z,
            r21=r21, r23=r23, t21=t21, t23=t23,
            U=U, W=W, Tup=Tup, Tdn=Tdn, den=den)
end


function _internal_total_TM(kx::Float64,
                                                ρ::Float64,
                                                zloc::Float64,
                                                ω::Float64,
                                                pzamp::ComplexF64,
                                                ϵ1, ϵ2, ϵ3,
                                                d2::Float64,
                                                zs::Float64)

    c = _slab_internal_coeffs(kx, ω, ϵ1, ϵ2, ϵ3, d2, zs)

    ε1 = ComplexF64(ϵ1)
    ε2 = ComplexF64(ϵ2)
    ε3 = ComplexF64(ϵ3)

    J1 = besselj1(kx * ρ)
    J0 = besselj0(kx * ρ)

    gρ = 0.0 + 0.0im
    gz = 0.0 + 0.0im

    if zloc < 0.0
        # ---------------- region 1: above layer 2 ----------------
        h1 = -zloc

        # total transmitted scalar field in region 1
        Ψ = c.Tup * exp(1im * c.k1z * h1)

        # Ez = Ψ / ε_obs
        gz = (-1im * pzamp) / (4π * ε1) * (kx^3 / c.k2z) * J0 * Ψ

        # upward wave in region 1: kz = -k1z  => Eρ = +(k1z/kx) Ez
        gρ = (pzamp) / (4π * ε1) * (kx^2 * c.k1z / c.k2z) * J1 * Ψ

    elseif zloc <= zs
        # ---------------- region 2a: inside layer 2, above source ----------------
        Ψup = c.U * exp(1im * c.k2z * (zs - zloc))
        Ψdn = c.r21 * c.U * exp(1im * c.k2z * (zs + zloc))

        # total Ψ = Ψup + Ψdn
        gz = (-1im * pzamp) / (4π * ε2) * (kx^3 / c.k2z) * J0 * (Ψup + Ψdn)

        # upward term has kz=-k2z, downward term has kz=+k2z
        # so Eρ ∝ Ψup - Ψdn
        gρ = (pzamp) / (4π * ε2) * (kx^2) * J1 * (Ψup - Ψdn)

    elseif zloc <= d2
        # ---------------- region 2b: inside layer 2, below source ----------------
        Ψdn = c.W * exp(1im * c.k2z * (zloc - zs))
        Ψup = c.r23 * c.W * exp(1im * c.k2z * (2*d2 - zloc - zs))

        gz = (-1im * pzamp) / (4π * ε2) * (kx^3 / c.k2z) * J0 * (Ψdn + Ψup)

        # downward term kz=+k2z, upward term kz=-k2z
        # so Eρ ∝ Ψup - Ψdn
        gρ = (pzamp) / (4π * ε2) * (kx^2) * J1 * (Ψup - Ψdn)

    else
        # ---------------- region 3: below layer 2 ----------------
        h3 = zloc - d2

        Ψ = c.Tdn * exp(1im * c.k3z * h3)

        gz = (-1im * pzamp) / (4π * ε3) * (kx^3 / c.k2z) * J0 * Ψ

        # downward wave in region 3: kz=+k3z => Eρ = -(k3z/kx) Ez
        gρ = -(pzamp) / (4π * ε3) * (kx^2 * c.k3z / c.k2z) * J1 * Ψ
    end

    return _finite_or_zero(gρ), _finite_or_zero(gz)
end


# ─────────────────────────────────────────────────────────────────────────────
#  Semi-analytical model: internal dipole (source inside layer 2)
# ─────────────────────────────────────────────────────────────────────────────

struct SemiAnalytical_Internal
    kx_nodes::Vector{Float64}
    wt_nodes::Vector{Float64}
    ω::Float64
    μz::ComplexF64
    ϵ1::ComplexF64; ϵ2::ComplexF64; ϵ3::ComplexF64
    d2::Float64
    zs::Float64
    d1::Float64
end

"""
    _semi_common(stack, source, pz, zref; N_prop, N_band, N_eva, kx_ratio_max)

Shared setup for the three `build_semi_*` constructors: pulls the 3-layer
ϵ/d/ω/source-z data every model needs and builds the common kx quadrature grid
around `zref` (the effective source-to-interface distance that sets the grid's
evanescent tail, model-specific -- see each `build_semi_*` call site).

# Defaults (N_prop/N_band/N_eva)
Raised from 250/125/250 (2026-08-22): confirmed via localize_vgv.jl that the
old defaults were simply under-resolved for a near-interface source, not
`kx_ratio_max`-tail-limited -- capping kx_ratio_max alone (80->20) left the
old-resolution answer unchanged, while raising N_prop/N_band/N_eva alone
reproduced the fully "converged+capped" answer exactly. Since this model's
whole purpose is to be the converged ground truth other paths are checked
against, its defaults should already BE that converged setting, not
something a caller has to opt into.
"""
function _semi_common(stack, source, pz, zref;
                      N_prop::Int, N_band::Int, N_eva::Int, kx_ratio_max::Float64)
    ϵ1 = ComplexF64(stack[1].ϵ[1,1]); ϵ2 = ComplexF64(stack[2].ϵ[1,1]); ϵ3 = ComplexF64(stack[3].ϵ[1,1])
    d1 = stack[1].d; d2 = stack[2].d
    ω  = source.ω

    kbreaks = branch_points(stack, ω)
    kx_nodes, wt_nodes = build_kx_grid(
        kbreaks, pz, zref;
        N_prop=N_prop, N_band=N_band, N_eva=N_eva, kx_ratio_max=kx_ratio_max)

    return (ϵ1=ϵ1, ϵ2=ϵ2, ϵ3=ϵ3, d1=d1, d2=d2, ω=ω, kx_nodes=kx_nodes, wt_nodes=wt_nodes)
end


function build_semi_internal(stack, source, pz;
                             N_prop::Int=1000,
                             N_band::Int=max(80, N_prop ÷ 2),
                             N_eva::Int=2000,
                             semi_pz_sign::Real=-1.0,
                             kx_ratio_max::Float64=80.0)
    zs = source.z
    c = _semi_common(stack, source, pz, max(0.5*pz, min(zs, stack[2].d - zs));
                      N_prop=N_prop, N_band=N_band, N_eva=N_eva, kx_ratio_max=kx_ratio_max)

    return SemiAnalytical_Internal(
        c.kx_nodes, c.wt_nodes, c.ω,
        ComplexF64(semi_pz_sign),
        c.ϵ1, c.ϵ2, c.ϵ3, c.d2, zs, c.d1)
end

function eval_semi_internal(m::SemiAnalytical_Internal, x::Float64, posz_c::Float64)
    # SIGNED, not `abs(x)`: `_internal_total_TM`'s `gρ`/`gz` are built purely from
    # besselj1(kx*ρ)/besselj0(kx*ρ) times ρ-independent factors, and J1 is odd while
    # J0 is even -- feeding the signed lateral offset through directly makes `gρ`
    # come out as the genuinely antisymmetric-in-x Cartesian Ex component (matching
    # `integrate_pixel`'s own cosψ=x/ρ convention for a z-oriented dipole, whose Eρ
    # necessarily points opposite ways on either side of the source), with `gz`
    # unaffected (even in ρ already). Using `abs(x)` here previously discarded that
    # sign, silently comparing a same-sign-both-sides quantity against
    # `CalculateField_dipole`'s genuinely antisymmetric Ex output whenever x ≠ the
    # source's own x -- confirmed via a direct sign-flip test (see the memory file
    # documenting this fix) rather than assumed.
    ρ    = x
    zloc = posz_c - m.d1

    valρ = zero(ComplexF64)
    valz = zero(ComplexF64)

    @inbounds for k in eachindex(m.kx_nodes)
        kx = m.kx_nodes[k]
        wk = m.wt_nodes[k]
        gρ, gz = _internal_total_TM(
            kx, ρ, zloc, m.ω, m.μz,
            m.ϵ1, m.ϵ2, m.ϵ3, m.d2, m.zs)
        valρ += wk * gρ
        valz += wk * gz
    end
    return valρ, valz
end


# ─────────────────────────────────────────────────────────────────────────────
#  Semi-analytical model: external dipole (source in layer 1)
# ─────────────────────────────────────────────────────────────────────────────

struct SemiAnalytical_External
    kx_nodes::Vector{Float64}
    wt_nodes::Vector{Float64}
    ω::Float64
    μz::ComplexF64
    ϵ1::ComplexF64; ϵ2::ComplexF64; ϵ3::ComplexF64
    d1::Float64
    d2::Float64
    z0s::Float64
    k1::ComplexF64
    k2::ComplexF64
    k3::ComplexF64
end

function build_semi_external(stack, source, pz;
                             # See _semi_common's docstring on this same raise.
                             N_prop::Int=1000,
                             N_band::Int=max(80, N_prop ÷ 2),
                             N_eva::Int=2000,
                             semi_pz_sign::Real=-1.0,
                             kx_ratio_max::Float64=80.0)
    z0s = source.z
    c = _semi_common(stack, source, pz, max(0.5*pz, Float64(real(z0s)));
                      N_prop=N_prop, N_band=N_band, N_eva=N_eva, kx_ratio_max=kx_ratio_max)

    k1 = c.ω * sqrt(c.ϵ1)
    k2 = c.ω * sqrt(c.ϵ2)
    k3 = c.ω * sqrt(c.ϵ3)

    return SemiAnalytical_External(
        c.kx_nodes, c.wt_nodes, c.ω,
        ComplexF64(semi_pz_sign),
        c.ϵ1, c.ϵ2, c.ϵ3, c.d1, c.d2, z0s, k1, k2, k3)
end

function eval_semi_external(m::SemiAnalytical_External, x::Float64, posz_c::Float64)
    # SIGNED, not `abs(x)` -- see `eval_semi_internal`'s identical comment for the
    # full derivation. Same J1(odd)/J0(even) structure in the main kx-loop below,
    # AND in `freespace`: `s=ρ/r` is odd in ρ (r=sqrt(ρ²+z²) is even), `cv=z/r` is
    # even, so `Er*s+Eθ*cv` (Er even, Eθ odd) is a sum of two odd terms -- odd
    # overall -- while `Er*cv-Eθ*s` is even. Both cases land exactly on the
    # Cartesian Ex(odd)/Ez(even) parity a z-oriented dipole must have, so no
    # separate sign(x) correction is needed anywhere else in this function.
    ρ     = x
    z_int = m.d1
    z     = z_int - posz_c 

    function Acoeffs(kx)
        k1z = _kz_branch(m.k1, kx);  k2z = _kz_branch(m.k2, kx);  k3z = _kz_branch(m.k3, kx)
        A1, A2, A3, A4, _ = _A_coeffs(kx, m.ϵ1, m.ϵ2, m.ϵ3, k1z, k2z, k3z, m.d2)
        return A1, A2, A3, A4, k1z, k2z, k3z
    end

    function freespace(ρ, z_dip)
        r = sqrt(ρ^2 + z_dip^2)
        r < 1e-12 && return zero(ComplexF64), zero(ComplexF64)
        s = ρ/r;  cv = z_dip/r;  kr = m.k1 * r
        phase = exp(im*kr) * (m.ω^2 / (4π*r))
        Er = phase * cv * (2/kr^2 - 2im/kr)
        Eθ = phase * s  * (1/kr^2 - im/kr - 1)
        return Er*s + Eθ*cv, Er*cv - Eθ*s
    end

    sρ = zero(ComplexF64);  sz = zero(ComplexF64)

    @inbounds for ki in eachindex(m.kx_nodes)
        kx = m.kx_nodes[ki];  wk = m.wt_nodes[ki]
        A1, A2, A3, A4, k1z, k2z, k3z = Acoeffs(kx)

        if z >= 0
            ph  = exp(1im * k1z * (z + m.z0s))
            gρ  = (1im*m.μz)/(4π*m.ϵ1) * besselj1(kx*ρ) * A1 * kx * k1z * ph
            gz  = (   m.μz)/(4π*m.ϵ1) * besselj0(kx*ρ) * A1 * kx^2 * ph
            sρ -= wk * gρ;  sz -= wk * gz

        elseif z >= -m.d2
            ph  = exp(1im * k1z * m.z0s)
            gρ  = (1im*m.μz)/(4π*m.ϵ1) * besselj1(kx*ρ) * kx * ph *
                  (A2*k2z*exp(-1im*k2z*z) - A3*k2z*exp(1im*k2z*z))
            gz  = (   m.μz)/(4π*m.ϵ1) * besselj0(kx*ρ) * kx^2 * ph *
                  (A2*exp(-1im*k2z*z) + A3*exp(1im*k2z*z))
            sρ += wk * gρ;  sz -= wk * gz

        else
            ph  = exp(1im * (k1z*m.z0s - k3z*z))
            gρ  = (1im*m.μz)/(4π*m.ϵ1) * besselj1(kx*ρ) * A4 * kx * k3z * ph
            gz  = (   m.μz)/(4π*m.ϵ1) * besselj0(kx*ρ) * A4 * kx^2 * ph
            sρ += wk * gρ;  sz -= wk * gz
        end
    end

    # Add free-space contribution in layer 1
    if posz_c <= z_int
        z_dip_obs = posz_c - (m.d1 - m.z0s)
        Eρ_fs, Ez_fs = freespace(ρ, z_dip_obs)
        sρ += Eρ_fs;  sz += Ez_fs
    end

    return sρ, sz
end


# ─────────────────────────────────────────────────────────────────────────────
#  Semi-analytical model: external dipole (source in layer 1) X ORIENTED
# ─────────────────────────────────────────────────────────────────────────────

struct SemiAnalytical_External_X
    kx_nodes::Vector{Float64}
    wt_nodes::Vector{Float64}
    ω::Float64
    μz::ComplexF64
    ϵ1::ComplexF64; ϵ2::ComplexF64; ϵ3::ComplexF64
    d1::Float64
    d2::Float64
    z0s::Float64
    k1::ComplexF64
    k2::ComplexF64
    k3::ComplexF64
end

function build_semi_external_x(stack, source, pz;
                             # See _semi_common's docstring on this same raise.
                             N_prop::Int=1000,
                             N_band::Int=max(80, N_prop ÷ 2),
                             N_eva::Int=2000,
                             semi_pz_sign::Real=-1.0,
                             kx_ratio_max::Float64=80.0)
    z0s = source.z
    c = _semi_common(stack, source, pz, max(0.5*pz, Float64(real(z0s)));
                      N_prop=N_prop, N_band=N_band, N_eva=N_eva, kx_ratio_max=kx_ratio_max)

    k1 = c.ω * sqrt(c.ϵ1)
    k2 = c.ω * sqrt(c.ϵ2)
    k3 = c.ω * sqrt(c.ϵ3)

    return SemiAnalytical_External_X(
        c.kx_nodes, c.wt_nodes, c.ω,
        ComplexF64(semi_pz_sign),
        c.ϵ1, c.ϵ2, c.ϵ3, c.d1, c.d2, z0s, k1, k2, k3)
end

function eval_semi_external_x(m::SemiAnalytical_External_X, x::Float64, posz_c::Float64)
    ρ     = abs(x)
    sgnx  = sign(x)
    z_int = m.d1
    z     = z_int - posz_c

    function ABCcoeffs(kx)
        k1z = _kz_branch(m.k1, kx);  k2z = _kz_branch(m.k2, kx);  k3z = _kz_branch(m.k3, kx)
        A1, A2, A3, A4, exp2 = _A_coeffs(kx, m.ϵ1, m.ϵ2, m.ϵ3, k1z, k2z, k3z, m.d2)

        g1 = k1z - k2z; g2 = k1z + k2z
        g3 = k2z - k3z; g4 = k2z + k3z

        denB = g2*g4 + g1*g3*exp2

        B1 = 1im*kx*(g1*g4 + g2*g3*exp2) / (k1z * denB)
        B2 = 1im*(m.ϵ1/m.ϵ2)*(2*kx*g4) / denB
        B3 = 1im*(m.ϵ1/m.ϵ2)*(2*kx*g3*exp2) / denB
        B4 = 1im*(m.ϵ1/m.ϵ3)*(4*kx*k2z*exp(1im*(k2z - k3z)*m.d2)) / denB

        C1 = 1im*(k1z*A1/kx + k1z*B1/kx)
        C2 = 1im*(k1z*A2/kx - k2z*B2/kx)
        C3 = 1im*(k1z*A3/kx + k2z*B3/kx)
        C4 = 1im*(k1z*A4/kx - k3z*B4/kx)

        return A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4, k1z, k2z, k3z
    end

    # Free-space field of an x-oriented dipole, projected onto (Ex, Ez) at
    # azimuth phi=0/pi (i.e. y=0). Ex is even in x, Ez is odd in x (carries sgnx).
    function freespace_x(ρ, z_dip)
        r = sqrt(ρ^2 + z_dip^2)
        r < 1e-12 && return zero(ComplexF64), zero(ComplexF64)
        s  = ρ/r;  cv = z_dip/r;  kr = m.k1 * r
        phase = exp(im*kr) * (m.ω^2 / (4π*r))
        Ex = phase * ((1 - s^2) + (3*s^2 - 1) * (1/kr^2 - im/kr))
        Ez = sgnx * phase * s * cv * (3*(1/kr^2 - im/kr) - 1)
        return Ex, Ez
    end

    sρ = zero(ComplexF64);  sz = zero(ComplexF64)

    @inbounds for ki in eachindex(m.kx_nodes)
        kx = m.kx_nodes[ki];  wk = m.wt_nodes[ki]
        _, _, _, _, B1, B2, B3, B4, C1, C2, C3, C4, k1z, k2z, k3z = ABCcoeffs(kx)

        J0  = besselj0(kx*ρ);  J1 = besselj1(kx*ρ)
        J1ρ = ρ == 0.0 ? kx/2 : J1/ρ

        if z >= 0
            ph  = exp(1im * k1z * (z + m.z0s))
            gρ  = J1ρ*(kx*B1 - 1im*k1z*C1) + J0*(k1z^2*B1 + 1im*k1z*kx*C1)
            gz  = J1*(kx^2*C1 - 1im*kx*k1z*B1)
            sρ -= wk * gρ * ph
            sz += wk * gz * ph

        elseif z >= -m.d2
            ph  = exp(1im * k1z * m.z0s)
            gρ_up = J1ρ*(kx*B2 + 1im*k2z*C2) + J0*(k2z^2*B2 - 1im*k2z*kx*C2)
            gρ_dn = J1ρ*(kx*B3 - 1im*k2z*C3) + J0*(k2z^2*B3 + 1im*k2z*kx*C3)
            gρ  = gρ_up*exp(-1im*k2z*z) + gρ_dn*exp(1im*k2z*z)
            gz  = kx*J1 * ((1im*k2z*B2 + kx*C2)*exp(-1im*k2z*z)
                         + (-1im*k2z*B3 + kx*C3)*exp(1im*k2z*z))
            sρ -= wk * gρ * ph
            sz += wk * gz * ph

        else
            ph  = exp(1im * (k1z*m.z0s - k3z*z))
            gρ  = J1ρ*(kx*B4 + 1im*k3z*C4) + J0*(k3z^2*B4 - 1im*k3z*kx*C4)
            gz  = kx*J1*(1im*k3z*B4 + kx*C4)
            sρ -= wk * gρ * ph
            sz += wk * gz * ph
        end
    end

    sρ *= m.μz / (4π * m.ϵ1)
    sz  *= sgnx * m.μz / (4π * m.ϵ1)

    # Add free-space contribution in layer 1
    if posz_c <= z_int
        z_dip_obs = posz_c - (m.d1 - m.z0s)
        Eρ_fs, Ez_fs = freespace_x(ρ, z_dip_obs)
        sρ += Eρ_fs;  sz += Ez_fs
    end

    return sρ, sz
end



eval_semi(m::SemiAnalytical_Internal, x, posz_c) = eval_semi_internal(m, x, posz_c)
eval_semi(m::SemiAnalytical_External, x, posz_c) = eval_semi_external(m, x, posz_c)
eval_semi(m::SemiAnalytical_External_X, x, posz_c) = eval_semi_external_x(m, x, posz_c)


# ─────────────────────────────────────────────────────────────────────────────
#  True-3D Cartesian evaluation (2026-08-22): lets a semi-analytical model be
#  compared against a genuine (x,y,z) grid instead of only the y=0 line
#  `eval_semi`'s callers have always used.
# ─────────────────────────────────────────────────────────────────────────────

"""
    eval_semi_cartesian(m, x, y, posz_c) -> (Ex, Ey, Ez)

For the z-oriented models (`SemiAnalytical_Internal`/`SemiAnalytical_External`), the
field is genuinely cylindrically symmetric about the dipole's own axis (`Eφ≡0`
identically) -- `eval_semi` at `ρ=hypot(x,y)` gives the exact `(Eρ,Ez)` pair for
that ring, and the plain in-plane rotation `Ex=Eρ·cosφ, Ey=Eρ·sinφ` (with
`cosφ=x/ρ, sinφ=y/ρ`) reconstructs the true Cartesian field at ANY `(x,y)`, not
just `y=0`. At `ρ=0` (the dipole's own axis) `cosφ/sinφ` are undefined but
`Eρ→0` there regardless (besselj1(0)=0), so both components are just 0.

`SemiAnalytical_External_X` (x-oriented dipole) has no such symmetry: its
`eval_semi_external_x` derivation is specific to the y=0 mirror plane (the
plane containing the dipole axis and z) -- off that plane the field picks up
a genuinely different, azimuthal (TE-driven) `Eφ` component this model never
derived. Rather than silently fabricate a wrong off-axis value, this returns
exactly zero away from `y=0` -- callers restrict their x-oriented-dipole
Cartesian comparisons to the `y=0` row (still exact there) and fall back to a
numerical-only display off it, the same convention `evaluate_semi_analytical`
already uses for a `layers`-restricted row it never fills.
"""
function eval_semi_cartesian(m::Union{SemiAnalytical_Internal,SemiAnalytical_External},
                              x::Float64, y::Float64, posz_c::Float64)
    ρ = hypot(x, y)
    Eρ, Ez = eval_semi(m, ρ, posz_c)
    ρ == 0.0 && return zero(Eρ), zero(Eρ), Ez
    cosφ, sinφ = x / ρ, y / ρ
    return Eρ * cosφ, Eρ * sinφ, Ez
end

function eval_semi_cartesian(m::SemiAnalytical_External_X, x::Float64, y::Float64, posz_c::Float64)
    y != 0.0 && return zero(ComplexF64), zero(ComplexF64), zero(ComplexF64)
    Ex, Ez = eval_semi_external_x(m, x, posz_c)
    return Ex, zero(ComplexF64), Ez
end


end