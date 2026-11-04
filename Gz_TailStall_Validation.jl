include(joinpath(@__DIR__, "ClosedForm.jl"))
using LinearAlgebra, Printf, FastGaussQuadrature

# ─────────────────────────────────────────────────────────────────────────────
# Gz_TailStall_Validation.jl -- regression test for adaptive_kx_grid's tail-stall
# handling (tail_stall = :keep vs :fallback), 2026-11-03.
#
# Case: x-dipole 30 nm above a 150 nm ZrS₂ slab (uniaxial, type-II hyperbolic at
# 250 cm⁻¹) on 5 nm Al₂O₃ / Au, under a Si half-space. Lengths in nm, ω = k₀.
# Every polariton pole sits far past kmax (the Si light line), in the "tail", and a
# few of point_kx_grid's probes stall there. Field in the slab vs a 44k-node fixed
# Gauss–Legendre reference.
# Expected: :fallback ≈ 2e-1 rel. error (475 nodes), :keep ≈ 1e-3 (≈14k nodes).
# Run from the repository root: julia --project Gz_TailStall_Validation.jl
# ─────────────────────────────────────────────────────────────────────────────

ν = 250.0; ω = 2π * ν * 1e-7
lor(ν, ε∞, TO, LO, γ) = ε∞ * (1 + (LO^2 - TO^2) / (TO^2 - ν^2 - im * ν * γ))
ε_t = lor(ν, 9.4, 181.0, 351.0, 17.0); ε_z = lor(ν, 6.26, 346.0, 368.0, 16.2)
ε_Au = 1 - 72800.0^2 / (ν^2 + im * ν * 216.0)
stack = [FlatOpticalLayer(IsotropicMaterial(11.7 + 0im), 300.0),
         FlatOpticalLayer(UniaxialMaterial(ε_t, ε_z), 150.0),
         FlatOpticalLayer(IsotropicMaterial(9.0 + 0.1im), 5.0),
         FlatOpticalLayer(IsotropicMaterial(ε_Au), 150.0)]
dip = Dipole_constructor(:x, ω, 0.0, 30.0, 1)
xs = collect(-1900.0:25.0:1900.0); zs = [10.0, 75.0, 140.0]      # depths in the slab

field(k, w) = closed_form_G(build_cavity_transfer_1d(stack, ω, 1, k, w), 2, dip.z, zs, xs, [0.0])[:, 1, :, :, 1]

function gl_panels(a, b, w; order = 8)
    t, wq = gausslegendre(order); n = max(1, ceil(Int, (b - a) / w)); h = (b - a) / n
    ks = Float64[]; ws = Float64[]
    for j in 0:(n - 1); append!(ks, h / 2 .* t .+ (a + (j + 0.5) * h)); append!(ws, h / 2 .* wq); end
    return ks, ws
end
kr = Float64[]; wr = Float64[]
for (a, b, w) in ((0.0, 2ω * sqrt(11.7), ω / 80), (2ω * sqrt(11.7), 0.2, 6.25e-5), (0.2, 1.0, 6.25e-4), (1.0, 4.0, 6.25e-3))
    k_, w_ = gl_panels(a, b, w); append!(kr, k_); append!(wr, w_)
end
Eref = field(kr, wr)
relerr(E) = maximum(norm(E[:, j, :] - Eref[:, j, :]) / norm(Eref[:, j, :]) for j in eachindex(zs))

# point_kx_grid uses adaptive_kx_grid's default tail_stall (:keep after the fix).
# Before the fix this same call gave 475 nodes and 2.3e-1.
kx, wt, info = point_kx_grid(stack, ω, 1; ρ_max = maximum(abs, xs), pz = 2.5, maxevals = 10_000, kρ_cap = 4.0)
e = relerr(field(kx, wt))
@printf("reference: %d GL nodes\n", length(kr))
@printf("point_kx_grid (tail_stall = :keep default): %d nodes, tail stalled = %s, rel. error %.1e  -> %s\n",
        length(kx), info.any_tail_stalled, e, e < 1e-2 ? "PASS" : "FAIL")
