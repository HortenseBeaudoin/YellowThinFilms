include(joinpath(@__DIR__, "ClosedForm.jl"))
using LinearAlgebra, Printf, FastGaussQuadrature

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Quadrature_Validation.jl -- the feature-driven k-space quadrature
# (mod_KSpaceFeatures.jl, HANDOUT_quadrature_hyperbolic_gyrotropic.md), 2026-11-06.
#
#  [L] light lines (full ε, μ tensors) and material classes, vs closed forms
#  [W] _walk_stack's conditioning guard: the field transmitted into a hyperbolic
#      slab is continuous in kρ (it used to be zeroed past kρ ≈ 0.142 nm⁻¹)
#  [Z] ZrS₂ slab on Au under Si (hyperbolic, 1D, nm): point_kx_grid (auto → :fixed)
#      vs a 44k-node hand-built Gauss–Legendre rule (was Gz_TailStall_Validation.jl)
#  [H] hBN film on BaF₂, γ = 1 cm⁻¹, 1500 and 812 cm⁻¹ (1D, µm): vs the hand rule of
#      Fig_hBN_BackwardWave.jl with every panel quartered and the cut at 25/h (the
#      halved one is itself off by 1.2e-6: cut at 120 µm⁻¹, no grading onto k₀)
#  [U] dipole above a magnetic half-space (μ ≠ 1; isotropic → 1D, uniaxial → 2D):
#      vs the Sommerfeld integrals with analytic Fresnel coefficients
#  [I] InSb magnetoplasmon, Ω = 1.09, γ = 1e-3 ω_p (gyrotropic, 2D, scattered field
#      on the source plane, ±30 µm lines): point_kρ_grid_2d vs the same rule refined
#      (pole panels ×2, grading 2⁻²⁰, background ×2), and vs 2N_φ; the handout's hand
#      rule for scale
#  [M] twisted α-MoO₃ bilayer, 72° (in-plane hyperbolic, 2D, scattered field on the
#      source plane, ±0.3 µm lines): point_kρ_grid_2d vs the hand rule of
#      Fig_MoO3_Twisted.jl with panels halved, grading 2⁻⁴⁰ instead of 2⁻¹⁰ (at 2⁻¹⁰ it
#      is itself off by 6.8e-6) and the same cut-off; and vs 2N_φ
#  [C] check=true reports the halving change (1D and 2D)
# Each rule is built with its default knobs (rtol = 1e-6).
# Run from the repository root: julia --project -t 8 Gz_ClosedForm_Quadrature_Validation.jl
# ─────────────────────────────────────────────────────────────────────────────

results = Tuple{String,Float64,Float64}[]
report(lbl, err, tol) = (push!(results, (lbl, err, tol));
                         @printf("  %-64s %.2e  (tol %.0e)  %s\n", lbl, err, tol, err <= tol ? "PASS" : "FAIL"))
function gl_panels(a, b, w; order = 8)
    t, wq = gausslegendre(order); n = max(1, ceil(Int, (b - a) / w)); h = (b - a) / n
    ks = Float64[]; ws = Float64[]
    for j in 0:(n - 1); append!(ks, h / 2 .* t .+ (a + (j + 0.5) * h)); append!(ws, h / 2 .* wq); end
    return ks, ws
end
function rule_from(spec)                                   # [(a, b, panel width), …]
    k = Float64[]; w = Float64[]
    for (a, b, pw) in spec; k_, w_ = gl_panels(a, b, pw); append!(k, k_); append!(w, w_); end
    return k, w
end
lorentz(ν, ε∞, TO, LO, γ) = ε∞ * (1 + (LO^2 - TO^2) / (TO^2 - ν^2 - im * ν * γ))
wn2k_um(ν) = 2π * ν * 1e-4
relerr(a, b) = norm(a - b) / norm(b)

# ── [L] ──
println("[L] light lines and material classes")
ω = 2π
ll(m, μ = nothing) = layer_light_lines(μ === nothing ? FlatOpticalLayer(m, 1.0) : FlatLayer(m, μ, 1.0), ω, 0.0)
e = abs(only(ll(IsotropicMaterial(2.25 + 0.01im))) - ω * real(sqrt(2.25 + 0.01im)))
εt, εz, μt, μz = 3 + 0.1im, -2 + 0.1im, 1.6 + 0.05im, 0.7 + 0.05im       # TE √(εt μz), TM √(εz μt) (Re < 0: none)
L = ll(UniaxialMaterial(εt, εz), UniaxialMaterial(μt, μz))
e = max(e, length(L) == 1 ? abs(L[1] - ω * real(sqrt(εt * μz))) : Inf)
L = ll(IsotropicMaterial(2.25 + 0im), IsotropicMaterial(1.5 + 0im))
e = max(e, abs(only(L) - ω * sqrt(2.25 * 1.5)))
ε1, g, ε3 = 1.0 + 0.0im, 0.3 + 0.0im, 2.0 + 0.0im                         # Voigt: x̂ → (ε1² − g²)/ε1 and ε3; ŷ → ε1 ± g
vg = GyrotropicMaterial(ComplexF64[ε1 0 im*g; 0 ε3 0; -im*g 0 ε1])
Lx = layer_light_lines(FlatOpticalLayer(vg, 1.0), ω, 0.0); Ly = layer_light_lines(FlatOpticalLayer(vg, 1.0), ω, π / 2)
e = max(e, maximum(abs.(Lx .- sort(ω .* real.(sqrt.([(ε1^2 - g^2) / ε1, ε3]))))),
           maximum(abs.(Ly .- sort(ω .* real.(sqrt.([ε1 - g, ε1 + g]))))))
report("light lines: iso, uniaxial ε & μ, magnetic, Voigt along x̂ and ŷ", e / ω, 1e-12)
hbn(ν) = UniaxialMaterial(lorentz(ν, 4.99, 1361.0, 1614.0, 1.0), lorentz(ν, 2.94, 760.5, 825.0, 1.0))
cls = [material_class(FlatOpticalLayer(m, 1.0)) for m in
       (IsotropicMaterial(2.25 + 0.01im), IsotropicMaterial(-20.0 + 1im), hbn(1500.0), hbn(812.0), vg,
        GyrotropicMaterial(ComplexF64[2 0.5 0; 0.5 -3 0; 0 0 1]))]
want = [:dielectric, :metallic, :hyperbolic, :hyperbolic, :gyrotropic, :hyperbolic]
report("material_class (dielectric, metal, hBN ×2, Voigt, tilted)", Float64(count(cls .!= want)), 0.0)
eng = [choose_engine([FlatOpticalLayer(IsotropicMaterial(1.0 + 0im), 1.0), FlatOpticalLayer(m, 1.0)])
       for m in (IsotropicMaterial(2.25 + 0.01im), hbn(1500.0))]
report("choose_engine: dielectric → :adaptive, hyperbolic → :fixed", Float64(eng != [:adaptive, :fixed]), 0.0)

# ── ZrS₂ stack (nm) ──
ν = 250.0; ωz = 2π * ν * 1e-7
ε_t = lorentz(ν, 9.4, 181.0, 351.0, 17.0); ε_z = lorentz(ν, 6.26, 346.0, 368.0, 16.2)
ε_Au = 1 - 72800.0^2 / (ν^2 + im * ν * 216.0)
stZ = [FlatOpticalLayer(IsotropicMaterial(11.7 + 0im), 300.0), FlatOpticalLayer(UniaxialMaterial(ε_t, ε_z), 150.0),
       FlatOpticalLayer(IsotropicMaterial(9.0 + 0.1im), 5.0), FlatOpticalLayer(IsotropicMaterial(ε_Au), 150.0)]
dipZ = Dipole_constructor(:x, ωz, 0.0, 30.0, 1)
xsZ = collect(-1900.0:25.0:1900.0); zsZ = [10.0, 75.0, 140.0]
fieldZ(k, w) = closed_form_G(build_cavity_transfer_1d(stZ, ωz, 1, k, w), 2, dipZ.z, zsZ, xsZ, [0.0])[:, 1, :, :, 1]

# ── [W] ──
println("\n[W] transmitted field into the ZrS₂ slab is continuous across kρ = 0.14209 nm⁻¹")
gk(k) = closed_form_G(build_cavity_transfer_1d(stZ, ωz, 1, [k], [1.0]), 2, dipZ.z, [75.0], [0.0], [0.0])[1, 1, 1, :, 1]
report("|G(0.1420890) − G(0.1420880)| / |G(0.1420880)|", relerr(gk(0.142089), gk(0.142088)), 1e-3)

# ── [Z] ──
println("\n[Z] ZrS₂ slab on Au under Si, x-dipole 30 nm above, field in the slab (1D)")
kr, wr = rule_from(((0.0, 2ωz * sqrt(11.7), ωz / 80), (2ωz * sqrt(11.7), 0.2, 6.25e-5), (0.2, 1.0, 6.25e-4), (1.0, 4.0, 6.25e-3)))
Eref = fieldZ(kr, wr)
t = @elapsed kx, wt, info = point_kx_grid(stZ, ωz, 1; ρ_max=1900.0, pz=2.5, kρ_cap=4.0, check=true)
eZ = maximum(relerr(fieldZ(kx, wt)[:, j, :], Eref[:, j, :]) for j in eachindex(zsZ))
@printf("  engine %s, %d nodes (reference %d), %d poles, %.1f s\n", info.engine, length(kx), length(kr), length(info.features.poles), t)
report("point_kx_grid vs 44k-node reference", eZ, 1e-6)
report("[C] check=true: halving change of the probes", info.check.rel_change, 1e-6)

# ── [H] ──
println("\n[H] hBN 100 nm on BaF₂, γ = 1 cm⁻¹, z-dipole 0.2 µm above, Ez 0.05 µm above the film (1D)")
ε_BaF2(ν) = lorentz(ν, 2.16, 184.0, 184.0 * sqrt(7.33 / 2.16), 3.0)
ρsH = collect(0.0:0.02:7.08)
for νh in (1500.0, 812.0)
    ωh = wn2k_um(νh)
    st = [FlatOpticalLayer(IsotropicMaterial(1.0 + 0im), 1.0), FlatOpticalLayer(hbn(νh), 0.1),
          FlatOpticalLayer(IsotropicMaterial(ε_BaF2(νh)), 1.0)]
    dip = Dipole_constructor(:z, ωh, 0.0, 0.2, 1)
    Ez(k, w) = closed_form_G(build_cavity_transfer_1d(st, ωh, 1, k, w), 1, dip.z, [st[1].d - 0.05], ρsH, [0.0])[:, 1, 1, 3, 3]
    kh, wh = rule_from(((0.0, 2ωh, ωh / 160), (2ωh, 8.0, 5e-4), (8.0, 40.0, 5e-3), (40.0, 25 / 0.15, 0.025)))
    t = @elapsed kx, wt, info = point_kx_grid(st, ωh, 1; ρ_max=7.08, h_min=0.15)
    @printf("  %.0f cm⁻¹: engine %s, %d nodes (hand rule quartered: %d), %d poles, %.1f s\n",
            νh, info.engine, length(kx), length(kh), length(info.features.poles), t)
    report(@sprintf("%.0f cm⁻¹: point_kx_grid vs quartered hand rule", νh), relerr(Ez(kx, wt), Ez(kh, wh)), 1e-6)
end

# ── [U] ──
println("\n[U] dipole at height d above air / (ε, μ) half-space: G^scat on axis vs Sommerfeld")
const k0u = 1.0
kzroot(x) = (q = sqrt(complex(x)); imag(q) < 0 ? -q : q)
function fresnel(k, εt, εz, μt, μz)
    kz1 = kzroot(k0u^2 - k^2)
    kzs = kzroot(εt * μt * k0u^2 - k^2 * μt / μz); kzp = kzroot(εt * μt * k0u^2 - k^2 * εt / εz)
    return (μt * kz1 - kzs) / (μt * kz1 + kzs), (εt * kz1 - kzp) / (εt * kz1 + kzp)
end
gl(a, b, n) = (t = gausslegendre(n); ((b - a) / 2 .* t[1] .+ (a + b) / 2, (b - a) / 2 .* t[2]))
graded(x, lo, hi) = filter(t -> lo < t < hi, vcat(x, [x + sg * 0.5^j for j in 1:40, sg in (-1, 1)]...))
function panels(br, n)
    br = sort(unique(br)); xs = Float64[]; ws = Float64[]
    for i in 1:(length(br) - 1); x, w = gl(br[i], br[i + 1], n); append!(xs, x); append!(ws, w); end
    return xs, ws
end
function branchpts(m)
    pts = sqrt.([m.ε[1] * m.μ[2], m.μ[1] * m.ε[2]])
    if m.ε[1] == m.ε[2] && m.μ[1] == m.μ[2]
        ε, μ = m.ε[1], m.μ[1]; append!(pts, sqrt.([ε * (μ - ε) / (1 - ε^2), μ * (ε - μ) / (1 - μ^2)]))
    end
    return filter(x -> x > 0, unique(real.(pts)))
end
function sommerfeld(m, d)          # s < 1 in s_z, s > 1 in v = √(s² − 1): smooth at s = 1 (Test_Mu.jl [2])
    kb = branchpts(m) ./ k0u
    szb = [sqrt(1 - s^2) for s in kb if s < 1]; vb = [sqrt(s^2 - 1) for s in kb if s > 1]
    vmax = 50 / (2k0u * d); Izz = 0.0im; Ixx = 0.0im
    x, w = panels(vcat(collect(0:0.025:1.0), [graded(b, 0.0, 1.0) for b in szb]...), 20)
    for (sz, wi) in zip(x, w)
        s = sqrt(1 - sz^2); rs, rp = fresnel(s * k0u, m.ε..., m.μ...); e = cis(2k0u * d * sz)
        Izz += wi * s^2 * rp * e; Ixx += wi * (rs - sz^2 * rp) * e
    end
    x, w = panels(filter(<=(vmax), vcat(0.0, 1e-4 .* 1.15 .^ (0:120), vmax, [graded(b, 0.0, vmax) for b in vb]...)), 16)
    for (v, wi) in zip(x, w)
        s = sqrt(1 + v^2); rs, rp = fresnel(s * k0u, m.ε..., m.μ...); e = exp(-2k0u * d * v)
        Izz += wi * (-im) * s^2 * rp * e; Ixx += wi * (-im) * (rs + v^2 * rp) * e
    end
    return im * k0u^3 / 4π * Izz, im * k0u^3 / 8π * Ixx
end
mats = [(name = "ε=2.25, μ=1.5",             ε = (2.25 + 0im, 2.25 + 0im), μ = (1.5 + 0im, 1.5 + 0im)),
        (name = "ε=−3+0.2i, μ=2+0.1i",        ε = (-3 + 0.2im, -3 + 0.2im), μ = (2 + 0.1im, 2 + 0.1im)),
        (name = "ε=−2+0.3i, μ=−1.5+0.3i",     ε = (-2 + 0.3im, -2 + 0.3im), μ = (-1.5 + 0.3im, -1.5 + 0.3im)),
        (name = "uniaxial ε=(3,−2), μ=(1.6,0.7)", ε = (3 + 0.1im, -2 + 0.1im), μ = (1.6 + 0.05im, 0.7 + 0.05im))]
for m in mats, d in (0.2, 1.0)
    iso = m.ε[1] == m.ε[2] && m.μ[1] == m.μ[2]
    st = iso ? [FlatOpticalLayer(IsotropicMaterial(1.0 + 0im), 2d), FlatLayer(IsotropicMaterial(m.ε[1]), IsotropicMaterial(m.μ[1]), 1.0)] :
               [FlatOpticalLayer(IsotropicMaterial(1.0 + 0im), 2d), FlatLayer(UniaxialMaterial(m.ε...), UniaxialMaterial(m.μ...), 1.0)]
    G = if iso
        kx, wt, _ = point_kx_grid(st, k0u, 1; h_min=2d)
        closed_form_G(build_cavity_transfer_1d(st, k0u, 1, kx, wt), 1, d, [d], [0.0], [0.0]; terms=:scattered)[1, 1, 1, :, :]
    else
        kρ, wρ, φs, wφ, _ = point_kρ_grid_2d(st, k0u, 1; ρ_max=0.0, h_min=2d, N_phi=16)   # on axis |m| ≤ 2: exact
        closed_form_G(build_cavity_transfer_2d(st, k0u, 1, kρ, wρ, φs, wφ; obs_layers=[1]), 1, d, [d], [0.0], [0.0];
                      terms=:scattered)[1, 1, 1, :, :]
    end
    Gzz, Gxx = sommerfeld(m, d)
    e = max(abs(G[3, 3] - Gzz) / abs(Gzz), abs(G[1, 1] - Gxx) / abs(Gxx), abs(G[2, 2] - Gxx) / abs(Gxx))
    report(@sprintf("%-30s d=%.1f [%s]", m.name, d, iso ? "1D" : "2D"), e, 1e-6)
end

# ── [I] ──
println("\n[I] InSb magnetoplasmon, Ω = 1.09, z-dipole 5 µm above, Gzz^scat on the source plane (2D)")
const ωp_um = 2π * 2e12 / 2.99792458e14
function ε_InSb(Ω; γ = 0.001, ωc = 0.25, ε∞ = 15.6)
    w = Ω + im * γ
    a = ε∞ * (1 - w / (Ω * (w^2 - ωc^2))); g = ε∞ * ωc / (Ω * (w^2 - ωc^2)); c = ε∞ * (1 - 1 / (Ω * w))
    GyrotropicMaterial(ComplexF64[a 0 im*g; 0 c 0; -im*g 0 a])
end
Ωi = 1.09; k0i = Ωi * ωp_um; hi = 5.0
stI = [FlatOpticalLayer(IsotropicMaterial(1.0 + 0im), 2hi), FlatOpticalLayer(ε_InSb(Ωi), 1.0)]
zsI = Dipole_constructor(:z, k0i, 0.0, hi, 1).z
xl = collect(-30.0:1.0:30.0)
# linear in the nodes: kρ chunks (chunked_2d) and 2N_φ = mean of N_φ and N_φ shifted by π/N_φ, one ≤1.5M-node cache at a time
GzzI(c) = vcat(vec(closed_form_G(c, 1, zsI, [zsI], xl, [0.0]; terms=:scattered)[:, 1, 1, 3, 3]),
               vec(closed_form_G(c, 1, zsI, [zsI], [0.0], xl; terms=:scattered)[1, :, 1, 3, 3]))
function Gline(kρ, wρ, Nφ; shift=0.0)
    φn = [2π * p / Nφ + shift for p in 0:Nφ-1]
    return chunked_2d(GzzI, stI, k0i, 1, kρ, wρ, φn, fill(2π / Nφ, Nφ); obs_layers=[1])
end
G2φ(G, kρ, wρ, Nφ) = 0.5 .* (G(kρ, wρ, Nφ) .+ G(kρ, wρ, Nφ; shift=π / Nφ))
t = @elapsed kρ, wρ, φs, wφ, info = point_kρ_grid_2d(stI, k0i, 1; ρ_max=30.0, h_min=2hi, terms=:scattered)
@printf("  %d kρ × %d φ = %.1fM nodes (N_φ: lateral %d, pole tracks %d), %d poles, %.1f s\n", info.n_kρ, info.N_phi,
        info.n_nodes / 1e6, info.features.N_φ_lateral, info.features.N_φ_poles, length(info.features.poles), t)
G0 = Gline(kρ, wρ, info.N_phi)
kf, wf, _ = fixed_kρ_rule(info.features; pole_safety=0.25, grade=20, r_rel=0.025, c_h=2.0, c_ρ=2.0)
report("point_kρ_grid_2d vs refined kρ rule (same N_φ)", relerr(G0, Gline(kf, wf, info.N_phi)), 1e-6)
report("point_kρ_grid_2d vs 2N_φ", relerr(G0, G2φ(Gline, kρ, wρ, info.N_phi)), 1e-6)
kh = Float64[]; wh = Float64[]                               # Fig_InSb_OneWay.jl's rule, cut at Kcut
for (a, b, pw) in vcat([(k0i * (1 - 0.5^(j - 1)), k0i * (1 - 0.5^j), k0i * 0.5^j) for j in 1:12], [(k0i * (1 - 0.5^12), k0i, k0i * 0.5^12)],
                       [(k0i, k0i * (1 + 0.5^12), k0i * 0.5^12)],
                       [(k0i * (1 + 0.5^j), k0i * (1 + 0.5^(j - 1)), min(k0i / 400, k0i * 0.5^j)) for j in 12:-1:6],
                       [(k0i * 1.03125, 2.5k0i, k0i / 400), (2.5k0i, 6k0i, k0i / 100), (6k0i, info.tail_bound, k0i / 10)])
    k_, w_ = gl_panels(min(a, b), max(a, b), pw); append!(kh, k_); append!(wh, w_)
end
@printf("  for scale: the handout's hand rule (pole panels k₀/400) differs by %.1e\n", relerr(Gline(kh, wh, info.N_phi), G0))

# ── [M] ──
println("\n[M] twisted α-MoO₃ bilayer (72°) on BaF₂, 930 cm⁻¹, z-dipole 0.1 µm above, Gzz^scat on the source plane (2D)")
νm = 930.0; ωm = wn2k_um(νm)
function ε_MoO3(θdeg)
    ex = lorentz(νm, 5.78, 820.0, 972.0, 4.0); ey = lorentz(νm, 6.07, 545.0, 851.0, 4.0); ez = lorentz(νm, 4.47, 958.0, 1004.0, 2.0)
    c, s = cosd(θdeg), sind(θdeg); R = [c -s 0; s c 0; 0 0 1]
    GyrotropicMaterial(ComplexF64.(R * Diagonal([ex, ey, ez]) * R'))
end
stM = [FlatOpticalLayer(IsotropicMaterial(1.0 + 0im), 0.2), FlatOpticalLayer(ε_MoO3(0.0), 0.1),
       FlatOpticalLayer(ε_MoO3(72.0), 0.1), FlatOpticalLayer(IsotropicMaterial(ε_BaF2(νm)), 1.0)]
zsM = Dipole_constructor(:z, ωm, 0.0, 0.1, 1).z
xm = collect(-0.3:0.03:0.3)
GzzM(c) = vcat(vec(closed_form_G(c, 1, zsM, [zsM], xm, [0.0]; terms=:scattered)[:, 1, 1, 3, 3]),
               vec(closed_form_G(c, 1, zsM, [zsM], [0.0], xm; terms=:scattered)[1, :, 1, 3, 3]))
function GlineM(kρ, wρ, Nφ; shift=0.0)
    φn = [2π * p / Nφ + shift for p in 0:Nφ-1]
    return chunked_2d(GzzM, stM, ωm, 1, kρ, wρ, φn, fill(2π / Nφ, Nφ); obs_layers=[1])
end
t = @elapsed kρ, wρ, φs, wφ, info = point_kρ_grid_2d(stM, ωm, 1; ρ_max=0.3, h_min=0.2, terms=:scattered)
@printf("  %d kρ × %d φ = %.1fM nodes (N_φ: lateral %d, pole tracks %d), %d poles, %.1f s\n", info.n_kρ, info.N_phi,
        info.n_nodes / 1e6, info.features.N_φ_lateral, info.features.N_φ_poles, length(info.features.poles), t)
G0 = GlineM(kρ, wρ, info.N_phi)
kn = sqrt(real(ε_BaF2(νm))) * ωm
br = sort(unique(vcat(0.0, ωm .* (1 .- 0.5 .^ (1:40)), ωm, ωm .* (1 .+ 0.5 .^ (40:-1:1)),
                      kn .* (1 .- 0.5 .^ (2:40)), kn, kn .* (1 .+ 0.5 .^ (40:-1:2)), 1.0, info.tail_bound)))
kh, wh = rule_from([(br[i], br[i+1], min(0.025, br[i+1] - br[i])) for i in 1:length(br)-1])
# kρ only, so both at 128 φ (the hand rule has 20× our kρ nodes: at our N_φ it is a 28M-node, 40-min reference)
report("point_kρ_grid_2d vs hand rule (0.025 µm⁻¹, graded 2⁻⁴⁰), both at 128 φ", relerr(GlineM(kρ, wρ, 128), GlineM(kh, wh, 128)), 1e-6)
report("point_kρ_grid_2d vs 2N_φ", relerr(G0, G2φ(GlineM, kρ, wρ, info.N_phi)), 1e-6)

# ── [C] 2D check ──
println("\n[C] check=true on the 2D builder (MoO₃, ρ_max = 0.2)")
_, _, _, _, info = point_kρ_grid_2d(stM, ωm, 1; ρ_max=0.2, h_min=0.2, terms=:scattered, check=true)
report("halved kρ panels", info.check.kρ.rel_change, 1e-6)
report("2N_φ", info.check.φ.rel_change, 1e-6)

nfail = count(r -> !(r[2] <= r[3]), results)
println("\nQUADRATURE: ", nfail == 0 ? "PASS" : "FAIL ($nfail of $(length(results)))")
