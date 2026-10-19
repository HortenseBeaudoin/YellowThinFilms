include("ClosedForm.jl")
using LinearAlgebra, StaticArrays, Printf, FastGaussQuadrature
using Plots

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Box_Validation.jl -- the box source (plan §1.1, §5; Phases 5-7),
# against exact averages of point-source fields and independent solves.
#
#  Lateral factor S(kt) (§1.1):
#   [L1] box_lateral G = Gauss–Legendre footprint average of point fields
#        (S is exactly that average's Fourier transform), iso (rotation fill)
#        and biaxial (full solve), cross- and same-layer
#   [L2] Δx,Δy → 0 recovers the point source
#  Axial factor D, exterior of the slab (§5.1-5.2):
#   [X1] 1D cache, S ≡ 1: face-difference D (reflected + cross-layer + direct
#        |ζ| ≥ Δz/2) = exact slab average of point fields, incl. rows 0.02
#        from a face and a layer-1 (external) box
#   [X2] range guard: a thick box with |Im Q|·Δz/2 > 700 (D alone overflows)
#        and a box touching an interface: finite, = slab average
#   [X3] 2D, biaxial & gyrotropic (per-mode D, Qf₁ ≠ Qf₂ ≠ −Qb), box_lateral +
#        Δz_box = slab average of box_lateral point fields
#  Interior of the slab (§5.3):
#   [I1] ramps' constants = Ψ_p = −(iM)⁻¹Js/Δz, pins σ_b = −1 (σ_b = +1 fails)
#   [I2] Ez_local sign: |Ez(Ψ_p) + Ez_local| ∝ 1/kρ²; the other sign → 2|Ez_local|
#   [I3] Ψ_p + Ez_local = bulk 3×3 Maxwell solve at k = (kx,ky,0) for J/Δz, with
#        the conventions (∇×E = γ·iωμH, ∇×H = −iωη·ϵE + s·J) read off the code's
#        own modes (γ, η) and J_Ht (s), biaxial and gyrotropic
#   [I4] continuity of the modal fields at z_lo, z_hi (Ez_local jumps, by design)
#   [I5] interior G without Ez_local = exact slab average of point fields;
#        G(with) − G(without) lives in (Ez, p_z) only
#  [P] the source-plane profile |Ez|² through the slab, point vs box, saved as
#      Gz_ClosedForm_SourcePlane.png
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")
lay(m, d) = FlatOpticalLayer(m, d)
m_vac = IsotropicMaterial(1.0 + 0.001im)
m_gl  = IsotropicMaterial(2.25 + 0.001im)
m_uni = UniaxialMaterial(2.0 + 0.01im, 3.5 + 0.01im)
m_bi  = BiaxialMaterial(2.0 + 0.01im, 3.0 + 0.01im, 4.0 + 0.01im)
m_gyr = GyrotropicMaterial(ComplexF64[2.1+0.1im 1.2im 0; -1.2im 2.1+0.1im 0; 0 0 1.5+0.05im])
worst = Dict{String,Float64}()
upd!(k, e) = (worst[k] = max(get(worst, k, 0.0), isnan(e) ? Inf : e))

grid = uniform_kxky_grid(L=0.5, Δx=0.1)
xs, ys = grid_offsets(grid)
φs, wφ = uniform_φ(48)

# ── [L1] / [L2] ──
println("[L1/L2] lateral factor vs footprint average of point fields; Δ → 0")
function footprint_average(c, obs, zs, zo, Δx, Δy; Ng=8)
    t, w = gausslegendre(Ng)
    G = zeros(ComplexF64, length(xs), length(ys), length(zo), 3, 3)
    for a in 1:Ng, b in 1:Ng
        G .+= (w[a] * w[b] / 4) .* closed_form_G(c, obs, zs, zo, xs .- Δx / 2 * t[a], ys .- Δy / 2 * t[b])
    end
    return G
end
for (lbl, st) in (("vac/glass/vac (rotation fill)", [lay(m_vac, 1.0), lay(m_gl, 0.6), lay(m_vac, 1.0)]),
                  ("biaxial sandwich (full solve)", [lay(m_vac, 1.0), lay(m_bi, 0.8), lay(m_gl, 0.5), lay(m_vac, 1.0)]))
    kρ, wρ = kρ_grid(st)
    for src in 1:(length(st) - 1)
        c = build_cavity_transfer_2d(st, ω, src, kρ, wρ, φs, wφ)
        zs = 0.45 * st[src].d
        for obs in unique([src, 1, length(st)]), box in ((0.1, 0.1), (0.2, 0.1))
            zo = [0.1, 0.9] .* st[obs].d
            obs == src && (zo = filter(z -> abs(z - (src == 1 ? st[src].d - zs : zs)) > 0.05, zo))
            upd!("L1", relerr(closed_form_G(c, obs, zs, zo, xs, ys; box_lateral=box), footprint_average(c, obs, zs, zo, box...)))
        end
        zo = [0.9 * st[src].d]
        upd!("L2", relerr(closed_form_G(c, src, zs, zo, xs, ys; box_lateral=(1e-9, 1e-9)), closed_form_G(c, src, zs, zo, xs, ys)))
    end
    @printf("   %-30s done\n", lbl)
end

# ── [X1] / [X2] exterior D, 1D cache ──
println("[X1] exterior D (1D cache, S ≡ 1) vs exact slab average of point fields")
for (lbl, st, src, zbar, Δz) in (("glass layer", [lay(m_vac, 1.0), lay(m_gl, 1.2), lay(m_vac, 1.0)], 2, 0.55, 0.2),
                                 ("external box (layer 1)", [lay(m_vac, 1.0), lay(m_gl, 1.2), lay(m_vac, 1.0)], 1, 0.75, 0.2),
                                 ("uniaxial layer", [lay(m_vac, 1.0), lay(m_uni, 0.8), lay(m_gl, 0.5), lay(m_vac, 1.0)], 2, 0.4, 0.1))
    kx, wt, _ = point_kx_grid(st, ω, src; ρ_max=0.8, pz=0.05)
    c = build_cavity_transfer_1d(st, ω, src, kx, wt)
    zpos = src == 1 ? st[src].d - zbar : zbar
    e = 0.0
    for obs in 1:length(st)
        zo = obs == src ? filter(z -> 0 < z < st[src].d, [zpos + Δz / 2 + 0.02, zpos - Δz / 2 - 0.02, zpos + Δz / 2 + 0.25, zpos - Δz / 2 - 0.25]) :
                          [0.3, 0.8] .* st[obs].d
        for z in zo
            e = max(e, relerr(closed_form_G(c, obs, zbar, [z], xs, ys; Δz_box=Δz), slab_average(c, obs, zbar, Δz, z, xs, ys)))
        end
    end
    upd!("X1", e)
    @printf("   %-24s z̄=%.2f Δz=%.2f  worst %.2e\n", lbl, zbar, Δz, e)
end
println("[X2] range guard")
stC = [lay(m_vac, 1.0), lay(m_gl, 2.0), lay(m_vac, 1.0)]
kx, wt, _ = point_kx_grid(stC, ω, 2; ρ_max=0.8, pz=0.05)
cC = build_cavity_transfer_1d(stC, ω, 2, kx, wt)
for (lbl, zbar, Δz) in (("thick box", 1.0, 1.6), ("box touching the upper interface", 1.9, 0.2))
    imq = maximum(abs(imag(q)) for Q in vcat(cC.Qf_src, cC.Qb_src) for q in Q) * Δz / 2
    e = 0.0; fin = true
    for (obs, zo) in ((3, 0.05), (3, 0.3), (1, 0.7), (2, zbar - Δz / 2 - 0.03))
        Gn = closed_form_G(cC, obs, zbar, [zo], xs, ys; Δz_box=Δz)
        fin &= all(isfinite, Gn)
        e = max(e, relerr(Gn, slab_average(cC, obs, zbar, Δz, zo, xs, ys)))
    end
    upd!("X2", fin ? e : Inf)
    @printf("   %-34s max |Im Q|·Δz/2 = %4.0f  finite: %s  worst vs slab average %.2e\n", lbl, imq, fin, e)
end

# ── [X3] 2D, anisotropic per-mode D ──
println("[X3] 2D: box_lateral + Δz_box vs slab average of box_lateral point fields")
for (lbl, st, src, zbar, Δz) in (("biaxial", [lay(m_vac, 1.0), lay(m_bi, 0.8), lay(m_gl, 0.5), lay(m_vac, 1.0)], 2, 0.4, 0.15),
                                 ("gyrotropic", [lay(m_vac, 1.0), lay(m_gyr, 0.7), lay(m_vac, 1.0)], 2, 0.3, 0.12))
    kρ, wρ = kρ_grid(st; kmax=15.0)
    c = build_cavity_transfer_2d(st, ω, src, kρ, wρ, φs, wφ)
    e = 0.0
    for (obs, zo) in ((src, zbar + Δz / 2 + 0.03), (src, zbar - Δz / 2 - 0.1), (1, 0.6), (length(st), 0.4))
        e = max(e, relerr(closed_form_G(c, obs, zbar, [zo], xs, ys; box_lateral=(0.1, 0.1), Δz_box=Δz),
                          slab_average(c, obs, zbar, Δz, zo, xs, ys; box_lateral=(0.1, 0.1))))
    end
    upd!("X3", e)
    @printf("   %-11s worst %.2e\n", lbl, e)
end

# ── [I1]-[I3] per-node interior identities ──
function node_data(mat, kx, ky)
    md = Modes_Stack([FlatOpticalLayer(mat, 1.0)], ω, [kx, ky])[1]
    sif, sib = Source_projection_Berreman_xyz([kx, ky], ω, mat, md)
    M = ModeSolver._berreman_core(mat, IsotropicMaterial(1.0 + 0im), ω, SVector(kx, ky))
    ϵ = SMatrix{3,3,ComplexF64}(mat[i, j] for i in 1:3, j in 1:3)
    ϵtz = SVector(ϵ[1, 3], ϵ[2, 3]); ϵzz = ϵ[3, 3]; kts = SVector{2,ComplexF64}(kx, ky)
    Jt = (-im * ω) .* SMatrix{2,3,ComplexF64}(1, 0, 0, 1, 0, 0); Jz = (-im * ω) .* SVector{3,ComplexF64}(0, 0, 1)
    Js = vcat(kts * transpose(Jz) / (ϵzz * ω), ModeSolver.R2s * Jt - (ModeSolver.R2s * ϵtz) * transpose(Jz) / ϵzz)
    return (; md, sif, sib, M, ϵ, Js, Jt, Jz, kx, ky)
end
Ez_of(nd, Ψ) = [-(nd.ϵ[3, 1] * Ψ[1, q] + nd.ϵ[3, 2] * Ψ[2, q] + (nd.kx * Ψ[4, q] - nd.ky * Ψ[3, q]) / ω) / nd.ϵ[3, 3] for q in 1:3]
function Ψp_modal(nd; σb=-1)     # constant parts of the face ramps (Δz = 1)
    af = SMatrix{2,3,ComplexF64}([-nd.sif[m, q] / (im * nd.md.Qf[m]) for m in 1:2, q in 1:3])
    ab = SMatrix{2,3,ComplexF64}([σb * nd.sib[m, q] / (im * nd.md.Qb[m]) for m in 1:2, q in 1:3])
    return nd.md.Vf * af + nd.md.Vb * ab
end
Ezl(nd) = [0, 0, -1 / nd.ϵ[3, 3]]          # ez_local_density, Δz = 1
kts_test = [(0.7, 0.2), (3.1, -1.4), (9.0, 4.0), (25.0, -7.0)]

println("[I1] Ψ_p vs −(iM)⁻¹Js: σ_b = −1 (and σ_b = +1 for contrast)")
for (lbl, mat) in (("isotropic", m_gl), ("biaxial", m_bi), ("gyrotropic", m_gyr))
    e1 = e2 = 0.0
    for (kx, ky) in kts_test
        nd = node_data(mat, kx, ky); ref = -((im * nd.M) \ nd.Js)
        e1 = max(e1, relerr(Ψp_modal(nd), ref)); e2 = max(e2, relerr(Ψp_modal(nd; σb=+1), ref))
    end
    upd!("I1", e1)
    @printf("   %-11s σ_b=−1: %.2e   σ_b=+1: %.2e\n", lbl, e1, e2)
end
println("[I2] Ez_local sign: kρ²·|Ez(Ψ_p) + Ez_local|/|Ez_local| must tend to a constant")
for (lbl, mat) in (("isotropic", m_gl), ("biaxial", m_bi))
    r(kρ; s=+1) = (nd = node_data(mat, kρ * cos(0.4), kρ * sin(0.4)); abs(Ez_of(nd, Ψp_modal(nd))[3] + s * Ezl(nd)[3]) / abs(Ezl(nd)[3]))
    a, b = r(160.0) * 160^2, r(320.0) * 320^2
    upd!("I2", abs(b / a - 1))
    @printf("   %-11s kρ²·ratio at kρ=160, 320: %.2f, %.2f | wrong sign at kρ=320: %.3f\n", lbl, a, b, r(320.0; s=-1))
end
println("[I3] Ψ_p + Ez_local vs bulk 3×3 Maxwell solve at kz = 0")
function conventions(nd)       # i k×E = γ·iωμH,  i k×H = −iωη·ϵE  (μ = 1), from one eigenmode
    V = nd.md.Vf[:, 1]; k = SVector(nd.kx, nd.ky, nd.md.Qf[1])
    E = SVector(V[1], V[2], Ez_of(nd, hcat(V, V, V))[1]); ikE = im * cross(k, E)
    γ = dot(SVector(V[3], V[4]), ikE[1:2]) / (im * ω * dot(SVector(V[3], V[4]), SVector(V[3], V[4])))
    H = SVector(V[3], V[4], ikE[3] / (γ * im * ω)); rhs = -im * ω * (nd.ϵ * E)
    return γ, dot(rhs, im * cross(k, H)) / dot(rhs, rhs)
end
for (lbl, mat) in (("biaxial", m_bi), ("gyrotropic", m_gyr))
    e = 0.0
    for (kx, ky) in kts_test
        nd = node_data(mat, kx, ky); γ, η = conventions(nd)
        s = 1                           # code's J_Ht: ΔHx = Jy, ΔHy = −Jx
        k = SVector{3,ComplexF64}(kx, ky, 0)
        A = im * (k * transpose(k) - dot(k, k) * I) / (γ * ω) + im * ω * η * nd.ϵ
        Ebulk = A \ (s * vcat(nd.Jt, transpose(nd.Jz)))
        Ψp = Ψp_modal(nd)
        e = max(e, relerr(vcat(Ψp[1:2, :], transpose(Ez_of(nd, Ψp) .+ Ezl(nd))), Ebulk))
    end
    upd!("I3", e)
    @printf("   %-11s worst %.2e\n", lbl, e)
end

# ── [I4] / [I5] full G inside the slab ──
for (lbl, st, src, zbar, Δz, box) in (("biaxial", [lay(m_vac, 1.0), lay(m_bi, 0.8), lay(m_gl, 0.5), lay(m_vac, 1.0)], 2, 0.4, 0.15, (0.1, 0.1)),
                                      ("external box, layer 1", [lay(m_vac, 1.0), lay(m_gl, 0.6), lay(m_vac, 1.0)], 1, 0.7, 0.2, (0.2, 0.1)))
    kρ, wρ = kρ_grid(st; kmax=15.0)
    c = build_cavity_transfer_2d(st, ω, src, kρ, wρ, φs, wφ)
    zpos = src == 1 ? st[src].d - zbar : zbar
    kw = (box_lateral=box, Δz_box=Δz)
    e4 = 0.0
    for (zi, zo) in ((zpos - Δz / 2 + 1e-9, zpos - Δz / 2 - 1e-9), (zpos + Δz / 2 - 1e-9, zpos + Δz / 2 + 1e-9))
        e4 = max(e4, relerr(closed_form_G(c, src, zbar, [zi], xs, ys; kw..., ez_local=false), closed_form_G(c, src, zbar, [zo], xs, ys; kw...)))
    end
    e5 = 0.0; off = 0.0
    for zo in (zpos - 0.3Δz, zpos + 0.1Δz, zpos + 0.45Δz)
        G0 = closed_form_G(c, src, zbar, [zo], xs, ys; kw..., ez_local=false)
        G1 = closed_form_G(c, src, zbar, [zo], xs, ys; kw...)
        e5 = max(e5, relerr(G0, slab_average(c, src, zbar, Δz, zo, xs, ys; box_lateral=box)))
        D = G1 .- G0; off += norm(D[:, :, :, 1:2, :]) + norm(D[:, :, :, 3, 1:2])
    end
    upd!("I4", e4); upd!("I5", e5); upd!("I5-offdiag", off)
    @printf("[I4/I5] %-22s continuity at the faces %.2e | interior vs point-field average %.2e | Ez_local outside (Ez,p_z): %.1e\n",
            lbl, e4, e5, off)
end

# ── [P] source-plane profile ──
println("[P] |Ez|² through the source slab (p_z, box 0.3×0.3×0.2, 5 px off-axis)")
stF = [lay(m_vac, 2.0), lay(m_vac, 2.0), lay(m_vac, 2.0)]
Δzb, box, z0 = 0.2, (0.3, 0.3), 1.0
ζ = 0.0125 .* (collect(-24:23) .+ 0.5); zq = z0 .+ ζ   # no sample exactly on ζ = 0 (a point source is undefined there)
kx, wt, _ = point_kx_grid(stF, ω, 2; ρ_max=0.6, pz=0.05)
cpt = build_cavity_transfer_1d(stF, ω, 2, kx, wt; obs_layers=[2])
kρF, wρF, φF, wφF, _ = box_kρ_grid(stF, ω, 2; box=(box..., Δzb), N_phi=96, rtol=1e-3, kρ_cap=1500.0, ρ_far=0.6)
cbx = build_cavity_transfer_2d(stF, ω, 2, kρF, wρF, φF, wφF; obs_layers=[2])
Ipt = abs2.(closed_form_G(cpt, 2, z0, zq, [0.5], [0.0])[1, 1, :, 3, 3])
Ibx = abs2.(closed_form_G(cbx, 2, z0, zq, [0.5], [0.0]; box_lateral=box, Δz_box=Δzb)[1, 1, :, 3, 3])
ins = abs.(ζ) .< Δzb / 2
@printf("   inside the slab: max |Ez|² point %.3e | box %.3e ; total variation point %.3e | box %.3e\n",
        maximum(Ipt[ins]), maximum(Ibx[ins]), sum(abs.(diff(Ipt[ins]))), sum(abs.(diff(Ibx[ins]))))
plt = plot(ζ, Ipt; lw=2, color=:crimson, label="point source", yscale=:log10, xlabel="ζ = z' − z̄",
           ylabel="|Ez|² (p_z, 5 px off-axis)", title="Source-plane profile: box $(box[1])×$(box[2])×$(Δzb)", legend=:top, size=(900, 550))
plot!(plt, ζ, Ibx; lw=2, color=:darkorange, label="box: S(kt)·D seed + interior slab + Ez_local")
vspan!(plt, [-Δzb / 2, Δzb / 2]; alpha=0.12, color=:gray, label="source slab")
savefig(plt, "Gz_ClosedForm_SourcePlane.png")
println("   saved Gz_ClosedForm_SourcePlane.png")

println()
for k in sort(collect(keys(worst)))
    @printf("  [%s] worst %.3e\n", k, worst[k])
end
ok = worst["L1"] < 1e-9 && worst["L2"] < 1e-12 && worst["X1"] < 1e-8 && worst["X2"] < 1e-8 && worst["X3"] < 1e-8 &&
     worst["I1"] < 1e-10 && worst["I2"] < 0.05 && worst["I3"] < 1e-10 && worst["I4"] < 1e-6 && worst["I5"] < 1e-8 && worst["I5-offdiag"] == 0.0
println(ok ? "BOX: PASS" : "BOX: FAIL")
