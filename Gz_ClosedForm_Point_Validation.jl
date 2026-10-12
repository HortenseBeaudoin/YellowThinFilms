include("ClosedForm.jl")
include("mod_Dipz_SemiAnalytical.jl")
using .ZSemiAnalytical
using LinearAlgebra, StaticArrays, Printf

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Point_Validation.jl -- point sources (plan Phases 1-4 and the
# §4.4 assembly), against independent references and physical identities:
#
#  [a] point_kx_grid + CavityTransferCache1D vs the semi-analytical model
#      (mod_Dipz_SemiAnalytical.jl), vacuum, source outside (layer 1) and
#      inside (layer 2) the stack, whole stack. Rows within 0.3 of the source
#      are excluded: the semi-analytical INTERNAL model sizes its own evanescent
#      grid for the source-to-interface distance (its `zref`), so it is not
#      itself converged closer than that (confirmed: a dense 64k-node fixed rule
#      and point_kx_grid agree to ~2e-5 there, both ~1e-1 off the model at
#      |ζ| = 0.15). Those rows are reported, not failed on.
#  [b] closed_form_field: batched multi-dipole (several depths, several lateral
#      positions and moments) = Σ single-dipole applies, 1D and 2D caches
#  [c] 2D cache on an azimuthally symmetric stack: rotation fill = full per-node
#      solve = 1D path on the same kρ nodes; NUFFT = exact separable sum
#  [d] 90° rotation covariance, full-solve branch: G_B(R r) = R G_A(r) Rᵀ with B
#      the medium A rotated by R (biaxial diag(a,b,c) → diag(b,a,c)); a
#      gyrotropic medium (ϵxy = −ϵyx = iγ) is itself invariant, G(R r) = R G(r) Rᵀ
#  [e] reciprocity between two internal layers, G_ij(r ← r') = G_ji(r' ← r),
#      isotropic (1D cache) and biaxial (2D cache)
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")
lay(m, d) = FlatOpticalLayer(m, d)
m_vac = IsotropicMaterial(1.0 + 0.001im)
m_gl  = IsotropicMaterial(2.25 + 0.001im)
m_uni = UniaxialMaterial(2.0 + 0.01im, 3.5 + 0.01im)
m_bi  = BiaxialMaterial(2.0 + 0.01im, 3.0 + 0.01im, 4.0 + 0.01im)
m_biT = BiaxialMaterial(3.0 + 0.01im, 2.0 + 0.01im, 4.0 + 0.01im)       # m_bi rotated by 90° about z
m_gyr = GyrotropicMaterial(ComplexF64[2.1+0.1im 1.2im 0; -1.2im 2.1+0.1im 0; 0 0 1.5+0.05im])
passed = Dict{String,Bool}()

# ── [a] vs semi-analytical ──
println("[a] point_kx_grid + 1D cache vs semi-analytical (vacuum, whole stack, rows with |ζ| > 0.3)")
stack_v = [lay(m_vac, 2.0), lay(m_vac, 2.0), lay(m_vac, 2.0)]
pz = 0.1
grid = uniform_kxky_grid(L=1.0, Δx=0.1)
xs, ys = grid_offsets(grid)
rows, cum = stack_rows(stack_v, pz)
worst_a = 0.0
for (label, dip, builder) in (("external z-dipole (layer 1, z=1.95)", Dipole_constructor(:z, ω, 0.0, 1.95, 1), build_semi_external),
                              ("internal z-dipole (layer 2, z=1.0)",  Dipole_constructor(:z, ω, 0.0, 1.0, 2),  build_semi_internal))
    src = dip.layer
    t_grid = @elapsed kx, wt, info = point_kx_grid(stack_v, ω, src; ρ_max=hypot(1.0, 1.0), pz=pz)
    t_cache = @elapsed c = build_cavity_transfer_1d(stack_v, ω, src, kx, wt)
    E = zeros(ComplexF64, length(xs), length(ys), length(rows), 3)
    t_G = @elapsed for o in 1:3
        idx = [j for j in eachindex(rows) if rows[j][1] == o]
        G = closed_form_G(c, o, dip.z, [rows[j][2] for j in idx], xs, ys)
        E[:, :, idx, :] .= G[:, :, :, :, 3]
    end
    semi = builder(stack_v, dip, pz)
    Esa = [eval_semi_cartesian(semi, xs[i], ys[j], (k - 0.5) * pz)[c] for i in eachindex(xs), j in eachindex(ys), k in eachindex(rows), c in 1:3]
    zpos = src == 1 ? stack_v[1].d - dip.z : dip.z
    ζ = [(k - 0.5) * pz - (cum[src] + zpos) for k in eachindex(rows)]
    keep = [k for k in eachindex(rows) if abs(ζ[k]) > 0.3]
    near = [k for k in eachindex(rows) if 0.1 < abs(ζ[k]) <= 0.3]
    e = relerr(E[:, :, keep, :], Esa[:, :, keep, :])
    en = relerr(E[:, :, near, :], Esa[:, :, near, :])
    global worst_a = max(worst_a, e)
    @printf("   %-38s %d kx nodes (grid %.1fs, cache %.2fs, G %.1fs)  vs SA %.2e  (0.1<|ζ|≤0.3: %.1e)\n",
            label, length(kx), t_grid, t_cache, t_G, e, en)
end
passed["a"] = worst_a < 1e-3

# ── [b] linearity of the multi-source assembly ──
println("[b] closed_form_field: batched = Σ singles")
dips = [Dipole_constructor(:z, ω, 0.0, 0.6, 2), Dipole_constructor(:x, ω, 0.2, 0.6, 2; y=0.1),
        Dipole_constructor([0.3, -0.7im, 1.0], ω, -0.3, 1.3, 2; y=-0.2)]
zobs = Dict(1 => [1.5], 2 => [0.2, 1.0, 1.8], 3 => [0.4])
stack_gv = [lay(m_vac, 1.0), lay(m_gl, 2.0), lay(m_vac, 1.0)]
kx, wt, _ = point_kx_grid(stack_gv, ω, 2; ρ_max=1.4, pz=pz)
c1 = Dict(2 => build_cavity_transfer_1d(stack_gv, ω, 2, kx, wt))
kρ, wρ = kρ_grid(stack_gv; kmax=15.0); φs, wφ = uniform_φ(32)
stack_bi = [lay(m_vac, 1.0), lay(m_bi, 2.0), lay(m_vac, 1.0)]
c2 = Dict(2 => build_cavity_transfer_2d(stack_bi, ω, 2, kρ, wρ, φs, wφ))
worst_b = 0.0
for (lbl, caches) in (("1D, vac/glass/vac", c1), ("2D, biaxial", c2))
    Eb = closed_form_field(caches, grid, zobs, dips)
    Es = [closed_form_field(caches, grid, zobs, [d]) for d in dips]
    e = maximum(relerr(Eb[o], sum(E[o] for E in Es)) for o in keys(zobs))
    global worst_b = max(worst_b, e)
    @printf("   %-20s %.2e\n", lbl, e)
end
passed["b"] = worst_b < 1e-12

# ── [c] 2D rotation fill / full solve / 1D, NUFFT ──
println("[c] 2D rotation fill vs full solve vs 1D (same kρ nodes); NUFFT vs exact sum")
stack_u = [lay(m_vac, 1.0), lay(m_uni, 0.8), lay(m_gl, 0.5), lay(m_vac, 1.0)]
kρ, wρ = kρ_grid(stack_u); φs, wφ = uniform_φ(64)
g5 = uniform_kxky_grid(L=0.5, Δx=0.1); x5, y5 = grid_offsets(g5)
worst_c = 0.0; worst_nu = 0.0
for src in 1:3
    cr = build_cavity_transfer_2d(stack_u, ω, src, kρ, wρ, φs, wφ)
    cf = build_cavity_transfer_2d(stack_u, ω, src, kρ, wρ, φs, wφ; rotation=false)
    c1d = build_cavity_transfer_1d(stack_u, ω, src, kρ, wρ)
    zs = 0.4 * stack_u[src].d
    for obs in 1:4
        zo = obs == src ? [0.05 * stack_u[obs].d, 0.95 * stack_u[obs].d] : [0.3 * stack_u[obs].d]
        Gr = closed_form_G(cr, obs, zs, zo, x5, y5)
        e = max(relerr(Gr, closed_form_G(cf, obs, zs, zo, x5, y5)), relerr(Gr, closed_form_G(c1d, obs, zs, zo, x5, y5)))
        en = relerr(closed_form_G(cr, obs, zs, zo, x5, y5; lateral=:nufft), Gr)
        global worst_c = max(worst_c, e); global worst_nu = max(worst_nu, en)
    end
end
@printf("   rotation vs full vs 1D: worst %.2e | NUFFT vs exact: worst %.2e\n", worst_c, worst_nu)
passed["c"] = worst_c < 1e-10 && worst_nu < 1e-9

# ── [d] 90° rotation covariance ──
println("[d] 90° rotation covariance, full-solve branch")
xo = collect(-0.4:0.1:0.4); No = length(xo)
R = @SMatrix [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
φs, wφ = uniform_φ(48)
worst_d = 0.0
for (lbl, A, B) in (("biaxial diag(a,b,c) → diag(b,a,c)", m_bi, m_biT), ("gyrotropic (invariant)", m_gyr, m_gyr))
    stA = [lay(m_vac, 1.0), lay(A, 0.7), lay(m_gl, 0.5), lay(m_vac, 1.0)]
    stB = [lay(m_vac, 1.0), lay(B, 0.7), lay(m_gl, 0.5), lay(m_vac, 1.0)]
    local kρ, wρ = kρ_grid(stA; kmax=15.0)
    cA = build_cavity_transfer_2d(stA, ω, 2, kρ, wρ, φs, wφ)
    cB = build_cavity_transfer_2d(stB, ω, 2, kρ, wρ, φs, wφ)
    for (obs, zo) in ((2, 0.1), (2, 0.6), (1, 0.8), (3, 0.25))
        GA = closed_form_G(cA, obs, 0.35, [zo], xo, xo)
        GB = closed_form_G(cB, obs, 0.35, [zo], xo, xo)
        # G_B(x', y') at (x', y') = R(x, y) = (−y, x)  ⇔  GB[No+1−j, i] = R·GA[i, j]·Rᵀ
        err = maximum(norm(SMatrix{3,3}(GB[No + 1 - j, i, 1, :, :]) - R * SMatrix{3,3}(GA[i, j, 1, :, :]) * R') for i in 1:No, j in 1:No)
        e = err / maximum(norm(GA[i, j, 1, :, :]) for i in 1:No, j in 1:No)
        global worst_d = max(worst_d, e)
        @printf("   %-36s obs=%d z'=%.2f  %.2e\n", lbl, obs, zo, e)
    end
end
passed["d"] = worst_d < 1e-10

# ── [e] reciprocity ──
println("[e] reciprocity between internal layers 2 and 3")
worst_e = 0.0
for (lbl, mat, use2d) in (("isotropic (1D)", m_gl, false), ("biaxial (2D)", m_bi, true))
    st = [lay(m_vac, 1.0), lay(mat, 0.7), lay(m_gl, 0.6), lay(m_vac, 1.0)]
    local kρ, wρ = kρ_grid(st; kmax=15.0)
    c2_, c3_ = use2d ? (build_cavity_transfer_2d(st, ω, 2, kρ, wρ, φs, wφ), build_cavity_transfer_2d(st, ω, 3, kρ, wρ, φs, wφ)) :
                       (build_cavity_transfer_1d(st, ω, 2, kρ, wρ), build_cavity_transfer_1d(st, ω, 3, kρ, wρ))
    z2, z3 = 0.45, 0.2
    for (dx, dy) in ((0.0, 0.0), (0.3, 0.0), (0.2, -0.25))
        G23 = closed_form_G(c2_, 3, z2, [z3], [dx], [dy])[1, 1, 1, :, :]      # field in layer 3 from a source in layer 2
        G32 = closed_form_G(c3_, 2, z3, [z2], [-dx], [-dy])[1, 1, 1, :, :]    # field in layer 2 from a source in layer 3
        e = norm(G23 - transpose(G32)) / norm(G23)
        global worst_e = max(worst_e, e)
        @printf("   %-15s Δ=(%.2f,%.2f)  |G₂₃ − G₃₂ᵀ|/|G₂₃| = %.2e\n", lbl, dx, dy, e)
    end
end
passed["e"] = worst_e < 1e-10

println()
for k in sort(collect(keys(passed)))
    println("  [$k] ", passed[k] ? "pass" : "FAIL")
end
println(all(values(passed)) ? "POINT: PASS" : "POINT: FAIL")
