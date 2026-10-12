include("ClosedForm.jl")
using LinearAlgebra, StaticArrays, Printf

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Assembly_Validation.jl -- the FFT multi-source assembly
# (mod_ClosedFormAssembly.jl) against a direct, per-source reference:
#
#  [a] closed_form_field (zero-padded linear convolution) = Σ over dipoles of
#      closed_form_G evaluated at that dipole's own offsets x_i − x_dip
#      (lateral=:direct, exact sum), with OFF-CENTRE dipoles near the grid edges
#      at two depths and observation in every layer. 1D cache (point,
#      vac/glass/vac) and 2D cache (box, biaxial slab, NUFFT). Without padding,
#      offsets beyond ±L wrapped onto the opposite edge (O(1) errors there).
#  [b] precompute_ghats! then closed_form_field: the apply builds no new Ĝ and
#      gives the same field; reports precompute vs apply time.
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")
lay(m, d) = FlatOpticalLayer(m, d)
m_vac = IsotropicMaterial(1.0 + 0.001im)
m_gl  = IsotropicMaterial(2.25 + 0.001im)
m_bi  = BiaxialMaterial(2.25 + 0.01im, 4.6 + 0.08im, -1.0 + 0.3im)
passed = Dict{String,Bool}()

snap(grid, x, y) = (round(x / grid.Δx) * grid.Δx, round(y / grid.Δy) * grid.Δy)

function reference_field(cache, grid, zobs, dips; kw...)
    xs, ys = grid_offsets(grid)
    out = Dict(o => zeros(ComplexF64, grid.Nx, grid.Ny, length(zs), 3) for (o, zs) in zobs)
    for d in dips, (o, zs) in zobs
        xd, yd = snap(grid, d.x, d.y)
        kwd = cache isa CavityTransferCache2D ? (; kw..., lateral=:direct) : kw
        G = closed_form_G(cache, o, d.z, zs, xs .- xd, ys .- yd; kwd...)
        for q in 1:3
            out[o] .+= G[:, :, :, :, q] .* d.p[q]
        end
    end
    return out
end

s = 1 / 16
grid = uniform_kxky_grid(L=0.5, Δx=s)

# ── [a] padded assembly vs direct per-source reference ──
println("[a] closed_form_field (padded FFT) vs Σ per-dipole closed_form_G at shifted offsets")
stack_g = [lay(m_vac, 1.0), lay(m_gl, 0.5), lay(m_vac, 1.0)]
kx, wt, _ = point_kx_grid(stack_g, ω, 1; ρ_max=1.5, pz=0.1)
c1 = build_cavity_transfer_1d(stack_g, ω, 1, kx, wt)
stack_b = [lay(m_vac, 1.0), lay(m_bi, 0.5), lay(m_vac, 1.0)]
kρ, wρ = kρ_grid(stack_b; kmax=25.0); φs, wφ = uniform_φ(48)
c2 = build_cavity_transfer_2d(stack_b, ω, 1, kρ, wρ, φs, wφ)
zobs = Dict(1 => [0.1, 0.4, 0.8], 2 => [0.2], 3 => [0.3])
dips = [Dipole_constructor(:z, ω, 0.4375, 0.55, 1; y=-0.375),
        Dipole_constructor([1.0, 0.3im, -0.5], ω, -0.5, 0.55, 1; y=0.25),
        Dipole_constructor(:x, ω, 0.0625, 0.3, 1; y=0.4375)]
worst_a = 0.0
for (lbl, c, kw, tol) in (("1D point, vac/glass/vac", c1, (;), 1e-12),
                          ("2D box, biaxial, NUFFT", c2, (box_lateral=(s, s), Δz_box=s, lateral=:nufft, nufft_eps=1e-12), 1e-9))
    E = closed_form_field(Dict(1 => c), grid, zobs, dips; kw...)
    R = reference_field(c, grid, zobs, dips; kw...)
    e = maximum(relerr(E[o], R[o]) for o in keys(zobs))
    edge = maximum(relerr(E[o][1:2, :, :, :], R[o][1:2, :, :, :]) for o in keys(zobs))
    global worst_a = max(worst_a, e / tol)
    @printf("   %-26s all pixels %.2e   two left-edge columns %.2e   (tol %.0e)\n", lbl, e, edge, tol)
end
passed["a"] = worst_a < 1

# ── [b] precompute, then a grid-work-free apply ──
println("[b] precompute_ghats! then closed_form_field")
kw = (box_lateral=(s, s), Δz_box=s, lateral=:nufft)
gc = Dict()
t_pre = @elapsed precompute_ghats!(gc, Dict(1 => c2), grid, Dict(1 => unique(d.z for d in dips)), zobs; kw...)
n_pre = length(gc)
t_app = @elapsed E1 = closed_form_field(Dict(1 => c2), grid, zobs, dips; ghat_cache=gc, kw...)
t_app = @elapsed E1 = closed_form_field(Dict(1 => c2), grid, zobs, dips; ghat_cache=gc, kw...)
E0 = closed_form_field(Dict(1 => c2), grid, zobs, dips; kw...)
e = maximum(relerr(E1[o], E0[o]) for o in keys(zobs))
@printf("   %d Ĝ precomputed in %.2f s; apply %.4f s, Ĝ built during apply: %d, vs uncached %.1e\n",
        n_pre, t_pre, t_app, length(gc) - n_pre, e)
passed["b"] = length(gc) == n_pre && e < 1e-14

println()
for k in sort(collect(keys(passed)))
    println("  [$k] ", passed[k] ? "PASS" : "FAIL")
end
