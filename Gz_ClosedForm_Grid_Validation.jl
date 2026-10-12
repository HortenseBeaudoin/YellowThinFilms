include("ClosedForm.jl")
using LinearAlgebra, Printf, FastGaussQuadrature

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Grid_Validation.jl -- the adaptive grids of mod_ClosedFormGrid.jl
# (plan §4.2, Phase 8), validated by CONVERGENCE against dense fixed rules.
#
#  [P] point_kx_grid (1D, point source): field on rows away from the source vs
#      a dense fixed GL rule, at rtol 1e-4 / 1e-6
#  [B] box_kρ_grid (2D, box 0.1×0.1×0.1 in a glass layer, fixed φ grid shared by
#      every rule so only kρ is compared), per row group:
#        self slab    -- z' inside the box's own slab (algebraic tail only)
#        neighbour    -- the voxel directly above/below (h = Δz/2)
#        across iface -- just across each interface
#      vs a dense GL reference to K_ref (evaluated in chunks), at rtol 1e-2/-3/-4;
#      compared with the point-probed axis (point_kx_grid nodes reused as kρ) and
#      with box_kρ_grid without its interior probes.
# The self-slab tail is algebraic: the field is converged TO THE TAIL BOUND, not
# absolutely -- the reference itself moves by e_ref between K_ref/2 and K_ref.
# That is plan §4.2's optional quasi-static (Newell) subtraction, not done yet.
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")
m_vac = IsotropicMaterial(1.0 + 0.001im)
m_gl  = IsotropicMaterial(2.25 + 0.05im)   # guided-mode poles ~0.1 wide: resolvable by the fixed references
stack = [FlatOpticalLayer(m_vac, 1.0), FlatOpticalLayer(m_gl, 0.6), FlatOpticalLayer(m_vac, 1.0)]
src, zbar = 2, 0.3
grid = uniform_kxky_grid(L=0.5, Δx=0.1)
xs, ys = grid_offsets(grid)

function dense_grid(Kmax)
    brk = sort(unique(vcat(0.0, branch_points(stack, ω), 12.0, 60.0, Kmax)))
    x, w = gausslegendre(8)
    ks = Float64[]; ws = Float64[]
    for i in 1:(length(brk) - 1)
        a, b = brk[i], brk[i + 1]
        width = b <= 12 ? 0.01 : (b <= 60 ? 0.25 : 1.0)
        npan = ceil(Int, (b - a) / width)
        for j in 0:(npan - 1)
            lo = a + j * (b - a) / npan; hi = lo + (b - a) / npan
            append!(ks, (hi - lo) / 2 .* x .+ (hi + lo) / 2); append!(ws, (hi - lo) / 2 .* w)
        end
    end
    return ks, ws
end
passed = Dict{String,Bool}()

# ── [P] point_kx_grid ──
println("[P] point_kx_grid vs dense fixed rule (rows ≥ 0.1 from the source)")
prow = [(1, 0.9), (2, 0.05), (2, 0.45), (3, 0.3)]         # source at z = 0.3 in layer 2
function pfield(kx, wt)
    c = build_cavity_transfer_1d(stack, ω, src, kx, wt)
    [closed_form_G(c, o, zbar, [z], xs, ys) for (o, z) in prow]
end
kd, wd = dense_grid(800.0)
Pref = pfield(kd, wd)
okP = true
for rtol in (1e-4, 1e-6)
    t = @elapsed kx, wt, info = point_kx_grid(stack, ω, src; ρ_max=0.71, pz=0.1, rtol=rtol)
    e = maximum(relerr(F, R) for (F, R) in zip(pfield(kx, wt), Pref))
    global okP &= e <= 10rtol
    @printf("   rtol=%.0e: %5d kx nodes (%.1fs)  worst vs dense (%d nodes) %.2e\n", rtol, length(kx), t, length(kd), e)
end
passed["P"] = okP

# ── [B] box_kρ_grid ──
box = (0.1, 0.1, 0.1)
N_phi = 96
φs, wφ = build_kphi_grid(N_phi)
rows = Dict("self slab"    => [(2, zbar), (2, zbar + 0.03), (2, zbar - 0.045)],
            "neighbour"    => [(2, zbar + 0.1), (2, zbar - 0.1)],
            "across iface" => [(1, 0.95), (3, 0.05)])
groups = ["self slab", "neighbour", "across iface"]
function fields(kρ, wρ; chunk=1500)
    out = Dict(g => [zeros(ComplexF64, length(xs), length(ys), 1, 3, 3) for _ in rows[g]] for g in groups)
    for i0 in 1:chunk:length(kρ)
        idx = i0:min(i0 + chunk - 1, length(kρ))
        c = build_cavity_transfer_2d(stack, ω, src, kρ[idx], wρ[idx], φs, wφ)
        for g in groups, (r, (obs, zo)) in enumerate(rows[g])
            out[g][r] .+= closed_form_G(c, obs, zbar, [zo], xs, ys; box_lateral=box[1:2], Δz_box=box[3])
        end
    end
    return out
end
grouperr(F, R) = Dict(g => maximum(relerr(F[g][r], R[g][r]) for r in eachindex(rows[g])) for g in groups)

K_ref = 6000.0
kr, wr = dense_grid(K_ref)
t_ref = @elapsed R = fields(kr, wr)
e_ref = grouperr(fields(dense_grid(K_ref / 2)...), R)
@printf("[B] reference: %d kρ × %d φ up to %.0f (%.0fs); cut at %.0f vs full: self slab %.1e | neighbour %.1e | iface %.1e\n",
        length(kr), N_phi, K_ref, t_ref, K_ref / 2, e_ref["self slab"], e_ref["neighbour"], e_ref["across iface"])
function report(label, kρ, wρ, t)
    E = grouperr(fields(kρ, wρ), R)
    @printf("   %-42s %6d kρ (max %5.0f, %5.1fs)  self slab %.2e | neighbour %.2e | iface %.2e\n",
            label, length(kρ), maximum(kρ), t, E["self slab"], E["neighbour"], E["across iface"])
    return E
end
box_E = Dict{Float64,Any}()
for rtol in (1e-2, 1e-3, 1e-4)
    t = @elapsed kρ, wρ, _, _, info = box_kρ_grid(stack, ω, src; box=box, zbar=[zbar], N_phi=N_phi, rtol=rtol,
                                                  kρ_cap=K_ref, maxevals=50_000, ρ_far=hypot(0.5, 0.5))
    box_E[rtol] = report(@sprintf("box_kρ_grid, rtol=%.0e", rtol), kρ, wρ, t)
end
t = @elapsed kρ, wρ, _, _, _ = box_kρ_grid(stack, ω, src; box=box, zbar=[zbar], N_phi=N_phi, rtol=1e-3, include_interior=false)
report("box_kρ_grid, rtol=1e-3, exterior probes only", kρ, wρ, t)
t = @elapsed kx, wt, _ = point_kx_grid(stack, ω, src; ρ_max=0.71, pz=0.1, rtol=1e-4)
pt_E = report("point_kx_grid nodes as kρ axis, rtol=1e-4", kx, wt, t)

ok_ext  = all(box_E[r]["neighbour"] <= 10r && box_E[r]["across iface"] <= 10r for r in keys(box_E))
ok_slab = box_E[1e-4]["self slab"] <= 1e-3
passed["B"] = ok_ext && ok_slab && box_E[1e-4]["self slab"] < pt_E["self slab"] / 10
@printf("   neighbour/iface within 10×rtol: %s | self slab ≤ 1e-3 at rtol=1e-4: %s | box ≫ point-probed on the self slab: %s\n",
        ok_ext, ok_slab, box_E[1e-4]["self slab"] < pt_E["self slab"] / 10)
@printf("   (absolute self-slab tail: the reference moves by %.1e between kρ=%.0f and %.0f -- Newell subtraction pending)\n",
        e_ref["self slab"], K_ref / 2, K_ref)

println()
println(all(values(passed)) ? "GRID: PASS (box: to the tail bound)" : "GRID: FAIL")
