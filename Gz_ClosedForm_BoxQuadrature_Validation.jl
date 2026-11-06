include(joinpath(@__DIR__, "ClosedForm.jl"))
using LinearAlgebra, Printf, FastGaussQuadrature

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_BoxQuadrature_Validation.jl -- box_kρ_grid on the stacks the
# feature-driven quadrature was made for (2026-11-06): hyperbolic, metallic,
# gyrotropic and in-plane hyperbolic layers. Small cases (λ = 1, ω = 2π), so the
# dense references fit in memory.
#
#  [A] box 0.1³ inside a hyperbolic slab (ε = (2, 2, −3.2) + 0.02i) on glass
#  [B] box in air 0.1 above a hyperbolic slab on a metal (ε = −10 + i): every
#      polariton pole far past the light lines (the old "tail" case)
#  [C] box inside a gyrotropic, hyperbolic slab (ε_xy = 0.8i, ε_z = −2): full φ grid
#      and box_kρ_grid_grouped (one φ per group)
#  [D] box inside an in-plane hyperbolic biaxial slab (ε = (−3, 4, 2)):
#      box_kρ_grid_grouped (mirror classes)
#
# Each: box_kρ_grid at rtol 1e-3 and 1e-4 vs a dense Gauss–Legendre reference on the
# SAME φ grid and the SAME cut-off (kρ_cap = 150), per row group:
#   self slab    -- observation inside the box's own z-slab (algebraic tail: compared
#                   to the cut, as in the Grid suite)
#   neighbour    -- the voxels directly above/below (h = Δz/2)
#   across iface -- just across the source layer's interfaces
# Reference: panels 0.004 to 40, 0.02 to 150, geometric grading onto the half-space
# light lines with the u² substitution on the touching panels (KxGrid.gl_rule);
# its own convergence is checked by halving every panel. PASS: every row group within
# 10 × rtol, and no stall (box_kρ_grid raises on a stalled band).
# Run from the repository root: julia --project -t 8 Gz_ClosedForm_BoxQuadrature_Validation.jl
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")
const KCAP = 150.0
const BOX = (0.1, 0.1, 0.1)
grid = uniform_kxky_grid(L=0.3, Δx=0.1)
xs, ys = grid_offsets(grid)
const ρFAR = hypot(0.3, 0.3)
groups = ["self slab", "neighbour", "across iface"]
results = Tuple{String,Float64,Float64}[]
report(lbl, err, tol) = (push!(results, (lbl, err, tol));
                         @printf("  %-60s %.2e  (tol %.0e)  %s\n", lbl, err, tol, err <= tol ? "PASS" : "FAIL"))

function reference_rule(Ls; halve=false)
    f = halve ? 0.5 : 1.0
    br = Float64[0.0]
    for L in Ls, j in 0:20, sg in (-1, 1)
        push!(br, L * (1 + sg * 0.5 * 2.0^-j))
    end
    append!(br, Ls); push!(br, 40.0, KCAP)
    br = sort(unique(filter(b -> 0 <= b <= KCAP, br)))
    panels = NTuple{2,Float64}[]
    for i in 1:length(br)-1
        a, b = br[i], br[i+1]
        w = f * (b <= 40 ? 0.004 : 0.02)
        n = max(1, ceil(Int, (b - a) / w))
        for j in 0:n-1; push!(panels, (a + j * (b - a) / n, a + (j + 1) * (b - a) / n)); end
    end
    return gl_rule(panels; singular=Ls)
end

# Σ over kρ chunks and over φ groups (each group's own φ nodes): the 2D sum is linear in both
function fields(st, src, zbar, rows, rules)            # rules: [(kρ, wρ, φ, wφ), …]
    out = Dict(g => [zeros(ComplexF64, length(xs), length(ys), 1, 3, 3) for _ in rows[g]] for g in groups)
    for (kρ, wρ, φ, wφ) in rules
        step = max(1, 1_200_000 ÷ length(φ))
        for r in Iterators.partition(eachindex(kρ), step)
            c = build_cavity_transfer_2d(st, ω, src, kρ[r], wρ[r], φ, wφ)
            for g in groups, (i, (obs, zo)) in enumerate(rows[g])
                out[g][i] .+= closed_form_G(c, obs, zbar, [zo], xs, ys; box_lateral=BOX[1:2], Δz_box=BOX[3])
            end
        end
        GC.gc(false)
    end
    return out
end
grouperr(F, R; rows) = Dict(g => maximum(relerr(F[g][r], R[g][r]) for r in eachindex(rows[g])) for g in groups)

function run_case(name, st, src, zbar, rows, Ls; N_phi, grouped=false, also_full=true)
    println("\n", name)
    φs, wφ = build_kphi_grid(N_phi)
    kr, wr = reference_rule(Ls)
    t = @elapsed R = fields(st, src, zbar, rows, [(kr, wr, φs, wφ)])
    kr2, wr2 = reference_rule(Ls; halve=true)
    eR = grouperr(fields(st, src, zbar, rows, [(kr2, wr2, φs, wφ)]), R; rows=rows)
    @printf("  reference: %d kρ × %d φ to %.0f (%.0f s); halved: self %.1e | neighbour %.1e | iface %.1e\n",
            length(kr), N_phi, KCAP, t, eR["self slab"], eR["neighbour"], eR["across iface"])
    kw = (; box=BOX, zbar=[zbar], N_phi=N_phi, kρ_cap=KCAP, maxevals=50_000, ρ_far=ρFAR)
    variants = Tuple{String,Function}[]
    also_full && push!(variants, ("box_kρ_grid", rtol -> begin
        kρ, wρ, φ, wφ_, info = box_kρ_grid(st, ω, src; kw..., rtol=rtol)
        ([(kρ, wρ, φ, wφ_)], length(kρ), length(info.features.poles))
    end))
    grouped && push!(variants, ("box_kρ_grid_grouped", rtol -> begin
        gs, ginfo = box_kρ_grid_grouped(st, ω, src; kw..., rtol=rtol)
        ([(g.kρ, g.wρ, g.φ, g.wφ) for g in gs], round(Int, ginfo.n_nodes / N_phi), sum(length(i.features.poles) for i in ginfo.per_group))
    end))
    for (vname, build) in variants, rtol in (1e-3, 1e-4)
        local rules, nk, np
        t = @elapsed try
            rules, nk, np = build(rtol)
        catch e
            report(@sprintf("%s rtol=%.0e: built without stalling", vname, rtol), Inf, 0.0)
            println("    ", sprint(showerror, e)[1:min(end, 300)])
            continue
        end
        E = grouperr(fields(st, src, zbar, rows, rules), R; rows=rows)
        @printf("  %s rtol=%.0e: %d kρ (mean per φ), %d poles found, %.1f s\n", vname, rtol, nk, np, t)
        for g in groups
            report(@sprintf("%s rtol=%.0e: %s", vname, rtol, g), E[g], max(10rtol, 10eR[g]))
        end
    end
end

vac  = IsotropicMaterial(1.0 + 0.0im)
glass = IsotropicMaterial(2.25 + 0.01im)
hyp  = UniaxialMaterial(2.0 + 0.02im, -3.2 + 0.02im)
L_vac, L_gl = ω, ω * real(sqrt(2.25 + 0.01im))

# [A] box inside a hyperbolic slab on glass
stA = [FlatOpticalLayer(vac, 1.0), FlatOpticalLayer(hyp, 0.4), FlatOpticalLayer(glass, 1.0)]
rowsA = Dict("self slab" => [(2, 0.2), (2, 0.23), (2, 0.155)], "neighbour" => [(2, 0.3), (2, 0.1)],
             "across iface" => [(1, 0.95), (3, 0.05)])
run_case("[A] box 0.1³ inside a hyperbolic slab (ε_t = 2, ε_z = −3.2, +0.02i) 0.4 on glass", stA, 2, 0.2, rowsA,
         [L_vac, L_gl]; N_phi=32)

# [B] box in air above a hyperbolic slab on a metal (layer 1: caller depth = height above the interface)
stB = [FlatOpticalLayer(vac, 1.0), FlatOpticalLayer(hyp, 0.2), FlatOpticalLayer(IsotropicMaterial(-10.0 + 1.0im), 1.0)]
zB = 0.15; zpB = 1.0 - zB                                   # obs rows in layer 1's own frame
rowsB = Dict("self slab" => [(1, zpB), (1, zpB + 0.03), (1, zpB - 0.045)], "neighbour" => [(1, zpB - 0.1), (1, zpB + 0.1)],
             "across iface" => [(2, 0.05)])
run_case("[B] box in air, near face 0.1 above a hyperbolic slab 0.2 on a metal (ε = −10 + i)", stB, 1, zB, rowsB,
         [L_vac]; N_phi=32)

# [C] box inside a gyrotropic, hyperbolic slab
gyr = GyrotropicMaterial(ComplexF64[2.1+0.05im 0.8im 0; -0.8im 2.1+0.05im 0; 0 0 -2.0+0.05im])
stC = [FlatOpticalLayer(vac, 1.0), FlatOpticalLayer(gyr, 0.3), FlatOpticalLayer(vac, 1.0)]
rowsC = Dict("self slab" => [(2, 0.15), (2, 0.18), (2, 0.105)], "neighbour" => [(2, 0.25), (2, 0.05)],
             "across iface" => [(1, 0.95), (3, 0.05)])
run_case("[C] box inside a gyrotropic hyperbolic slab 0.3 (ε_xy = 0.8i, ε_z = −2), in air", stC, 2, 0.15, rowsC,
         [L_vac]; N_phi=16, grouped=true)

# [D] box inside an in-plane hyperbolic biaxial slab, grouped
bi = BiaxialMaterial(-3.0 + 0.05im, 4.0 + 0.05im, 2.0 + 0.05im)
stD = [FlatOpticalLayer(vac, 1.0), FlatOpticalLayer(bi, 0.2), FlatOpticalLayer(glass, 1.0)]
# the slab is two boxes thick, so the "neighbour" rows are its two faces (h = Δz/2)
rowsD = Dict("self slab" => [(2, 0.1), (2, 0.13), (2, 0.055)], "neighbour" => [(2, 1e-9), (2, 0.2 - 1e-9)],
             "across iface" => [(1, 0.95), (3, 0.05)])
run_case("[D] box inside an in-plane hyperbolic slab 0.2 (ε = (−3, 4, 2) + 0.05i) on glass", stD, 2, 0.1, rowsD,
         [L_vac, L_gl]; N_phi=32, grouped=true, also_full=false)

nfail = count(r -> !(r[2] <= r[3]), results)
println("\nBOX QUADRATURE: ", nfail == 0 ? "PASS" : "FAIL ($nfail of $(length(results)))")
