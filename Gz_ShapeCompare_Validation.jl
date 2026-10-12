include("ClosedForm.jl")
using LinearAlgebra, Printf, Plots

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ShapeCompare_Validation.jl -- VISUAL comparison, point source vs box source,
# closed-form implementation only, plus precompute/apply timings.
#
# Rewritten 2026-10-12 from the 2026-09-23 version (archived in
# G_z-z_Box-Source/ShapeCompare_figs/), which also ran the legacy voxel path.
# The legacy modules are gone; the stacks, dipoles, output volume and scenarios
# are the same.
#
# Two operators per (stack, source layer), on the same output volume (64 px ×
# 60 rows, pitch 0.1 laterally and in z, the y = 0 row):
#   point -- CavityTransferCache1D on point_kx_grid, J0/J1/J2 path
#   box   -- CavityTransferCache2D on box_kρ_grid: S(kt)·D seed + interior slab
#            (face ramps, particular solution, local Ez), footprint
#            box_dx × box_dy, Δz = the row pitch, NUFFT lateral transform
# Expected: box ≈ point away from the source (the box is a point source plus
# O((kΔ)²) corrections, Δ the box size); near the source and in its own plane
# the point field is singular (its own plane is dropped, ζ = 0) and the box
# field is finite and smooth.
#
# Figures in ShapeCompare_figs/, per scenario:
#   *_maps.png        |Ex|²+|Ez|², point | box on one shared log scale, and
#                     the relative intensity difference |I_box − I_point|/I_point
#   *_slice.png       |Ez| along z at x = X_SLICE
#   *_nearsource.png  |Ez|² along z, 2 px beside the source, through its plane
# Timings: build (grid + cache), precompute (Ĝ per source depth, all rows),
# apply (FFT · multiply · iFFT only). A small warm-up runs first, so JIT
# compilation is not counted.
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")      # stack_rows, relerr

Εv = IsotropicMaterial(1.0 + 0.001im)
vv = FlatOpticalLayer(Εv, 2.0)
Εglass = IsotropicMaterial(2.25 + 0.001im)
vg = FlatOpticalLayer(Εglass, 2.0)

stack_vdip   = [vv, vv, vv]
stack_vglass = [vv, vg, vv]

x0 = 0.0
n  = 20                 # rows per layer
v  = 64                 # pixels along x
vy = 2                  # pixels along y (the y = 0 row is plotted)
px = 0.1                # lateral pitch
pz = stack_vdip[1].d / n
X_SLICE = 2.0

box_dx = 0.3
box_dy = 0.3
Δz_box = pz

# box quadrature knobs (box_kρ_grid)
BOX_RTOL  = 1e-3
BOX_KCAP  = 1500.0      # upper kρ bound (the interior tail is algebraic -- see STATUS, Newell subtraction)
BOX_NPHI  = 128
BOX_ρFAR  = 1.0         # far lateral probe (the interior needs it to resolve e^{i kρ ρ cosφ})

figdir = joinpath(@__DIR__, "ShapeCompare_figs")
mkpath(figdir)
println("shapes: point, box($box_dx × $box_dy × Δz=$Δz_box) -- grid px=$px, $v×$vy px, $(3n) rows of $pz; figures in $figdir/")

# Sources -- same as the previous versions of this file.
za = 1.95;  dipa = Dipole_constructor(:z, ω, x0, za, 1)
zb = 0.95;  dipb = Dipole_constructor(:z, ω, x0, zb, 1)
xc = 1.15; zc = 0.45; dipc = Dipole_constructor(:z, ω, xc, zc, 1)

zia = 0.85; dipia = Dipole_constructor(:z, ω, x0, zia, 2)
zib = 1.65; dipib = Dipole_constructor(:z, ω, x0, zib, 2)
xic = 0.75; zic = 1.15; dipic = Dipole_constructor(:z, ω, xic, zic, 2)

# ── Operator: cache + Ĝ cache on the output volume ──

struct ShapeOperator
    grid; caches; kw; zobs; rows; ghat_cache
end

function build_operator(stack, src; box::Bool, nv=v, nrows=n, quick=false)
    grid = uniform_kxky_grid(L=nv * px / 2, Δx=px, Ly=vy * px / 2, Δy=px)
    rows, _ = stack_rows(stack, stack[1].d / nrows)
    zobs = Dict(o => [z for (l, z) in rows if l == o] for o in 1:length(stack))
    ρ_max = 2 * hypot(grid.L, grid.Ly)               # largest source–observation offset on this grid
    if box
        kρ, wρ, φs, wφ, _ = box_kρ_grid(stack, ω, src; box=(box_dx, box_dy, Δz_box),
                                         N_phi=quick ? 8 : BOX_NPHI, rtol=quick ? 1e-1 : BOX_RTOL,
                                         kρ_cap=quick ? 50.0 : BOX_KCAP, ρ_far=BOX_ρFAR, maxevals=50_000)
        cache = build_cavity_transfer_2d(stack, ω, src, kρ, wρ, φs, wφ)
        kw = (box_lateral=(box_dx, box_dy), Δz_box=Δz_box, lateral=:nufft)
        nodes = "$(length(kρ)) kρ × $(length(φs)) φ"
    else
        kx, wt, _ = point_kx_grid(stack, ω, src; ρ_max=ρ_max, pz=stack[1].d / nrows, rtol=quick ? 1e-2 : 1e-6)
        cache = build_cavity_transfer_1d(stack, ω, src, kx, wt)
        kw = NamedTuple()
        nodes = "$(length(kx)) kx"
    end
    return ShapeOperator(grid, Dict(src => cache), kw, zobs, rows, Dict()), nodes
end

precompute!(op, src, depths) = precompute_ghats!(op.ghat_cache, op.caches, op.grid, Dict(src => depths), op.zobs; op.kw...)

"Field of one dipole on the y = 0 row: (x, row, component)."
function field(op, dip)
    parts = closed_form_field(op.caches, op.grid, op.zobs, [dip]; ghat_cache=op.ghat_cache, op.kw...)
    E = zeros(ComplexF64, op.grid.Nx, length(op.rows), 3)
    seen = Dict(o => 0 for o in keys(op.zobs))
    for (j, (l, _)) in enumerate(op.rows)
        seen[l] += 1
        E[:, j, :] .= parts[l][:, op.grid.midy, seen[l], :]
    end
    return E
end

# ── Build both operators for one (stack, source layer), timed ──

const NAMES = ("point", "box")
timings = []
function build_pair(stack, label, src, depths)
    ops = Dict{String,ShapeOperator}()
    t = Dict{String,NTuple{4,Float64}}()
    for name in NAMES
        tb = @elapsed op, nodes = build_operator(stack, src; box=(name == "box"))
        tp = @elapsed precompute!(op, src, depths)
        dip = Dipole_constructor(:z, ω, x0, depths[1], src)
        ta = @elapsed field(op, dip)
        ta = min(ta, @elapsed field(op, dip))
        t[name] = (tb, tp, tp / length(depths), ta)
        ops[name] = op
        @printf("  [%s] %-5s build %6.2fs | precompute %d depths %6.2fs (%.2fs each) | apply %6.3fs  (%s)\n",
                label, name, tb, length(depths), tp, tp / length(depths), ta, nodes)
    end
    push!(timings, (label, t))
    return ops
end

# ── Figures ──

intensity(E, rows) = abs2.(E[:, rows, 1]) .+ abs2.(E[:, rows, 3])

function compare_pair(stack, ops, dip, zrange, label_stack, label_scenario, fname)
    E = Dict(k => field(ops[k], dip) for k in NAMES)
    grid = ops["point"].grid
    xs, _ = grid_offsets(grid)
    rows = zrange[1]:zrange[2]
    zvals = (rows .- 0.5) .* pz
    lines = filter(b -> zvals[1] <= b <= zvals[end], cumsum((l -> l.d).(stack)))
    I = Dict(k => intensity(E[k], rows) for k in NAMES)
    lo = max(minimum(minimum(I[k]) for k in NAMES), 1e-9)
    hi = maximum(maximum(I[k]) for k in NAMES)

    function hmap(M, title; clims, color, cbar_title="")
        plt = heatmap(zvals, xs, M; clims=clims, xlabel="z (λ₀)", ylabel="x (λ₀)", title=title, titlefontsize=9,
                      xlims=(zvals[1], zvals[end]), color=color, colorbar_scale=:log10, colorbar_title=cbar_title)
        vline!(plt, lines; color=:white, label=nothing)
        hline!(plt, [dip.x]; ls=:dash, color=:white, label=nothing)
        return plt
    end
    maps = [hmap(max.(I[k], 1e-9), "$k — $label_stack"; clims=(lo, hi), color=:linear_kbgyw_5_98_c62_n256) for k in NAMES]
    rel = clamp.(abs.(I["box"] .- I["point"]) ./ max.(I["point"], 1e-300), 1e-4, 1e1)
    push!(maps, hmap(rel, "|I_box − I_point| / I_point"; clims=(1e-4, 1e1), color=:viridis))
    plot(maps..., layout=(1, 3), size=(1800, 520), plot_title=label_scenario, plot_titlefontsize=11,
         left_margin=5Plots.mm, bottom_margin=5Plots.mm)
    savefig(joinpath(figdir, "$(fname)_maps.png"))

    styles = Dict("point" => (:crimson, :solid), "box" => (:royalblue, :dash))
    ix = round(Int, X_SLICE / px) + grid.midx
    plt = plot(; title="$label_scenario — |Ez| at x = $(xs[ix])", xlabel="z (λ₀)", ylabel="|Ez|", legend=:best, titlefontsize=9)
    for k in NAMES
        plot!(plt, zvals, abs.(E[k][ix, rows, 3]); label=k, color=styles[k][1], ls=styles[k][2], lw=2)
    end
    vline!(plt, lines; color=:gray, label=nothing)
    savefig(plt, joinpath(figdir, "$(fname)_slice.png"))

    isrc = round(Int, dip.x / px) + grid.midx + 2
    plt = plot(; title="$label_scenario — |Ez|², 2 px beside the source (x = $(xs[isrc]))", xlabel="z (λ₀)",
               ylabel="|Ez|²", yscale=:log10, legend=:best, titlefontsize=9)
    for k in NAMES
        plot!(plt, zvals, max.(abs2.(E[k][isrc, rows, 3]), 1e-12); label=k, color=styles[k][1], ls=styles[k][2], lw=2)
    end
    vline!(plt, lines; color=:gray, label=nothing)
    savefig(plt, joinpath(figdir, "$(fname)_nearsource.png"))

    # box vs point away from the source (rows ≥ 5 px from its plane, |Δz| ≥ 0.5) and the near-source peaks
    zsrc = Dict(1 => stack[1].d - dip.z, 2 => stack[1].d + dip.z)[dip.layer]   # global depth (layer 1 is height above layer 2)
    far = [j for j in rows if abs((j - 0.5) * pz - zsrc) >= 0.5]
    dfar = isempty(far) ? NaN : relerr(E["box"][:, far, :], E["point"][:, far, :])
    peak(k) = maximum(abs2.(E[k][isrc, rows, 3]))
    @printf("  -- %-36s box vs point, |Δz| ≥ 0.5: %.2e | near-source peak |Ez|²: point %.2e, box %.2e\n",
            label_scenario, dfar, peak("point"), peak("box"))
end

# ── Warm-up (JIT): coarse grids, tiny volume, not timed ──
for box in (false, true)
    op, _ = build_operator(stack_vdip, 2; box=box, nv=8, nrows=4, quick=true)
    precompute!(op, 2, [1.0]); field(op, Dipole_constructor(:z, ω, 0.0, 1.0, 2))
end

# ──────────────────────────────────── IN HOMOGENEOUS MATERIALS (vacuum) ────────────────────────────────────
println("\n[VACUUM] source outside the stack (layer 1)")
ops = build_pair(stack_vdip, "vac ex", 1, [za, zb, zc])
for (dip, zr, lbl, f) in ((dipa, (3, 60), "Dip.za, vacuum, all layers", "vac_ex_a"),
                          (dipc, (3, 60), "Dip.zc, vacuum, all layers", "vac_ex_c"),
                          (dipc, (3, 20), "Dip.zc, vacuum, layer 1",    "vac_ex_lay1_c"),
                          (dipb, (21, 40), "Dip.zb, vacuum, layer 2",   "vac_ex_lay2_b"),
                          (dipb, (41, 60), "Dip.zb, vacuum, layer 3",   "vac_ex_lay3_b"))
    compare_pair(stack_vdip, ops, dip, zr, "vacuum", lbl, f)
end

println("\n[VACUUM] source inside the stack (layer 2)")
ops = build_pair(stack_vdip, "vac in", 2, [zia, zib, zic])
for (dip, zr, lbl, f) in ((dipia, (3, 60), "Dip.zia, vacuum, all layers", "vac_in_a"),
                          (dipic, (3, 60), "Dip.zic, vacuum, all layers", "vac_in_c"),
                          (dipic, (3, 20), "Dip.zic, vacuum, layer 1",    "vac_in_lay1_c"),
                          (dipib, (21, 40), "Dip.zib, vacuum, layer 2",   "vac_in_lay2_b"),
                          (dipib, (41, 60), "Dip.zib, vacuum, layer 3",   "vac_in_lay3_b"))
    compare_pair(stack_vdip, ops, dip, zr, "vacuum", lbl, f)
end

# ──────────────────────────────────── WITH A MATERIAL LAYER (vac-glass-vac) ────────────────────────────────────
println("\n[VAC-GLASS-VAC] source outside the stack (layer 1)")
ops = build_pair(stack_vglass, "vgv ex", 1, [za, zc])
for (dip, zr, lbl, f) in ((dipa, (5, 60), "Dip.za, vac-glass-vac, all layers", "vgv_ex_a"),
                          (dipa, (41, 60), "Dip.za, vac-glass-vac, layer 3",   "vgv_ex_lay3_a"),
                          (dipc, (41, 60), "Dip.zc, vac-glass-vac, layer 3",   "vgv_ex_lay3_c"))
    compare_pair(stack_vglass, ops, dip, zr, "vac-glass-vac", lbl, f)
end

println("\n[VAC-GLASS-VAC] source inside the stack (layer 2)")
ops = build_pair(stack_vglass, "vgv in", 2, [zia, zic])
for (dip, zr, lbl, f) in ((dipia, (1, 60), "Dip.zia, vac-glass-vac, all layers", "vgv_in_a"),
                          (dipia, (41, 60), "Dip.zia, vac-glass-vac, layer 3",   "vgv_in_lay3_a"),
                          (dipic, (41, 60), "Dip.zic, vac-glass-vac, layer 3",   "vgv_in_lay3_c"))
    compare_pair(stack_vglass, ops, dip, zr, "vac-glass-vac", lbl, f)
end

# ── Timing table ──
println("\nTimings (seconds; after warm-up). precompute = Ĝ for the listed source depths, all 60 rows;")
println("apply = one dipole, FFT · multiply · iFFT only (every Ĝ precomputed).")
@printf("%-8s | %-34s | %-34s\n", "config", NAMES...)
@printf("%-8s | %-34s | %-34s\n", "", ("build / precomp (per depth) / apply" for _ in 1:2)...)
for (label, t) in timings
    cells = [@sprintf("%6.2f / %6.2f (%5.2f) / %6.3f", t[k]...) for k in NAMES]
    @printf("%-8s | %-34s | %-34s\n", label, cells...)
end
println("\nDone -- see $figdir/*_maps.png, *_slice.png, *_nearsource.png.")
