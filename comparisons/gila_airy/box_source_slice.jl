# box_source_slice.jl -- one box source above the three-film stack of `airy_comparison.jl`,
# field in the top vacuum layer on the x/z plane through the source: own closed-form solver
# (src_layer = obs_layer = 1) vs Gila, with one of two lateral boundaries:
#   absorber (default): finite grid, absorbing ramp in the films near the lateral edges;
#   bloch: Bloch-periodic grid (period = the grid, phase BLOCH_PHASE), no absorber. Gila then
#          solves a periodic array of sources, while the own field is the single source's.
#
#   julia -t auto --project=comparisons/gila_airy comparisons/gila_airy/box_source_slice.jl [absorber|bloch]
#
# Physical frame: z up, the stack's top face at z = 0, the source above it at x = y = s/2. Gila uses this frame
# directly. The own solver's z runs top → bottom (layer 1 = top vacuum, local depth from its top),
# so its moment goes in with pz negated and its Ez comes back negated.
# Source: Gila's uniform current over one s×s×s voxel = the own `s×s×s` box, same cell.

using GilaElectromagnetics, LinearAlgebra, StaticArrays, Printf, Random, Plots

const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "ClosedForm.jl"))

const ω = 2π
const s = 1//32                       # voxel = box size, both solvers
const H_TOP = 1//2                    # rendered top vacuum layer
const SRC_ROW = 8                     # source cell = 8th row above the stack's top face
const BOTTOM_CELLS = 4                # vacuum kept under the stack on Gila's grid
const GILA_BC = isempty(ARGS) ? "absorber" : ARGS[1]
GILA_BC in ("absorber", "bloch") || error("boundary must be absorber or bloch, got $GILA_BC")
const ABSORBER = (width_wl = 0.25, rel_imag = 1.0, power = 2)   # Gila films only, as in RESULTS_gila_anisotropic_comparison.md
const BLOCH_PHASE = (0.0, 0.0)        # per period along x, y (Gila: p(r + L) = e^{-iθ} p(r))
# Lateral cells (Gila wants even counts). The Bloch period is the grid, and at phase 0 a 2λ period
# (64 cells) puts the (0, ±2) diffraction order exactly on the light circle (a Rayleigh–Wood
# anomaly, which Gila refuses): 60 cells = 1.875λ has no order there.
const N_LAT = GILA_BC == "bloch" ? 60 : 64
const MID = N_LAT ÷ 2 + 1             # source column: the cell centred at x = y = s/2
const OWN_OPTS = (rtol = 1e-3, kρ_cap = 3000.0, N_phi = 256, maxevals = 50_000, lateral = :nufft,
                  near_rows = 6, N_phi_near = 8192, phi_chunk = 256)    # old pipeline's defaults, but N_phi_near
                                                                         # 2048 aliases the source row beyond ~0.5λ

# Bottom → top, as in airy_comparison.jl: the lossy film faces the source.
const FILMS = [(1.25 + 0im, 1//4), (3.0 + 0im, 3//16), (1.0 + 0.1im, 1//8)]
const T_STACK = sum(last, FILMS)

const p_phys = normalize(randn(Xoshiro(2026), ComplexF64, 3))           # random moment, z-up frame

const n_top = Int(H_TOP / s)
heights() = [(k - 0.5) * Float64(s) for k in 1:n_top]                  # cell centres above the stack
const z_src = heights()[SRC_ROW]

# ─────────────────────────────────────────────────────────────────────────────
#  Own solver: layer 1 → layer 1, box on the 2D cache
# ─────────────────────────────────────────────────────────────────────────────

function own_slice()
    sf = Float64(s)
    stack = [FlatOpticalLayer(IsotropicMaterial(1.0), Float64(H_TOP));
             [FlatOpticalLayer(IsotropicMaterial(1 + χ), Float64(d)) for (χ, d) in reverse(FILMS)];
             FlatOpticalLayer(IsotropicMaterial(1.0), BOTTOM_CELLS * sf)]
    p_own = SVector(p_phys[1], p_phys[2], -p_phys[3])
    dip = Dipole_constructor(p_own, ω, 0.0, z_src, 1; y = 0.0)          # layer 1: height above interface
    depth = [Float64(H_TOP) - h for h in heights()]                      # layer 1 obs: depth from its top
    grid = uniform_kxky_grid(L = MID * sf, Δx = sf)
    w = (1:N_LAT) .+ (grid.midx - MID)

    t0 = time()
    kρ, wρ, φs, wφ, info = box_kρ_grid(stack, ω, 1; box = (sf, sf, sf), zbar = [z_src], N_phi = OWN_OPTS.N_phi,
                                       rtol = OWN_OPTS.rtol, kρ_cap = OWN_OPTS.kρ_cap, ρ_far = N_LAT * sf / 2,
                                       maxevals = OWN_OPTS.maxevals)
    @info "own: box grid" n_kρ = length(kρ) n_φ = length(φs) elapsed = time() - t0
    kw = (box_lateral = (sf, sf), Δz_box = sf, lateral = OWN_OPTS.lateral)
    field(φ, wφ, z) = closed_form_field(Dict(1 => build_cavity_transfer_2d(stack, ω, 1, kρ, wρ, φ, wφ; obs_layers = [1])),
                                        grid, Dict(1 => z), [dip]; kw...)[1]

    # rows near the source need the fine φ grid (see G_z-z_Box-Source/mod_OwnSolverRun.jl), in chunks
    near = [k for k in 1:n_top if abs(k - SRC_ROW) <= OWN_OPTS.near_rows]   # `depth` is in height order
    far = setdiff(1:n_top, near)
    E = zeros(ComplexF64, grid.Nx, grid.Ny, n_top, 3)
    E[:, :, far, :] .= field(φs, wφ, depth[far])
    φf, wφf = build_kphi_grid(OWN_OPTS.N_phi_near)
    for r in Iterators.partition(eachindex(φf), OWN_OPTS.phi_chunk)
        E[:, :, near, :] .+= field(φf[r], wφf[r], depth[near])
    end
    @info "own: field" elapsed = time() - t0

    # x/z slice through the source, z-up components
    S = E[w, grid.midy, :, :]
    S[:, :, 3] .*= -1
    return S                                                             # (x, height, component)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Gila: finite grid of the films + top vacuum, absorber ramp at the lateral edges
# ─────────────────────────────────────────────────────────────────────────────

function gila_slice()
    nz = BOTTOM_CELLS + Int(T_STACK / s) + n_top
    z_lo = -T_STACK - BOTTOM_CELLS * s
    volume = GlaVol((N_LAT, N_LAT, nz), (s, s, s), (0//1, 0//1, z_lo + nz * s // 2))
    faces = cumsum([-T_STACK; [d for (_, d) in FILMS]])
    edge = N_LAT * Float64(s) / 2
    function χ_of(pos)
        x, y, z = Float64.(pos)
        j = findfirst(i -> faces[i] < z < faces[i+1], eachindex(FILMS))
        j === nothing && return 0.0 + 0im
        χ = ComplexF64(FILMS[j][1])
        d = edge - max(abs(x), abs(y))
        (GILA_BC == "absorber" && d < ABSORBER.width_wl) || return χ
        return χ + im * ABSORBER.rel_imag * abs(real(1 + χ)) * ((ABSORBER.width_wl - d) / ABSORBER.width_wl)^ABSORBER.power
    end

    t0 = time()
    G₀ = GILA_BC == "bloch" ?
        VacuumGreenOperator{Float64}(volume; blc = (ax = (1, 2), P = (N_LAT, N_LAT), θ = BLOCH_PHASE)) :
        VacuumGreenOperator{Float64}(volume)
    W = ScatteringOperator(G₀, χ_of; slv = GCRODRSolver(nothing, 5000, nothing, 1e-6))
    X = sus(W)
    zs = Float64.(volume.grd[3])
    iz = argmin(abs.(zs .- z_src))
    J = zeros(ComplexF64, volume)
    regionview(J, 1)[MID, MID, iz, :] .= p_phys ./ Float64(s)^3       # one voxel of uniform current
    f_inc = G₀ * J
    p_d = X * f_inc
    p_s = W * p_d
    res = norm(p_d.dat .- (p_s.dat .- (X * (G₀ * p_s)).dat)) / norm(p_d.dat)
    f = regionview(f_inc, 1) .+ regionview(G₀ * p_s, 1)
    @info "Gila: solved" relative_residual = res elapsed = time() - t0
    rows = findall(z -> z > 0, zs)
    @assert zs[rows] ≈ heights()
    return ComplexF64.(f[:, MID, rows, :])
end

# ─────────────────────────────────────────────────────────────────────────────
#  Comparison and figure: z vertical, the slice standing on the drawn stack
# ─────────────────────────────────────────────────────────────────────────────

intensity(S) = dropdims(sum(abs2, S; dims = 3); dims = 3)

function report(own, gila)
    xs = ((1:N_LAT) .- (N_LAT + 1) / 2) .* Float64(s)
    interior = GILA_BC == "absorber" ? abs.(xs) .< N_LAT * Float64(s) / 2 - ABSORBER.width_wl : trues(N_LAT)
    off_row = trues(N_LAT, n_top); off_row[:, SRC_ROW] .= false
    src_row = .!off_row; src_row[MID, SRC_ROW] = false
    near = falses(N_LAT, n_top)
    for i in 1:N_LAT, k in 1:n_top
        near[i, k] = abs(i - MID) <= 6 && abs(k - SRC_ROW) <= 6
    end
    relL2(m) = sqrt(sum(abs2, (own .- gila)[m, :]) / sum(abs2, gila[m, :]))
    @printf("\nmoment (z-up) p = [%s]\n", join((@sprintf("%.3f%+.3fi", real(z), imag(z)) for z in p_phys), ", "))
    @printf("relative L2, own vs Gila, top layer:\n")
    @printf("  off the source row                %.3e\n", relL2(off_row))
    @printf("    interior (no absorber band)     %.3e\n", relL2(off_row .& interior))
    @printf("    near source (±6 cells)          %.3e\n", relL2(off_row .& near))
    @printf("    away from source, interior      %.3e\n", relL2(off_row .& .!near .& interior))
    # The box's own z-slab: the in-plane longitudinal component converges only algebraically
    # in kρ (no quasi-static subtraction yet), so this row is ~20–30% off even far out.
    @printf("  source row (source voxel excluded) %.3e\n", relL2(src_row))
    println("  off the source row, by lateral distance from the source:")
    for (lo, hi) in ((0, 4), (5, 10), (11, 16), (17, 24), (25, 32))
        cols = falses(N_LAT, n_top); cols[[i for i in 1:N_LAT if lo <= abs(i - MID) <= hi], :] .= true
        @printf("    %4.2f–%4.2fλ  %.3e\n", lo * Float64(s), hi * Float64(s), relL2(off_row .& cols))
    end
end

function draw!(plt, xs, zs, M; clims, title, cmap = :inferno, cbtitle = "log₁₀ |E|²")
    heatmap!(plt, xs, zs, permutedims(M); clims, c = cmap, title, colorbar_title = cbtitle)
    hw, xl = Float64(s) / 2, extrema(xs) .+ (-Float64(s) / 2, Float64(s) / 2)
    z = 0.0
    for ((χ, d), col) in zip(reverse(FILMS), (:goldenrod, :steelblue, :lightblue))   # top film first
        plot!(plt, Shape([xl[1], xl[2], xl[2], xl[1]], [z - d, z - d, z, z]); color = col, lw = 0, label = "")
        annotate!(plt, 0.0, z - d / 2, text("χ = $(χ == real(χ) ? real(χ) : χ),  d = $(Float64(d))λ", 7, :black))
        z -= Float64(d)
    end
    plot!(plt, [xl...], [0.0, 0.0]; color = :white, lw = 1, label = "")
    scatter!(plt, [xs[MID]], [z_src]; marker = :square, ms = 3, mc = :cyan, msw = 0, label = "")
    a = N_LAT * Float64(s) / 2 - ABSORBER.width_wl
    GILA_BC == "absorber" && vline!(plt, [-a, a]; ls = :dash, color = :gray, label = "")
    return plt
end

function figure(own, gila)
    xs = ((1:N_LAT) .- (N_LAT + 1) / 2) .* Float64(s)
    zs = heights()
    Io, Ig = log10.(intensity(own)), log10.(intensity(gila))
    lo, hi = quantile_(vcat(vec(Io), vec(Ig)), 0.01), maximum(Ig[:, setdiff(1:n_top, SRC_ROW)])
    err = log10.(sqrt.(intensity(own .- gila) ./ intensity(gila)))
    common = (xlabel = "x (λ)", ylabel = "z (λ)", aspect_ratio = :equal, xticks = -1:0.5:1,
              xlims = extrema(xs), ylims = (-T_STACK - 0.02, Float64(H_TOP)))
    p1 = draw!(plot(; common...), xs, zs, Io; clims = (lo, hi), title = "own (closed form)")
    p2 = draw!(plot(; common...), xs, zs, Ig; clims = (lo, hi), title = "Gila ($GILA_BC)")
    p3 = draw!(plot(; common...), xs, zs, err; clims = (-3, 0), title = "|E_own − E_Gila| / |E_Gila|",
               cmap = :viridis, cbtitle = "log₁₀")
    plt = plot(p1, p2, p3; layout = (1, 3), size = (1500, 420), left_margin = 4Plots.mm, bottom_margin = 6Plots.mm)
    path = joinpath(mkpath(joinpath(@__DIR__, "results")), "box_source_slice_stack_$GILA_BC.png")
    savefig(plt, path)
    println("figure: ", path)
end

quantile_(v, q) = sort(v)[max(1, round(Int, q * length(v)))]

own = own_slice()
gila = gila_slice()
report(own, gila)
figure(own, gila)
