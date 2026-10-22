# plane_wave_slice.jl -- one plane wave of fixed k_t falling from above on the three-film stack
# of `airy_comparison.jl`; total field in the top vacuum layer on an x/z slice.
# Own solver: a `ModalSeed` at the top edge of layer 1 + `Modal_amp_profile` (src = obs = layer 1).
# Gila: the tutorial's Bloch-periodic 2×2 column, Bloch phases set by k_t, extended along x
# with the same Bloch phase. Both solve the same laterally infinite, periodic problem.
#
#   julia -t auto --project=comparisons/gila_airy comparisons/gila_airy/plane_wave_slice.jl
#
# Physical frame: z up, the stack's top face at z = 0, unit incident amplitude at x = y = z = 0.
# The own solver's z runs top → bottom (layer 1 local depth from its top edge), so vectors go in
# with their z component negated and E_z comes back negated.

using GilaElectromagnetics, LinearAlgebra, StaticArrays, Printf, Random, Plots

const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "ClosedForm.jl"))
import .CavityTransfer: _mode_field_core, _mode_field_eps

const k0 = 2π
const ω = k0
const Δ = 1//64                       # Gila cell, as in the tutorial
const period = 2Δ
const H_TOP = 1//2                    # rendered top vacuum layer
const BOTTOM_CELLS = 4
const X_SPAN = 1.0                    # slice: x ∈ [−X_SPAN, X_SPAN)

# Fixed in-plane wavevector (oblique, off the xz plane so both Bloch phases matter).
const KT = k0 .* (0.5, 0.3)

# Bottom → top, as in airy_comparison.jl: the lossy film faces the incoming wave.
const FILMS = [(1.25 + 0im, 1//4), (3.0 + 0im, 3//16), (1.0 + 0.1im, 1//8)]
const T_STACK = sum(last, FILMS)

const kz = sqrt(k0^2 - sum(abs2, KT))
const k_inc = SVector(KT[1], KT[2], -kz)                                # downward
const e_inc = let rng = Xoshiro(2026)
    ŝ = normalize(SVector(-KT[2], KT[1], 0.0))
    p̂ = normalize(cross(ŝ, k_inc))
    a = normalize(randn(rng, ComplexF64, 2))
    a[1] * ŝ + a[2] * p̂                                                 # random s/p mix, |e| = 1
end

const n_top = Int(H_TOP / Δ)
heights() = [(k - 0.5) * Float64(Δ) for k in 1:n_top]                  # cell centres above the stack
const XS = collect(-X_SPAN + Float64(Δ) / 2 : Float64(Δ) : X_SPAN)
const Y_SLICE = -Float64(Δ) / 2                                         # Gila's first y cell

# ─────────────────────────────────────────────────────────────────────────────
#  Own solver: plane wave = modal seed at the top edge of layer 1
# ─────────────────────────────────────────────────────────────────────────────

function own_profile()
    H = Float64(H_TOP)
    stack = [FlatOpticalLayer(IsotropicMaterial(1.0), H);
             [FlatOpticalLayer(IsotropicMaterial(1 + χ), Float64(d)) for (χ, d) in reverse(FILMS)];
             FlatOpticalLayer(IsotropicMaterial(1.0), BOTTOM_CELLS * Float64(Δ))]
    modes = Modes_Stack(stack, ω, [KT[1], KT[2]])
    Sint, Sprop = S_all_ingredients(stack, modes)
    Sall = S_all_Matrix(stack, modes, Sint, Sprop)

    # incident wave in the own frame, its tangential (Ex, Ey, Hx, Hy) at the top edge (h = H)
    k_own = SVector(k_inc[1], k_inc[2], -k_inc[3])
    E_own = SVector(e_inc[1], e_inc[2], -e_inc[3]) * cis(-kz * H)
    H_own = cross(k_own, E_own) / ω
    a = Matrix(modes[1].Vf) \ [E_own[1], E_own[2], H_own[1], H_own[2]]
    seed = ModalSeed(1, H, a, zeros(ComplexF64, 2))                     # layer 1: z = height ⇒ zpos = 0

    depth = [H - h for h in heights()]
    amps = Modal_amp_profile(stack, [seed], 1, modes, Sall, Sint, depth)
    ϵzt, ϵzz = _mode_field_eps(stack[1])
    kvec = SVector{2,ComplexF64}(KT...)
    E = zeros(ComplexF64, n_top, 3)
    for (i, (z, A)) in enumerate(zip(depth, amps)), m in 1:2
        exf, eyf, ezf, qf = _mode_field_core(ϵzt, ϵzz, modes[1], ω, kvec, :f, m)
        exb, eyb, ezb, qb = _mode_field_core(ϵzt, ϵzz, modes[1], ω, kvec, :b, m)
        φf, φb = A.Ef[m] * cis(qf * z), A.Eb[m] * cis(qb * (z - H))
        E[i, :] .+= φf .* SVector(exf, eyf, -ezf) .+ φb .* SVector(exb, eyb, -ezb)   # back to z-up
    end
    return E                                                            # (height, component) at x = y = 0
end

own_slice(profile) = [profile[k, c] * cis(KT[1] * x + KT[2] * Y_SLICE) for x in XS, k in 1:n_top, c in 1:3]

# ─────────────────────────────────────────────────────────────────────────────
#  Gila: Bloch-periodic column of films + top vacuum
# ─────────────────────────────────────────────────────────────────────────────

function gila_slice()
    nz = BOTTOM_CELLS + Int(T_STACK / Δ) + n_top
    z_lo = -T_STACK - BOTTOM_CELLS * Δ
    volume = GlaVol((2, 2, nz), (Δ, Δ, Δ), (0//1, 0//1, z_lo + nz * Δ // 2))
    faces = cumsum([-T_STACK; [d for (_, d) in FILMS]])
    function χ_of(pos)
        z = pos[3]
        for (j, (χ, _)) in enumerate(FILMS)
            faces[j] < z < faces[j+1] && return ComplexF64(χ)
        end
        return 0.0 + 0im
    end
    G₀ = VacuumGreenOperator{Float64}(volume; blc = (ax = (1, 2), P = (2, 2), θ = (-KT[1] * period, -KT[2] * period)))
    W = ScatteringOperator(G₀, χ_of)
    X = sus(W)
    f_inc = discretize!(zeros(ComplexF64, volume), r -> Tuple(e_inc .* cis(sum(k_inc .* Float64.(r)))))
    p_s = W * (X * f_inc)
    f = regionview(f_inc, 1) .+ regionview(G₀ * p_s, 1)

    xs, ys, zs = (Float64.(g) for g in volume.grd)
    @assert ys[1] ≈ Y_SLICE
    rows = findall(>(0), zs)
    @assert zs[rows] ≈ heights()
    # extend the 2-cell column along x with its Bloch phase, p(r + L) = e^{i kx L} p(r)
    L = 2 * Float64(Δ)
    S = zeros(ComplexF64, length(XS), n_top, 3)
    for (i, x) in enumerate(XS)
        j = argmin(abs.(mod.(x .- xs .+ L / 2, L) .- L / 2))
        S[i, :, :] .= f[j, 1, rows, :] .* cis(KT[1] * (x - xs[j]))
    end
    return S
end

# ─────────────────────────────────────────────────────────────────────────────
#  Comparison and figure: z vertical, the slice standing on the drawn stack
# ─────────────────────────────────────────────────────────────────────────────

intensity(S) = dropdims(sum(abs2, S; dims = 3); dims = 3)

function report(own, gila)
    rel = sqrt(sum(abs2, own .- gila) / sum(abs2, gila))
    pw = maximum(sqrt.(intensity(own .- gila))) / maximum(sqrt.(intensity(gila)))
    θ = rad2deg(asin(norm(KT) / k0))
    @printf("\nk_t = 2π·(%.2f, %.2f)  (θ = %.1f°, φ = %.1f°)\n", KT[1] / k0, KT[2] / k0, θ, rad2deg(atan(KT[2], KT[1])))
    @printf("e_inc (z-up) = [%s]\n", join((@sprintf("%.3f%+.3fi", real(z), imag(z)) for z in e_inc), ", "))
    @printf("top layer, own vs Gila: relative L2 %.3e, max |ΔE| / max |E| %.3e\n", rel, pw)
    for (c, name) in enumerate(("Ex", "Ey", "Ez"))
        @printf("  %s: relative L2 %.3e\n", name, sqrt(sum(abs2, (own .- gila)[:, :, c]) / sum(abs2, gila[:, :, c])))
    end
end

function draw_stack!(plt)
    xl = (first(XS) - Float64(Δ) / 2, last(XS) + Float64(Δ) / 2)
    z = 0.0
    for ((χ, d), col) in zip(reverse(FILMS), (:goldenrod, :steelblue, :lightblue))   # top film first
        plot!(plt, Shape([xl[1], xl[2], xl[2], xl[1]], [z - d, z - d, z, z]); color = col, lw = 0, label = "")
        annotate!(plt, 0.0, z - d / 2, text("χ = $(imag(χ) == 0 ? real(χ) : χ),  d = $(Float64(d))λ", 7, :black))
        z -= Float64(d)
    end
    return plt
end

function panel(M; title, clims, cmap, cbtitle)
    plt = heatmap(XS, heights(), permutedims(M); c = cmap, clims, title, colorbar_title = cbtitle,
                  xlabel = "x (λ)", ylabel = "z (λ)", aspect_ratio = :equal, xticks = -1:0.5:1,
                  xlims = (first(XS), last(XS)), ylims = (-T_STACK - 0.02, Float64(H_TOP)))
    return draw_stack!(plt)
end

function figure(own, gila)
    Io, Ig = intensity(own), intensity(gila)
    Ilim = (0.0, maximum(Ig))
    Rlim = maximum(abs, real.(gila[:, :, 1])) .* (-1, 1)
    err = log10.(sqrt.(intensity(own .- gila)) ./ maximum(sqrt.(Ig)))
    p = [panel(Io; title = "own |E|²", clims = Ilim, cmap = :inferno, cbtitle = ""),
         panel(Ig; title = "Gila |E|²", clims = Ilim, cmap = :inferno, cbtitle = ""),
         panel(err; title = "|E_own − E_Gila| / max|E|", clims = (-4, -1), cmap = :viridis, cbtitle = "log₁₀"),
         panel(real.(own[:, :, 1]); title = "own Re Ex", clims = Rlim, cmap = :balance, cbtitle = ""),
         panel(real.(gila[:, :, 1]); title = "Gila Re Ex", clims = Rlim, cmap = :balance, cbtitle = "")]
    prof = plot(Io[1, :], heights(); label = "own", xlabel = "|E|²", ylabel = "z (λ)", lw = 2,
                ylims = (-T_STACK - 0.02, Float64(H_TOP)), title = "|E|²(z), x-independent", legend = :bottomright)
    scatter!(prof, Ig[1, :], heights(); label = "Gila", ms = 2.5, msw = 0)
    hline!(prof, [0.0]; color = :gray, ls = :dash, label = "stack top")
    plt = plot(p..., prof; layout = (2, 3), size = (1500, 900), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm)
    path = joinpath(mkpath(joinpath(@__DIR__, "results")), "plane_wave_slice_stack.png")
    savefig(plt, path)
    println("figure: ", path)
end

own = own_slice(own_profile())
gila = gila_slice()
report(own, gila)
figure(own, gila)
