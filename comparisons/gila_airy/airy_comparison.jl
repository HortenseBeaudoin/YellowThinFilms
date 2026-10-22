# airy_comparison.jl -- plane-wave reflectance of layered films: Gila (Bloch-periodic unit
# cell) vs our S-matrix core (ModeSolver + Redheffer) vs an independent transfer-matrix
# (Airy) reference. Follows Gila's tutorial 07_thin_film:
#   https://moleskysean.github.io/GilaElectromagnetics.jl/stable/tutorials/07_thin_film/
#
#   julia --project=comparisons/gila_airy comparisons/gila_airy/airy_comparison.jl [case ...]
#
# Cases: `film` (the tutorial: χ = 1.25, λ/2) and `stack` (three films in vacuum, one lossy).
# This folder has its own environment because the Bloch lattice (`blc`, `setBlc!`) only
# exists in the moleskysean fork of Gila, not in the version pinned in the main Manifest.
# Units: λ = 1, so ω = k0 = 2π. Layers are listed bottom to top; the wave comes from below.

using GilaElectromagnetics, LinearAlgebra, StaticArrays, Printf, Plots

const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "mod_ModeSolver.jl"))
include(joinpath(ROOT, "mod_ScatteringMatrix.jl"))
using .ModeSolver, .ScatteringMatrix

const k0 = 2π
const Δ = 1//64                 # Gila cell, as in the tutorial
const period = 2Δ               # narrowest Bloch cell Gila allows
const vacuum_cells = 4

# A film is (χ, thickness in λ); thicknesses must be multiples of Δ so faces sit on cell faces.
const CASES = Dict(
    "film"  => [(1.25 + 0im, 1//2)],
    "stack" => [(1.25 + 0im, 1//4), (3.0 + 0im, 3//16), (1.0 + 0.1im, 1//8)],
)

# ─────────────────────────────────────────────────────────────────────────────
#  Independent reference: characteristic (transfer) matrix, vacuum on both sides
# ─────────────────────────────────────────────────────────────────────────────

function tmm_reflectance(films, θ_inc, polarization)
    kx = k0 * sind(θ_inc)
    kz0 = k0 * cosd(θ_inc)
    η(ε, kz) = polarization == :s ? kz / k0 : ε * k0 / kz
    η0 = η(1.0, kz0)
    M = Matrix{ComplexF64}(I, 2, 2)
    for (χ, d) in films
        ε = 1 + χ
        kz = sqrt(complex(ε * k0^2 - kx^2))
        δ, ηj = kz * d, η(ε, kz)
        M = M * [cos(δ) -im * sin(δ) / ηj; -im * ηj * sin(δ) cos(δ)]   # e^{-iωt}: loss is Im ε > 0
    end
    num = η0 * M[1, 1] + η0^2 * M[1, 2] - M[2, 1] - η0 * M[2, 2]
    den = η0 * M[1, 1] + η0^2 * M[1, 2] + M[2, 1] + η0 * M[2, 2]
    return abs2(num / den)
end

# The tutorial's closed Airy formula, kept to check the TMM on the single film.
function airy_film(χ, thickness, θ_inc, polarization)
    n = sqrt(1 + χ)
    cosθ_t = sqrt(1 - (sind(θ_inc) / n)^2)
    r₁₂ = polarization == :s ? (cosd(θ_inc) - n * cosθ_t) / (cosd(θ_inc) + n * cosθ_t) :
                               (n * cosd(θ_inc) - cosθ_t) / (n * cosd(θ_inc) + cosθ_t)
    round_trip = cis(4π * n * cosθ_t * thickness)
    return abs2(r₁₂ * (1 - round_trip) / (1 - r₁₂^2 * round_trip))
end

# ─────────────────────────────────────────────────────────────────────────────
#  Ours: S_21 of the whole stack, in the eigenmode basis of the incidence layer
# ─────────────────────────────────────────────────────────────────────────────

function own_stack(films)
    vac = FlatOpticalLayer(IsotropicMaterial(1.0), 0.0)
    return [vac; [FlatOpticalLayer(IsotropicMaterial(1 + χ), Float64(d)) for (χ, d) in films]; vac]
end

function own_reflectance(stack, θ_inc, polarization)
    kx = k0 * sind(θ_inc)
    modes = Modes_Stack(stack, k0, kx)
    S = S_all_Matrix(stack, modes)[1].SRight          # whole stack, seen from layer 1
    # Incident tangential field (Ex, Ey, Hx, Hy) of a unit-|E| plane wave, H = k×E/ω.
    ψ_inc = polarization == :s ? SVector(0.0, 1.0, -cosd(θ_inc), 0.0) :
                                 SVector(cosd(θ_inc), 0.0, 0.0, 1.0)
    a = Matrix(modes[1].Vf) \ Vector(ψ_inc)
    ψ_r = modes[1].Vb * (S[SVector(3, 4), SVector(1, 2)] * a)
    # Reflected wave in vacuum: |E|² from its tangential part (Ez = ±Ex tanθ for p).
    return abs2(ψ_r[2]) + abs2(ψ_r[1]) / cosd(θ_inc)^2
end

# ─────────────────────────────────────────────────────────────────────────────
#  Gila: one Bloch-periodic 2×2 column of the stack, read in the bottom vacuum cell
# ─────────────────────────────────────────────────────────────────────────────

function gila_setup(films)
    total = sum(last, films)
    nz = Int(total / Δ) + 2vacuum_cells
    volume = GlaVol((2, 2, nz), (Δ, Δ, Δ))
    faces = cumsum([-total / 2; [d for (_, d) in films]])       # bottom face of each film, then top
    function χ_of(pos)
        z = pos[3]
        for (j, (χ, _)) in enumerate(films)
            faces[j] < z < faces[j+1] && return ComplexF64(χ)
        end
        return 0.0 + 0im
    end
    G₀ = VacuumGreenOperator(volume; blc = (ax = (1, 2), P = (2, 2), θ = (0.0, 0.0)))
    W = ScatteringOperator(G₀, χ_of)
    return (; volume, G₀, W, X = sus(W))
end

function gila_reflectance(g, θ_inc, polarization)
    k = k0 .* (sind(θ_inc), 0, cosd(θ_inc))
    e = polarization == :s ? (0, 1, 0) : (cosd(θ_inc), 0, -sind(θ_inc))
    setBlc!(g.W, (-k[1] * period, 0.0))                   # Gila: p(r + L) = e^{-iθ} p(r)
    f_inc = discretize!(zeros(g.volume), r -> e .* cis(sum(k .* r)))
    f_s = g.G₀ * (g.W * (g.X * f_inc))
    return sum(abs2, regionview(f_s, 1)[1, 1, 1, :])
end

# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────

function run_case(name; angles = 0:2.5:85)
    films = CASES[name]
    println("\n== case `$name`: films (χ, d/λ) = ", films)
    stack = own_stack(films)
    g = gila_setup(films)
    outdir = mkpath(joinpath(@__DIR__, "results"))
    plt = plot(xlabel = "angle of incidence (°)", ylabel = "|r|²", title = "$name, Δ = λ/64",
               legend = :topleft)
    open(joinpath(outdir, "$name.tsv"), "w") do io
        println(io, "pol\tangle\tR_tmm\tR_own\tR_gila")
        for (pol, color) in ((:s, 1), (:p, 2))
            R_tmm = [tmm_reflectance(films, θ, pol) for θ in angles]
            R_own = [own_reflectance(stack, θ, pol) for θ in angles]
            t = @elapsed R_gila = [gila_reflectance(g, θ, pol) for θ in angles]
            for i in eachindex(angles)
                @printf(io, "%s\t%.2f\t%.10f\t%.10f\t%.10f\n", pol, angles[i], R_tmm[i], R_own[i], R_gila[i])
            end
            own_err, gila_err = maximum(abs, R_own .- R_tmm), abs.(R_gila .- R_tmm)
            igila = argmax(gila_err)
            @printf("  %s: max |own − tmm| = %.2e   max |gila − tmm| = %.2e at %.1f°   R(0°) tmm/own/gila = %.5f / %.5f / %.5f   (Gila %.1f s)\n",
                    pol, own_err, gila_err[igila], angles[igila], R_tmm[1], R_own[1], R_gila[1], t)
            if length(films) == 1
                χ, d = only(films)
                @printf("     tmm vs closed Airy formula: %.2e\n",
                        maximum(abs(tmm_reflectance(films, θ, pol) - airy_film(χ, d, θ, pol)) for θ in angles))
            end
            fine = 0:0.25:last(angles)
            plot!(plt, fine, [tmm_reflectance(films, θ, pol) for θ in fine]; color, label = "TMM, $pol")
            scatter!(plt, angles, R_own; color, marker = :circle, ms = 3, label = "own, $pol")
            scatter!(plt, angles, R_gila; color, marker = :xcross, ms = 4, label = "Gila, $pol")
        end
    end
    savefig(plt, joinpath(outdir, "$name.png"))
end

for name in (isempty(ARGS) ? ["film", "stack"] : ARGS)
    run_case(name)
end
