# plane_wave_slice.jl -- one plane wave of fixed k_t falling from above on a three-film stack;
# field in the top vacuum layer on an x/z slice.
# Own solver: a `ModalSeed` at the top edge of layer 1 + `Modal_amp_profile` (src = obs = layer 1).
# Gila: the tutorial's Bloch-periodic 2×2 column, Bloch phases set by k_t, extended along x
# with the same Bloch phase. Both solve the same laterally infinite, periodic problem.
#
#   julia -t auto --project=comparisons/gila_airy comparisons/gila_airy/plane_wave_slice.jl [stack] [incidence] [cells per λ]
#
# stack: `base` (the three isotropic films of airy_comparison.jl) or `hyperbolic` (its middle film
# made uniaxial with εzz < 0). incidence: `oblique`, `normal`, `resonance` (evanescent k_t on the
# stack's strongest reflection pole, along the oblique direction) or `evanescent` (|k_t| = 2.5 k0).
#
# Physical frame: z up, the stack's top face at z = 0, unit incident amplitude at x = y = z = 0.
# An evanescent incident wave decays towards the stack (grows with z), like the near field of a
# source above it. The own solver's z runs top → bottom (layer 1 local depth from its top edge),
# so vectors go in with their z component negated and E_z comes back negated.
# The comparison is made on the scattered field E − E_inc (the stack's response); for evanescent
# incidence the incident part dominates the total and would hide it.

using GilaElectromagnetics, LinearAlgebra, StaticArrays, Printf, Random, Plots

const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "ClosedForm.jl"))
import .CavityTransfer: _mode_field_core, _mode_field_eps

const k0 = 2π
const ω = k0
const Δ = 1 // (length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 64)   # Gila cell (default λ/64, as in the tutorial)
const period = 2Δ
const H_TOP = 1//2                    # rendered top vacuum layer
const BOTTOM_CELLS = 4
const X_SPAN = 1.0                    # slice: x ∈ [−X_SPAN, X_SPAN)

# Films bottom → top as (εt, εz, d): diagonal, uniaxial about z (isotropic when εt = εz).
const STACKS = Dict(
    "base"       => [(2.25 + 0im, 2.25 + 0im, 1//4), (4.0 + 0im, 4.0 + 0im, 3//16), (2.0 + 0.1im, 2.0 + 0.1im, 1//8)],
    "hyperbolic" => [(2.25 + 0im, 2.25 + 0im, 1//4), (4.0 + 0.1im, -3.2 + 0.3im, 3//16), (2.0 + 0.1im, 2.0 + 0.1im, 1//8)],
)
const STACK_NAME = length(ARGS) >= 1 ? ARGS[1] : "base"
const INCIDENCE = length(ARGS) >= 2 ? ARGS[2] : "oblique"
haskey(STACKS, STACK_NAME) || error("unknown stack $STACK_NAME")
INCIDENCE in ("oblique", "normal", "resonance", "evanescent") || error("unknown incidence $INCIDENCE")
const FILMS = STACKS[STACK_NAME]
const T_STACK = sum(last, FILMS)
const DIRECTION = normalize([0.5, 0.3])                                 # in-plane direction of k_t

material(εt, εz) = εt == εz ? IsotropicMaterial(εt) : UniaxialMaterial(εt, εz)

# Own stack, top → bottom: layer 1 = the top vacuum (source and observation layer).
own_stack() = [FlatOpticalLayer(IsotropicMaterial(1.0), Float64(H_TOP));
               [FlatOpticalLayer(material(εt, εz), Float64(d)) for (εt, εz, d) in reverse(FILMS)];
               FlatOpticalLayer(IsotropicMaterial(1.0), BOTTOM_CELLS * Float64(Δ))]

# Polarization basis (ŝ, p̂) of a wave with in-plane k_t and vertical wavenumber kzv (z-up frame).
function sp_basis(kt, kzv)
    k = SVector{3,ComplexF64}(kt[1], kt[2], kzv)
    ŝ = iszero(norm(kt)) ? SVector{3,ComplexF64}(0, 1, 0) : SVector{3,ComplexF64}(-kt[2], kt[1], 0) / norm(kt)
    return ŝ, normalize(cross(ŝ, k))
end

# Physical reflection matrix of the stack at its top face, in the (s, p) basis of the incident
# (downward) and reflected (upward) waves, for |k_t| = q along DIRECTION. Layer 1 gets d = 0 so the
# S-matrix is referenced at the face (with d = H it would carry e^{2i kz H}, i.e. e^{−2κH} when evanescent).
function reflection_sp(q)
    stack = own_stack(); stack[1] = FlatOpticalLayer(IsotropicMaterial(1.0), 0.0)
    kt = q .* DIRECTION
    kzv = sqrt(complex(k0^2 - q^2))
    modes = Modes_Stack(stack, ω, kt)
    S21 = Matrix(S_all_Matrix(stack, modes)[1].SRight[SVector(3, 4), SVector(1, 2)])
    k_own = SVector{3,ComplexF64}(kt[1], kt[2], kzv)                     # downward = +z_own
    flip(v) = SVector(v[1], v[2], -v[3])
    in_basis, out_basis = sp_basis(kt, -kzv), sp_basis(kt, kzv)
    B = hcat((Vector(o[1:2]) for o in out_basis)...)                    # (Ex, Ey) of reflected s, p
    R = zeros(ComplexF64, 2, 2)
    for (j, e) in enumerate(in_basis)
        E = flip(e); Hm = cross(k_own, E) / ω
        ψr = modes[1].Vb * (S21 * (Matrix(modes[1].Vf) \ [E[1], E[2], Hm[1], Hm[2]]))
        R[:, j] = B \ ψr[1:2]
    end
    return R
end
reflection_gain(q) = opnorm(reflection_sp(q))

# The strongest reflection pole between k0 and 3 k0 (guided modes of the stack seen from above).
function resonance_kt()
    qs = range(1.0005k0, 3k0; length = 4000)
    g = [reflection_gain(q) for q in qs]
    i = argmax(g)
    lo, hi = qs[max(i - 1, 1)], qs[min(i + 1, end)]
    for _ in 1:60                                                       # golden-section refinement
        a, b = lo + 0.382(hi - lo), lo + 0.618(hi - lo)
        reflection_gain(a) > reflection_gain(b) ? (hi = b) : (lo = a)
    end
    q = (lo + hi) / 2
    return q, reflection_gain(q)
end

const KT, GAIN = let
    q = INCIDENCE == "oblique" ? k0 * norm([0.5, 0.3]) :
        INCIDENCE == "normal" ? 0.0 :
        INCIDENCE == "evanescent" ? 2.5k0 : resonance_kt()[1]
    Tuple(q .* DIRECTION), reflection_gain(q)
end
const EVANESCENT = sum(abs2, KT) > k0^2

const kz = sqrt(complex(k0^2 - sum(abs2, KT)))                         # iκ when evanescent
const k_inc = SVector{3,ComplexF64}(KT[1], KT[2], -kz)                  # downward / decaying downward
const e_inc = let rng = Xoshiro(2026)
    ŝ, p̂ = sp_basis(KT, -kz)
    a = normalize(randn(rng, ComplexF64, 2))
    a[1] * ŝ + a[2] * p̂                                                 # random s/p mix
end

const n_top = Int(H_TOP / Δ)
heights() = [(k - 0.5) * Float64(Δ) for k in 1:n_top]                  # cell centres above the stack
const XS = collect(-X_SPAN + Float64(Δ) / 2 : Float64(Δ) : X_SPAN)
const Y_SLICE = -Float64(Δ) / 2                                         # Gila's first y cell

incident_profile() = [e_inc[c] * cis(-kz * h) for h in heights(), c in 1:3]

# ─────────────────────────────────────────────────────────────────────────────
#  Own solver: plane wave = modal seed at the top edge of layer 1
# ─────────────────────────────────────────────────────────────────────────────

function own_profile()
    H = Float64(H_TOP)
    stack = own_stack()
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

slice(profile) = [profile[k, c] * cis(KT[1] * x + KT[2] * Y_SLICE) for x in XS, k in 1:n_top, c in 1:3]

# ─────────────────────────────────────────────────────────────────────────────
#  Gila: Bloch-periodic column of films + top vacuum; returns (total, scattered) slices
# ─────────────────────────────────────────────────────────────────────────────

function gila_slices()
    nz = BOTTOM_CELLS + Int(T_STACK / Δ) + n_top
    z_lo = -T_STACK - BOTTOM_CELLS * Δ
    volume = GlaVol((2, 2, nz), (Δ, Δ, Δ), (0//1, 0//1, z_lo + nz * Δ // 2))
    faces = cumsum([-T_STACK; [d for (_, _, d) in FILMS]])
    function χ_of(pos)
        z = pos[3]
        for (j, (εt, εz, _)) in enumerate(FILMS)
            faces[j] < z < faces[j+1] && return SMatrix{3,3,ComplexF64}(Diagonal(SVector(εt - 1, εt - 1, εz - 1)))
        end
        return zero(SMatrix{3,3,ComplexF64})
    end
    G₀ = VacuumGreenOperator{Float64}(volume; blc = (ax = (1, 2), P = (2, 2), θ = (-KT[1] * period, -KT[2] * period)))
    W = ScatteringOperator(G₀, χ_of)
    X = sus(W)
    f_inc = discretize!(zeros(ComplexF64, volume), r -> Tuple(e_inc .* cis(sum(k_inc .* Float64.(r)))))
    t = @elapsed (f_s = regionview(G₀ * (W * (X * f_inc)), 1))
    @info "Gila: solved" elapsed = t

    xs, ys, zs = (Float64.(g) for g in volume.grd)
    @assert ys[1] ≈ Y_SLICE
    rows = findall(>(0), zs)
    @assert zs[rows] ≈ heights()
    # extend the 2-cell column along x with its Bloch phase, p(r + L) = e^{i kx L} p(r)
    L = 2 * Float64(Δ)
    S = zeros(ComplexF64, length(XS), n_top, 3)
    for (i, x) in enumerate(XS)
        j = argmin(abs.(mod.(x .- xs .+ L / 2, L) .- L / 2))
        S[i, :, :] .= f_s[j, 1, rows, :] .* cis(KT[1] * (x - xs[j]))
    end
    return S .+ slice(incident_profile()), S
end

# ─────────────────────────────────────────────────────────────────────────────
#  Comparison and figure: z vertical, the slice standing on the drawn stack
# ─────────────────────────────────────────────────────────────────────────────

intensity(S) = dropdims(sum(abs2, S; dims = 3); dims = 3)
relL2(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
tag() = "$(STACK_NAME)_$(INCIDENCE)" * (Δ == 1//64 ? "" : "_$(denominator(Δ))")

function report(own, gila, own_s, gila_s)
    q = norm(KT) / k0
    @printf("\nstack %s, incidence %s: |k_t| = %.5f k0 along φ = %.1f°%s, ‖r_sp‖ = %.3g (largest s/p reflection at the face)\n", STACK_NAME, INCIDENCE, q,
            rad2deg(atan(DIRECTION[2], DIRECTION[1])), EVANESCENT ? " (evanescent)" : @sprintf(" (θ = %.1f°)", rad2deg(asin(q))), GAIN)
    @printf("e_inc (z-up) = [%s]\n", join((@sprintf("%.3f%+.3fi", real(z), imag(z)) for z in e_inc), ", "))
    @printf("top layer, own vs Gila, relative L2:  total %.3e   scattered %.3e\n", relL2(own, gila), relL2(own_s, gila_s))
    for (c, name) in enumerate(("Ex", "Ey", "Ez"))
        iszero(sum(abs2, gila_s[:, :, c])) ? @printf("  scattered %s: zero in both\n", name) :
            @printf("  scattered %s: %.3e\n", name, relL2(own_s[:, :, c], gila_s[:, :, c]))
    end
    @printf("|E_s| / |E_inc| at the stack face (own): %.3g\n",
            norm(own_s[1, 1, :]) / norm(slice(incident_profile())[1, 1, :]))
end

function draw_stack!(plt)
    xl = (first(XS) - Float64(Δ) / 2, last(XS) + Float64(Δ) / 2)
    show(z) = imag(z) == 0 ? string(real(z)) : string(z)
    z = 0.0
    for ((εt, εz, d), col) in zip(reverse(FILMS), (:goldenrod, :steelblue, :lightblue))   # top film first
        plot!(plt, Shape([xl[1], xl[2], xl[2], xl[1]], [z - d, z - d, z, z]); color = col, lw = 0, label = "")
        lab = εt == εz ? "ε = $(show(εt))" : "εt = $(show(εt)), εz = $(show(εz))"
        annotate!(plt, 0.0, z - d / 2, text("$lab,  d = $(Float64(d))λ", 7, :black))
        z -= Float64(d)
    end
    return plt
end

function panel(M; title, clims, cmap, cbtitle = "")
    plt = heatmap(XS, heights(), permutedims(M); c = cmap, clims, title, colorbar_title = cbtitle,
                  xlabel = "x (λ)", ylabel = "z (λ)", aspect_ratio = :equal, xticks = -1:0.5:1,
                  xlims = (first(XS), last(XS)), ylims = (-T_STACK - 0.02, Float64(H_TOP)))
    return draw_stack!(plt)
end

function figure(own, gila, own_s, gila_s)
    Io, Ig = intensity(own), intensity(gila)
    lg = EVANESCENT                                                     # incident field grows with z
    I(M) = lg ? log10.(M) : M
    Ilim = lg ? (minimum(I(Ig)), maximum(I(Ig))) : (0.0, maximum(Ig))
    Rlim = maximum(abs, real.(gila_s[:, :, 1])) .* (-1, 1)
    err = log10.(sqrt.(intensity(own_s .- gila_s)) ./ maximum(sqrt.(intensity(gila_s))))
    lab = lg ? "log₁₀ |E|²" : "|E|²"
    p = [panel(I(Io); title = "own $lab", clims = Ilim, cmap = :inferno),
         panel(I(Ig); title = "Gila $lab", clims = Ilim, cmap = :inferno),
         panel(err; title = "|E_s,own − E_s,Gila| / max|E_s|", clims = (-4, -1), cmap = :viridis, cbtitle = "log₁₀"),
         panel(real.(own_s[:, :, 1]); title = "own Re E_s,x (scattered)", clims = Rlim, cmap = :balance),
         panel(real.(gila_s[:, :, 1]); title = "Gila Re E_s,x (scattered)", clims = Rlim, cmap = :balance)]
    Is_o, Is_g = intensity(own_s)[1, :], intensity(gila_s)[1, :]
    prof = plot(Is_o, heights(); label = "own", xlabel = "|E_s|²", ylabel = "z (λ)", lw = 2, xscale = lg ? :log10 : :identity,
                ylims = (-T_STACK - 0.02, Float64(H_TOP)), title = "scattered |E_s|²(z)", legend = :bottomright)
    scatter!(prof, Is_g, heights(); label = "Gila", ms = 2.5, msw = 0)
    hline!(prof, [0.0]; color = :gray, ls = :dash, label = "stack top")
    q = norm(KT) / k0
    plt = plot(p..., prof; layout = (2, 3), size = (1500, 900), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm,
               plot_title = @sprintf("%s stack, %s incidence, |k_t| = %.4f k0, Gila λ/%d", STACK_NAME, INCIDENCE, q, denominator(Δ)))
    path = joinpath(mkpath(joinpath(@__DIR__, "results")), "plane_wave_$(tag()).png")
    savefig(plt, path)
    println("figure: ", path)
end

own_s = slice(own_profile() .- incident_profile())
own = own_s .+ slice(incident_profile())
gila, gila_s = gila_slices()
report(own, gila, own_s, gila_s)
figure(own, gila, own_s, gila_s)
