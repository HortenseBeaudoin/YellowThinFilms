# diagnostics/zrank_study.jl -- how low is the z-rank of the non-Toeplitz blocks, and
# where does the rank above 4 come from? (pre-study for step 3, low-rank z skeletons)
#
# Per quadrature node, every scattered block (cross layer, and the same-layer reflected
# part) factors through the 4 edge amplitudes of the source layer
#   [sf∘e^{iQf(d−z)}; sb∘e^{−iQb z}]   →   T^o   →   e^{iQ^o z'}
# so it has z-rank ≤ 4. The operator works per lateral pixel frequency k_m instead, where
# Ĝ(k_m) sums many nodes (window spread + aliases k_m + 2πn/Δ, each with its own Q):
# the rank grows. This script measures by how much, on the dense blocks of
# mod_GreenOperator (no approximation in them), per k_m:
#   M(k_m) = D[k_m, r, c, i, j] reshaped to (3·No) × (3·Ns)   (rows (r, j), columns (c, i))
#
#   - rank at tol τ: # σ > τ·σ₁,max (σ₁,max = largest σ₁ over all k_m: an absolute
#     error relative to the block's largest entry), max and mean over k_m
#   - e_R = sqrt(Σ_k Σ_{r>R} σ²) / sqrt(Σ_k Σ σ²): relative Frobenius error of the best
#     rank-R factorisation per k_m (= the relative field error for a random source)
#   - σ₅/σ₁ per k_m (the best ANY 4-amplitude edge state can do), by |k_m| band
#   - where the rank-4 residual lives: per observation plane j and per source plane i
#   - shared source side: all scattered blocks stacked ([D₁; D₂,scat; D₃]), i.e. one V
#     for every observation layer, as the edge picture predicts
#
# Box source in layer 2 (0.5 thick), cubic voxels Δ = 1/32 (16 planes per layer),
# window 1λ × 1λ (32² px), max-precision quadrature (kρ_cap 3000, 256 φ, nufft_eps 1e-12).
#
#   julia -t 8 --project diagnostics/zrank_study.jl
#   ZRANK_STACKS=iso,biax julia -t 8 --project diagnostics/zrank_study.jl
include(joinpath(@__DIR__, "..", "ClosedForm.jl"))
using Printf, LinearAlgebra, Statistics
const GO = GreenOperator
ω = 2π
lay(m, d) = FlatOpticalLayer(m, d)
m_vac(δ) = IsotropicMaterial(complex(1.0, δ))
m_gl(δ)  = IsotropicMaterial(complex(2.25, δ))
m_bi(δ)  = BiaxialMaterial(complex(2.25, δ), complex(4.6, δ), complex(2.0, δ))
m_hyp(δ) = BiaxialMaterial(complex(2.25, δ), complex(4.6, δ), complex(-3.2, δ))
stacks = [
    ("iso", "vac/glass/vac",       1e-3, δ -> [lay(m_vac(δ), 0.5), lay(m_gl(δ), 0.5),  lay(m_vac(δ), 0.5)]),
    ("biax", "vac/biax(+2)/vac",   1e-3, δ -> [lay(m_vac(δ), 0.5), lay(m_bi(δ), 0.5),  lay(m_vac(δ), 0.5)]),
    ("hyp", "vac/biax(−3.2)/vac",  1e-1, δ -> [lay(m_vac(δ), 0.5), lay(m_hyp(δ), 0.5), lay(m_vac(δ), 0.5)]),
]
let keep = split(get(ENV, "ZRANK_STACKS", "iso,biax,hyp"), ",")
    filter!(s -> s[1] in keep, stacks)
end

Δ = parse(Float64, get(ENV, "ZRANK_DELTA", "0.03125"))
L = parse(Float64, get(ENV, "ZRANK_L", "0.5"))            # half-width of the lateral window
src = 2
planes(d, Δ) = collect(((1:round(Int, d / Δ)) .- 0.5) .* Δ)
kw = (box_lateral=(Δ, Δ), Δz_box=Δ, lateral=:nufft, nufft_eps=1e-12)
NODE_BUDGET = 900_000
τs = (1e-3, 1e-6, 1e-9)
Rs = (4, 6, 8, 12, 16)

"Dense blocks (no z structure): cross layers with terms=:all, the source layer with terms=:scattered."
function dense_blocks(st, groups, g, zs, zobs)
    D = Dict{Int,Array{ComplexF64,6}}()
    for gr in groups
        chunk = clamp(NODE_BUDGET ÷ max(length(gr.kρ), 1), 1, length(gr.φ))
        for r in Iterators.partition(eachindex(gr.φ), chunk)
            c = build_cavity_transfer_2d(st, ω, src, gr.kρ, gr.wρ, gr.φ[r], gr.wφ[r]; obs_layers=sort(collect(keys(zobs))))
            for (o, zo) in zobs
                Do = GO._dense_planes(c, o, zs, zo, g, length(zs), length(zo); terms=(o == src ? :scattered : :all), kw...)
                haskey(D, o) ? (D[o] .+= Do) : (D[o] = Do)
            end
            c = nothing; GC.gc()
        end
    end
    return D
end

"(3·No) × (3·Ns) matrix of one block at lateral index (x, y): rows (r, j), columns (c, i)."
kmat(D, x, y) = (Ns = size(D, 5); No = size(D, 6);
                 reshape(permutedims(D[x, y, :, :, :, :], (1, 4, 2, 3)), 3No, 3Ns))

"|k_m| on the padded FFT grid (FFT-native order)."
function kabs(Px, Py, Δ)
    f(i, P) = 2π * (i - 1 - (i - 1 > P ÷ 2 ? P : 0)) / (P * Δ)
    [hypot(f(i, Px), f(j, Py)) for i in 1:Px, j in 1:Py]
end

function analyse(label, Ms::Matrix{<:AbstractMatrix}, kab, kmax_prop, Δ; plane_z=nothing, src_z=nothing)
    Px, Py = size(Ms)
    S = [svdvals(Ms[x, y]) for x in 1:Px, y in 1:Py]
    σ1max = maximum(first, S)
    nmax = length(S[1])
    @printf("  %-26s %3d×%-3d  σ₁,max %.2e |", label, size(Ms[1], 1), size(Ms[1], 2), σ1max)
    for τ in τs
        rk = [count(>(τ * σ1max), s) for s in S]
        @printf("  rank@%.0e max %2d mean %5.2f", τ, maximum(rk), mean(rk))
    end
    tot = sum(sum(abs2, s) for s in S)
    print("\n" * " "^35 * "| e_R (Frobenius):")
    for R in Rs
        R < nmax || continue
        @printf("  R=%-2d %.1e", R, sqrt(sum(sum(abs2, s[R+1:end]) for s in S) / tot))
    end
    # σ₅/σ₁ per k_m, by band: propagating somewhere in the stack, evanescent inside the pixel band, near the band edge
    edge = π / Δ
    bands = [("|k|<k_max", k -> k < kmax_prop), ("k_max≤|k|<0.6π/Δ", k -> kmax_prop <= k < 0.6edge),
             ("|k|≥0.6π/Δ", k -> k >= 0.6edge)]
    print("\n" * " "^35 * "| σ₅/σ₁ local, max (median):")
    for (bl, f) in bands
        r = [length(s) > 4 && s[1] > 0 ? s[5] / s[1] : 0.0 for (s, k) in zip(S, kab) if f(k)]
        isempty(r) || @printf("  %s %.1e (%.1e)", bl, maximum(r), median(r))
    end
    println()
    # one basis per side shared by every k_m: D(k_m) ≈ U·C(k_m)·Vᵀ. Ranks of the obs-side span
    # (eigenvalues of Σ_k M M^H) and of the source-side span (Σ_k M^H M), relative to the largest.
    Go = sum(M * M' for M in Ms); Gs = sum(M' * M for M in Ms)
    λo = sqrt.(max.(reverse(eigvals(Hermitian(Go))), 0)); λs = sqrt.(max.(reverse(eigvals(Hermitian(Gs))), 0))
    print(" "^35 * "| shared basis over all k_m (obs side U / source side V), rank of " * "$(length(λo)) / $(length(λs)):")
    for τ in τs
        @printf("  @%.0e %2d / %2d", τ, count(>(τ * λo[1]), λo), count(>(τ * λs[1]), λs))
    end
    println()
    # where the rank-4 residual lives: per row group (obs plane) / column group (source plane), over all k
    if plane_z !== nothing
        No = size(Ms[1], 1) ÷ 3; Ns = size(Ms[1], 2) ÷ 3
        rown = zeros(No); rowr = zeros(No); coln = zeros(Ns); colr = zeros(Ns)
        for M in Ms
            F = svd(M)
            R4 = F.U[:, 5:end] * Diagonal(F.S[5:end]) * F.Vt[5:end, :]
            for j in 1:No
                rows = (0:2) .* No .+ j
                rown[j] += sum(abs2, M[rows, :]); rowr[j] += sum(abs2, R4[rows, :])
            end
            for i in 1:Ns
                cols = (0:2) .* Ns .+ i
                coln[i] += sum(abs2, M[:, cols]); colr[i] += sum(abs2, R4[:, cols])
            end
        end
        println(" "^35 * "| rank-4 residual per obs plane (local z: relative residual):")
        println(" "^37 * join([@sprintf("%.3f: %.0e", plane_z[j], sqrt(rowr[j] / rown[j])) for j in 1:No], "  "))
        println(" "^35 * "| rank-4 residual per source plane:")
        println(" "^37 * join([@sprintf("%.3f: %.0e", src_z[i], sqrt(colr[i] / coln[i])) for i in 1:Ns], "  "))
    end
end

g = uniform_kxky_grid(L=L, Δx=Δ)
Px, Py = GO._padded_size(g)
zs = planes(0.5, Δ)
@printf("Δ = %.4f (cubic voxel), %d×%d px (padded %d×%d), Ns = No = %d, threads %d\n", Δ, g.Nx, g.Ny, Px, Py, length(zs), Threads.nthreads())

for (key, name, δ0, mk) in stacks
    δ = haskey(ENV, "ZRANK_LOSS") ? parse(Float64, ENV["ZRANK_LOSS"]) : δ0
    st = mk(δ)
    zobs = Dict(o => planes(0.5, Δ) for o in 1:3)
    rtol = ModeSolver._is_azimuthally_symmetric_stack(st) ? 1e-3 : 1e-4
    t_grid = @elapsed groups, ginfo = box_kρ_grid_grouped(st, ω, src; box=(Δ, Δ, Δ), N_phi=256, rtol=rtol,
                                                       kρ_cap=parse(Float64, get(ENV, "ZRANK_KCAP", "3000")), ρ_far=L, maxevals=200_000, classes_per_group=4)
    t_build = @elapsed D = dense_blocks(st, groups, g, zs, zobs)
    # largest propagating |k_t| anywhere in the stack: k₀·sqrt(max Re ε) (hyperbolic layers propagate beyond it)
    kmax_prop = ω * sqrt(maximum(maximum(real, l.ϵ.ϵ isa UniformScaling ? (l.ϵ.ϵ.λ,) : diag(l.ϵ.ϵ)) for l in st))
    @printf("\n== %s  %s, δ = %.0e: %d kρ × φ nodes, grid %.0f s, dense blocks %.0f s ==\n",
            key, name, δ, ginfo.n_nodes, t_grid, t_build)
    kab = kabs(Px, Py, Δ)
    for o in (1, 2, 3)
        Ms = [kmat(D[o], x, y) for x in 1:Px, y in 1:Py]
        analyse(o == src ? "2 → 2 (scattered)" : "2 → $o (cross)", Ms, kab, kmax_prop, Δ; plane_z=zobs[o], src_z=zs)
    end
    Ms = [vcat(kmat(D[1], x, y), kmat(D[2], x, y), kmat(D[3], x, y)) for x in 1:Px, y in 1:Py]
    analyse("all scattered, shared V", Ms, kab, kmax_prop, Δ)
    D = nothing; GC.gc()
    flush(stdout)
end
