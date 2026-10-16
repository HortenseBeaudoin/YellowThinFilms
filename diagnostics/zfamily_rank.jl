# diagnostics/zfamily_rank.jl -- numerical rank of a layer's z-exponential family
# {e^{iQ(kρ)z_j}} on N uniform planes (0.5-thick layer), kρ ∈ [0, 3000]: the size of the
# node-independent z skeleton of step 3 (see STATUS, "Step 3 pre-study"). No stack solve.
#
#   julia diagnostics/zfamily_rank.jl
using LinearAlgebra, Printf
k0 = 2π; d = 0.5
kρs = vcat(range(0, 40, length=4000), exp.(range(log(40), log(3000), length=2000)))
Qiso(ε, kρ) = (q = sqrt(complex(ε * k0^2 - kρ^2)); imag(q) < 0 ? -q : q)
# hyperbolic TM-like branch (εt = 2.25, εz = −3.2): Q² = εt(k0² − kρ²/εz) -> propagating at large kρ
Qhyp(kρ) = (q = sqrt(complex(2.25+0.1im) * (k0^2 - kρ^2 / complex(-3.2, 0.1)) + 0im); imag(q) < 0 ? -q : q)
function famrank(Qf, N; twosided=true)
    z = ((1:N) .- 0.5) .* (d / N)
    cols = [exp.(im .* Qf(k) .* z) for k in kρs]
    twosided && append!(cols, [exp.(im .* Qf(k) .* (d .- z)) for k in kρs])
    s = svdvals(reduce(hcat, cols))
    [count(>(τ * s[1]), s) for τ in (1e-3, 1e-6, 1e-9, 1e-12)]
end
@printf("%-34s %s\n", "family", join([@sprintf("N=%-3d rank@1e-3/6/9/12", N) for N in (8, 16, 32, 64, 128)], "   "))
for (lbl, f, two) in [("vacuum cladding (one-sided)", k -> Qiso(1.0 + 1e-3im, k), false),
                      ("glass layer, both branches", k -> Qiso(2.25 + 1e-3im, k), true),
                      ("hyperbolic layer, both branches", Qhyp, true)]
    @printf("%-34s %s\n", lbl, join([@sprintf("%-6s %-15s", "", join(famrank(f, N; twosided=two), "/")) for N in (8, 16, 32, 64, 128)], " "))
end
