# diagnostics/zskeleton_rank.jl -- size of the step 3 z skeletons with the criterion the
# operator uses (`GreenOperator._zskeleton`): max interpolation error over EVERY node term
# ≤ τ, absolute (all family entries have modulus ≤ 1), instead of σ relative to σ₁ of a
# sampled family (diagnostics/zfamily_rank.jl, which under-counts: σ₁ grows with the
# number of sampled columns). Families from each layer's own mode solve, both modes,
# biaxial layers on a φ grid. Ranks are reported without the 80% "keep whole" rule.
#
#   julia -t 8 --project diagnostics/zskeleton_rank.jl
include(joinpath(@__DIR__, "..", "ClosedForm.jl"))
using Printf
const GO = GreenOperator
ω = 2π
kρs = vcat(range(0, 40, length=4000), exp.(range(log(40), log(3000), length=2000))[2:end])
lay(m, d) = FlatOpticalLayer(m, d)
layers = [
    ("vacuum cladding (outgoing only)", lay(IsotropicMaterial(1.0 + 1e-3im), 1.0), :clad, false),
    ("glass layer", lay(IsotropicMaterial(2.25 + 1e-3im), 1.0), :mid, false),
    ("biaxial εzz = +2 layer", lay(BiaxialMaterial(2.25 + 1e-3im, 4.6 + 1e-3im, 2.0 + 1e-3im), 1.0), :mid, true),
    ("hyperbolic εzz = −3.2, δ = 0.1", lay(BiaxialMaterial(2.25 + 0.1im, 4.6 + 0.1im, -3.2 + 0.1im), 1.0), :mid, true),
]
cases = [(0.5, N) for N in (8, 16, 32, 64, 128)]
append!(cases, [(2.0, 64), (2.0, 128)])
τs = (1e-4, 1e-6, 1e-9)
@printf("%-32s %s\n", "rank (obs side / source side)", join([@sprintf("%-22s", @sprintf("t=%.1fλ N=%d", t, N)) for (t, N) in cases], ""))
@printf("%-32s %s\n", "", join([@sprintf("%-22s", "τ=1e-4/1e-6/1e-9") for _ in cases], ""))
for (lbl, l, where, biax) in layers
    φs = biax ? range(0, 2π, length=33)[1:end-1] : [0.0]
    kts = [(k * cos(φ), k * sin(φ)) for φ in φs for k in kρs]
    Qf, Qb = GO._layer_Q(l, ω, kts)
    for side in (:obs, :src)
        where === :clad && side === :src && continue
        row = String[]
        for (t, N) in cases
            z = collect(((1:N) .- 0.5) .* (t / N))
            fam = side === :obs ? GO._obs_family(Qf, Qb, t, where === :clad ? 3 : 2, 3) : GO._src_family(Qf, Qb, t, 2)
            push!(row, @sprintf("%-22s", join([GO._zskeleton(z, fam; tol=τ, whole_frac=2.0) |> s -> length(s.idx) for τ in τs], "/")))
        end
        @printf("%-32s %s\n", lbl * (side === :obs ? ", obs" : ", src"), join(row, ""))
        flush(stdout)
    end
end
