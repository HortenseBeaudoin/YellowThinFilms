# diagnostics/loss_sweep.jl -- how low can the losses go before box_kρ_grid stops
# converging, and does the field reach a limit as the loss → 0?
#
# Same loss δ as Im of every ε component, every layer (cladding included).
# Per (stack, δ): box_kρ_grid at the benchmark's max-precision knobs (box 1/8 in
# layer 2, rtol 1e-3, kρ_cap 3000, ρ_far 1, 256 φ), stalls, node count, time,
# narrowest finite panel; then G at a few observation points on that grid, and
# its change vs the previous (10× larger) δ.
#
#   julia -t auto --project diagnostics/loss_sweep.jl
#   LOSS_STACKS=hyp julia -t auto --project diagnostics/loss_sweep.jl
include(joinpath(@__DIR__, "..", "ClosedForm.jl"))
using Printf, LinearAlgebra
ω = 2π
lay(m, d) = FlatOpticalLayer(m, d)
iso(e, δ) = IsotropicMaterial(complex(e, δ))
only = get(ENV, "LOSS_STACKS", "iso,uni,hyp")
stacks = [
    ("iso vac/glass/vac", δ -> [lay(iso(1.0, δ), 0.5), lay(iso(2.25, δ), 0.5), lay(iso(1.0, δ), 0.5)]),
    ("uni vac/uni/glass/vac", δ -> [lay(iso(1.0, δ), 0.5), lay(UniaxialMaterial(complex(2.0, δ), complex(3.5, δ)), 0.5),
                                    lay(iso(2.25, δ), 0.25), lay(iso(1.0, δ), 0.5)]),
    ("hyp vac/biax(−3.2)/vac", δ -> [lay(iso(1.0, δ), 0.5), lay(BiaxialMaterial(complex(2.25, δ), complex(4.6, δ), complex(-3.2, δ)), 0.5),
                                     lay(iso(1.0, δ), 0.5)]),
]
filter!(s -> first(split(s[1])) in split(only, ","), stacks)
δs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.0]
Δ = 1/8; z̄ = 0.25
# (label, obs layer, z_obs local, x)
pts = [("vertical neighbour", 2, z̄ + Δ, 0.0), ("in slab, ρ = 1", 2, z̄, 1.0),
       ("layer above, ρ = 0.5", 1, 0.5 - 0.1, 0.5), ("layer below, ρ = 0.5", 3, 0.1, 0.5)]

for (name, mk) in stacks
    println("\n== $name ==")
    @printf("%8s | %7s %7s %6s %9s | %s\n", "δ", "kρ", "stalls", "grid s", "min panel",
            join([@sprintf("%-22s", p[1]) for p in pts], " "))
    prev = nothing
    for δ in δs
        st = mk(δ)
        GC.gc()
        t = @elapsed (kρ, wρ, φ, wφ, info) = box_kρ_grid(st, ω, 2; box=(Δ, Δ, Δ), N_phi=256, rtol=1e-3,
                                                         kρ_cap=3000.0, ρ_far=1.0, maxevals=200_000, on_stall=:warn)
        nst = count(p -> p.stalled, info.per_probe_fin) + count(p -> p.stalled, info.per_probe_tail)
        # G summed over φ chunks (exact: linear in the quadrature sum); a full 256-φ biaxial cache does not fit
        G = [zeros(ComplexF64, 1, 1, 1, 3, 3) for _ in pts]
        for r in Iterators.partition(eachindex(φ), 64)
            c = build_cavity_transfer_2d(st, ω, 2, kρ, wρ, φ[r], wφ[r])
            for (i, (_, o, zo, x)) in enumerate(pts)
                G[i] .+= closed_form_G(c, o, z̄, [zo], [x], [0.0]; box_lateral=(Δ, Δ), Δz_box=Δ)
            end
            c = nothing; GC.gc()
        end
        chg = prev === nothing ? ["" for _ in pts] :
              [@sprintf("Δrel %.1e", norm(G[i] - prev[i]) / norm(G[i])) for i in eachindex(pts)]
        @printf("%8.0e | %7d %3d/%-3d %6.0f %9.1e | %s\n", δ, length(kρ), nst, 2 * length(info.probes), t,
                info.min_panel_width, join([@sprintf("%-22s", s) for s in chg], " "))
        flush(stdout)
        prev = G
    end
end
