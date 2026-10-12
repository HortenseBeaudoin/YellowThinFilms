# Gz_ClosedForm_TestHelpers.jl -- shared by the Gz_ClosedForm_*_Validation.jl scripts.
# Include after ClosedForm.jl, with `ω` defined.

using LinearAlgebra, FastGaussQuadrature

relerr(a, b) = norm(a .- b) / max(norm(b), 1e-300)

# Fixed Gauss–Legendre kρ rule: panels of `width` (8 nodes each) broken at the layers'
# light lines, up to `kmax`. A plain reference/test rule, independent of any probing.
function kρ_grid(stack; kmax=20.0, width=0.5, order=8)
    brk = sort(unique(vcat(0.0, [real(sqrt(complex(l.ϵ[i, i]))) * ω for l in stack for i in 1:2], kmax)))
    brk = filter(b -> 0 <= b <= kmax, brk)
    x, w = gausslegendre(order)
    ks = Float64[]; ws = Float64[]
    for i in 1:(length(brk) - 1)
        a, b = brk[i], brk[i + 1]
        npan = max(1, ceil(Int, (b - a) / width))
        for j in 0:(npan - 1)
            lo = a + j * (b - a) / npan; hi = lo + (b - a) / npan
            append!(ks, (hi - lo) / 2 .* x .+ (hi + lo) / 2); append!(ws, (hi - lo) / 2 .* w)
        end
    end
    return ks, ws
end

uniform_φ(Np) = ([2π * p / Np for p in 0:(Np - 1)], fill(2π / Np, Np))

# Composite Gauss–Legendre panels on [a,b], geometrically graded toward both ends.
function graded_rule(a, b; levels=14, order=8)
    t, w = gausslegendre(order)
    m = (a + b) / 2
    pts = sort(unique(vcat(a, b, m, [a + (m - a) * 2.0^-j for j in 1:levels], [b - (b - m) * 2.0^-j for j in 1:levels])))
    zs = Float64[]; ws = Float64[]
    for i in 1:(length(pts) - 1)
        lo, hi = pts[i], pts[i + 1]
        append!(zs, (hi - lo) / 2 .* t .+ (hi + lo) / 2); append!(ws, (hi - lo) / 2 .* w)
    end
    return zs, ws
end

# Exact z-average over the slab [zbar-Δz/2, zbar+Δz/2] of POINT-source fields at one
# observation row zo (split at the kink zpos == zo when zo is inside the slab).
# kw go to closed_form_G (e.g. box_lateral on a 2D cache).
function slab_average(c, obs, zbar, Δz, zo, xs, ys; kw...)
    zlo, zhi = zbar - Δz / 2, zbar + Δz / 2
    zkink = c.source_layer == 1 ? c.d_src - zo : zo
    segs = (obs == c.source_layer && zlo < zkink < zhi) ? [(zlo, zkink), (zkink, zhi)] : [(zlo, zhi)]
    G = zeros(ComplexF64, length(xs), length(ys), 1, 3, 3)
    for (a, b) in segs
        zq, wq = graded_rule(a, b)
        for (z, w) in zip(zq, wq)
            G .+= (w / Δz) .* closed_form_G(c, obs, z, [zo], xs, ys; kw...)
        end
    end
    return G
end

# Whole-stack observation rows at global depths (j−0.5)·pz: (layer, local z) per row.
function stack_rows(stack, pz)
    cum = cumsum([0.0; [l.d for l in stack]])
    Nz = round(Int, cum[end] / pz)
    rows = Tuple{Int,Float64}[]
    for j in 1:Nz
        zg = (j - 0.5) * pz
        l = clamp(searchsortedlast(cum, zg), 1, length(stack))
        push!(rows, (l, zg - cum[l]))
    end
    return rows, cum
end
