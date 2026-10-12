include("ClosedForm.jl")
using LinearAlgebra, StaticArrays, Printf, Random

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Operator_Validation.jl -- the precomputed z-structured operator
# (mod_GreenOperator.jl):
#
#  [t] term split of closed_form_G: :toeplitz + :hankel = :all and
#      :direct + :scattered = :all (any medium); on isotropic/uniaxial source
#      layers :toeplitz depends on z'−zpos only and :hankel on z'+zpos only
#      (1D and 2D caches, source layers 1 -- flipped local z -- and 2);
#      the identities fail on a biaxial source layer (reported).
#  [o] apply_operator = closed_form_field (the per-(source depth) route) for
#      random moments on several source planes and lateral positions, every
#      observation layer: structured (iso point 1D, iso box 2D, iso source
#      layer in a biaxial stack), toeplitz_dense (biaxial source layer) and
#      structure=:dense; source layers 1 and 2. Point sources use observation
#      planes half a step off the source planes (a point source's own plane is
#      singular); boxes use the same planes (interior slab included).
#  [m] build / apply time and memory, structured vs dense, on a larger case.
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")
lay(m, d) = FlatOpticalLayer(m, d)
m_vac = IsotropicMaterial(1.0 + 0.001im)
m_gl  = IsotropicMaterial(2.25 + 0.01im)
m_uni = UniaxialMaterial(2.0 + 0.01im, 3.5 + 0.01im)
m_bi  = BiaxialMaterial(2.25 + 0.01im, 4.6 + 0.08im, -1.0 + 0.3im)
passed = Dict{String,Bool}()
Random.seed!(1)

s = 1 / 16
grid = uniform_kxky_grid(L=0.5, Δx=s)
xs, ys = [0.0, 0.13, -0.31], [0.0, 0.2]
planes(d; shift=0.0) = collect(((1:round(Int, d / s)) .- 0.5 .+ shift) .* s)

stack_iso = [lay(m_vac, 1.0), lay(m_gl, 0.5), lay(m_vac, 1.0)]
stack_uni = [lay(m_vac, 1.0), lay(m_uni, 0.5), lay(m_gl, 0.4), lay(m_vac, 1.0)]
stack_bi  = [lay(m_vac, 1.0), lay(m_bi, 0.5), lay(m_vac, 1.0)]

c2d(st, src; Nφ=32, kmax=25.0) = (kρ_w = kρ_grid(st; kmax=kmax); φ_w = uniform_φ(Nφ);
                                   build_cavity_transfer_2d(st, ω, src, kρ_w..., φ_w...))
c1d(st, src) = (kx_w = point_kx_grid(st, ω, src; ρ_max=1.5, pz=s); build_cavity_transfer_1d(st, ω, src, kx_w[1], kx_w[2]))

# ── [t] term split ──
println("[t] closed_form_G term split")
worst_split = 0.0; worst_struct = 0.0
for (lbl, st, src, mk, kw) in (("iso 1D, layer 2", stack_iso, 2, c1d, (;)),
                               ("iso 1D, layer 1", stack_iso, 1, c1d, (;)),
                               ("uniaxial 2D, layer 2", stack_uni, 2, c2d, (;)),
                               ("iso box 2D, layer 1", stack_iso, 1, c2d, (box_lateral=(s, s), Δz_box=s)),
                               ("biaxial 2D, layer 2", stack_bi, 2, c2d, (;)))
    c = mk(st, src)
    kwd = c isa CavityTransferCache2D ? (; kw..., lateral=:direct) : kw
    G(t, z, zo) = closed_form_G(c, src, z, [zo], xs, ys; terms=t, kwd...)
    zpos(z) = src == 1 ? st[1].d - z : z                       # cavity frame
    zl(zp) = src == 1 ? st[1].d - zp : zp                       # back to local
    d = st[src].d
    a, b, δ = 0.3d, 0.62d, 0.09d                                  # cavity-frame source / observation depths
    e1 = relerr(G(:toeplitz, zl(a), b) .+ G(:hankel, zl(a), b), G(:all, zl(a), b))
    e2 = relerr(G(:direct, zl(a), b) .+ G(:scattered, zl(a), b), G(:all, zl(a), b))
    eT = relerr(G(:toeplitz, zl(a + δ), b + δ), G(:toeplitz, zl(a), b))
    eH = relerr(G(:hankel, zl(a + δ), b - δ), G(:hankel, zl(a), b))
    global worst_split = max(worst_split, e1, e2)
    lbl == "biaxial 2D, layer 2" || (global worst_struct = max(worst_struct, eT, eH))
    @printf("   %-22s T+H=all %.1e  direct+scattered=all %.1e | T(ζ) shift %.1e  H(σ) shift %.1e%s\n",
            lbl, e1, e2, eT, eH, lbl == "biaxial 2D, layer 2" ? "   (biaxial: not expected to hold)" : "")
end
passed["t"] = worst_split < 1e-13 && worst_struct < 1e-12

# ── [o] operator vs per-depth route ──
println("[o] apply_operator vs closed_form_field")
function random_moments(op; nsrc=5)
    P = zeros(ComplexF64, op.grid.Nx, op.grid.Ny, length(op.z_src), 3)
    for _ in 1:nsrc
        P[rand(1:op.grid.Nx), rand(1:op.grid.Ny), rand(1:length(op.z_src)), :] .= randn(ComplexF64, 3)
    end
    return P
end
function reference(op, cache, P, zobs; kw...)
    xg, yg = grid_offsets(op.grid)
    dips = [Dipole_constructor(Vector(P[i, j, k, :]), ω, xg[i], op.z_src[k], op.source_layer; y=yg[j])
            for i in axes(P, 1), j in axes(P, 2), k in axes(P, 3) if any(!iszero, P[i, j, k, :])]
    return closed_form_field(Dict(op.source_layer => cache), op.grid, zobs, dips; kw...)
end
worst_o = 0.0
for (lbl, st, src, mk, kw, box, structure, expect) in (
        ("iso point 1D, layer 2",     stack_iso, 2, c1d, (;), false, :auto, :structured),
        ("iso point 1D, layer 1",     stack_iso, 1, c1d, (;), false, :auto, :structured),
        ("iso box 2D, layer 1",       stack_iso, 1, c2d, (box_lateral=(s, s), Δz_box=s, lateral=:nufft, nufft_eps=1e-12), true, :auto, :structured),
        ("vacuum box 2D in biaxial stack, layer 1", stack_bi, 1, c2d, (box_lateral=(s, s), Δz_box=s, lateral=:nufft, nufft_eps=1e-12), true, :auto, :structured),
        ("biaxial box 2D, layer 2",   stack_bi, 2, c2d, (box_lateral=(s, s), Δz_box=s, lateral=:nufft, nufft_eps=1e-12), true, :auto, :toeplitz_dense),
        ("iso point 1D, layer 2, dense", stack_iso, 2, c1d, (;), false, :dense, :dense))
    c = mk(st, src)
    zsrc = planes(st[src].d)
    zobs = Dict(o => planes(st[o].d; shift=(o == src && !box) ? 0.5 : 0.0)[1:end-(o == src && !box ? 1 : 0)] for o in 1:length(st))
    op = build_layer_operator(c, grid, zsrc, zobs; structure=structure, kw...)
    kinds = Dict(b.obs_layer => b.kind for b in op.blocks)
    P = random_moments(op)
    E = apply_operator(op, P)
    R = reference(op, c, P, zobs; kw...)
    e = maximum(relerr(E[o], R[o]) for o in keys(zobs))
    global worst_o = max(worst_o, kinds[src] == expect ? e : Inf)
    @printf("   %-40s same-layer block %-15s vs per-depth route %.2e\n", lbl, kinds[src], e)
end
passed["o"] = worst_o < 1e-10

# ── [m] cost: structured vs dense ──
println("[m] build / apply / memory, structured vs dense (iso box 2D, source layer 2 of vac 1.0 / glass 1.0 / vac 1.0, Δz = 1/32)")
stack_m = [lay(m_vac, 1.0), lay(m_gl, 1.0), lay(m_vac, 1.0)]
sm = 1 / 32
cm = c2d(stack_m, 2; Nφ=64, kmax=40.0)
gm = uniform_kxky_grid(L=0.5, Δx=sm)
zm = collect(((1:32) .- 0.5) .* sm)
kwm = (box_lateral=(sm, sm), Δz_box=sm, lateral=:nufft)
for structure in (:auto, :dense)
    tb = @elapsed op = build_layer_operator(cm, gm, zm, Dict(2 => zm); structure=structure, kwm...)
    P = zeros(ComplexF64, gm.Nx, gm.Ny, 32, 3); P[5, 9, 3, :] .= 1; P[20, 11, 30, 3] = 1
    apply_operator(op, P)
    ta = @elapsed E = apply_operator(op, P)
    @printf("   %-6s build %6.2f s  apply %.3f s  (%d source × %d obs planes)\n", structure, tb, ta, 32, 32)
    operator_summary(op)
    structure === :auto ? (global Em = E) : @printf("   structured vs dense: %.1e\n", relerr(Em[2], E[2]))
end

println()
for k in sort(collect(keys(passed)))
    println("  [$k] ", passed[k] ? "PASS" : "FAIL")
end
