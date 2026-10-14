include("ClosedForm.jl")
using LinearAlgebra, Printf, Statistics, Dates, Random

# ─────────────────────────────────────────────────────────────────────────────
# Bench_GreenOperator.jl -- informal timing of the precomputed operator
# (mod_GreenOperator.jl). Not a validation: no pass/fail, numbers only.
#
# Per (size, loss, stack): box source in layer 2, every finite layer + the first
# semi-infinite neighbour on each side as observation layers.
#   precompute = adaptive kρ grid + cavity-transfer cache + build_layer_operator
#                (per block too); cache and operator are built per φ chunk and
#                summed, which fits the 256-φ biaxial cases in 16 GB
#   apply      = apply_operator (median of `nrep`) for 1 source, 1 source per
#                source plane, and every voxel of the source layer. The operator's
#                apply cost does not depend on how many moments are non-zero; the
#                three columns are there to show it.
# Worst case / production precision: adaptive box_kρ_grid per (stack, voxel size)
# with the knobs of the Gila comparison (rtol 1e-3, kρ_cap 3000, N_phi 256,
# ρ_far = half the window) and nufft_eps = 1e-12. The interior (source-plane)
# field is still bounded by kρ_cap -- see STATUS, Newell subtraction.
#
# Loss: the same δ = Im ε on every component of every layer (claddings included),
# δ ∈ BENCH_LOSSES (default 1e-1, 1e-3, 1e-6); the real parts are fixed per stack.
# A case is skipped, and recorded as such, when box_kρ_grid stalls (it raises) or
# needs more than BENCH_KMAX kρ (default 40k; low-loss biaxial hyperbolic, STATUS);
# a (stack, δ) skipped at one size is skipped at the larger ones too, and so is one
# whose adaptive tail stalled (its row is kept, marked `!`: fixed-band fallback,
# not converged). BENCH_RESUME=1 skips the cases already in the CSV and reloads
# its skips.
#
#   julia -t auto --project Bench_GreenOperator.jl            # 4 stacks × 3 losses × S, M, L (hours)
#   BENCH_SIZES=S,M BENCH_STACKS=iso,uni BENCH_LOSSES=1e-3 julia -t auto --project Bench_GreenOperator.jl
#
# Appends one row per case to logs/bench_green_operator_loss.csv (git-ignored).
# ─────────────────────────────────────────────────────────────────────────────

ω = 2π
include("Gz_ClosedForm_TestHelpers.jl")
lay(m, d) = FlatOpticalLayer(m, d)
m_vac(δ) = IsotropicMaterial(complex(1.0, δ))
m_gl(δ)  = IsotropicMaterial(complex(2.25, δ))
m_uni(δ) = UniaxialMaterial(complex(2.0, δ), complex(3.5, δ))
m_bi(δ)  = BiaxialMaterial(complex(2.25, δ), complex(4.6, δ), complex(2.0, δ))    # Gila R3's real parts
m_hyp(δ) = BiaxialMaterial(complex(2.25, δ), complex(4.6, δ), complex(-3.2, δ))   # hyperbolic; not −1 (ε+1 = 0 against vacuum: plasmon condition)

# source is always layer 2; outer layers are 0.5 thick so their dense blocks stay small
stacks = [
    ("iso   vac/glass/vac",     δ -> [lay(m_vac(δ), 0.5), lay(m_gl(δ), 0.5),  lay(m_vac(δ), 0.5)]),
    ("uni   vac/uni/glass/vac", δ -> [lay(m_vac(δ), 0.5), lay(m_uni(δ), 0.5), lay(m_gl(δ), 0.25), lay(m_vac(δ), 0.5)]),
    ("biax  vac/biax/vac",      δ -> [lay(m_vac(δ), 0.5), lay(m_bi(δ), 0.5),  lay(m_vac(δ), 0.5)]),
    ("hyp   vac/hyp-biax/vac",  δ -> [lay(m_vac(δ), 0.5), lay(m_hyp(δ), 0.5), lay(m_vac(δ), 0.5)]),
]

# Δ = voxel side (lateral and z), L = half-width of the lateral window
sizes = Dict(
    "S" => (Δ=1/8,  L=1.0),
    "M" => (Δ=1/16, L=1.0),
    "L" => (Δ=1/32, L=1.0),
)

# quadrature / accuracy knobs (as in the Gila comparison, STATUS)
BOX_RTOL  = 1e-3
BOX_KCAP  = 3000.0
BOX_NPHI  = 256
BOX_MAXEV = 200_000     # the hyperbolic source layer's tails need > 50k at 256 φ (else a fixed-band fallback)
NUFFT_EPS = 1e-12
PHI_CHUNK = 64          # cache + operator built per φ chunk and summed (exact: G is linear in the
                        # quadrature sum); a 256-φ biaxial cache does not fit in 16 GB with the build
NODE_BUDGET = 900_000   # kρ × φ per chunk (≈ 14k × 64, the largest that fit): fewer φ per chunk past 14k kρ
KMAX = parse(Int, get(ENV, "BENCH_KMAX", "40000"))
losses = parse.(Float64, split(get(ENV, "BENCH_LOSSES", "1e-1,1e-3,1e-6"), ","))
size_order = filter(s -> haskey(sizes, s), split(get(ENV, "BENCH_SIZES", "S,M,L"), ","))
let keep = split(get(ENV, "BENCH_STACKS", "iso,uni,biax,hyp"), ",")       # by the label's first word
    filter!(st -> first(split(st[1])) in keep, stacks)
end
nrep = 5
Random.seed!(1)

"""
Cache + `build_layer_operator` per chunk of `PHI_CHUNK` φ nodes, summed block by block.
Returns `(op, t_cache, t_build)`; `op.timings` holds the per-block build time summed over chunks.
"""
function build_phi_chunked(st, src, kρ, wρ, φ, wφ, g, zsrc, zobs; kw...)
    op = nothing; t_cache = 0.0; t_build = 0.0
    chunk = clamp(NODE_BUDGET ÷ length(kρ), 1, PHI_CHUNK)
    for r in Iterators.partition(eachindex(φ), chunk)
        t_cache += @elapsed c = build_cavity_transfer_2d(st, ω, src, kρ, wρ, φ[r], wφ[r]; obs_layers=1:length(st))
        t_build += @elapsed opc = build_layer_operator(c, g, zsrc, zobs; kw...)
        c = nothing
        if op === nothing
            op = opc
        else
            for (b, bc) in zip(op.blocks, opc.blocks)
                b.kind == bc.kind || error("φ chunks chose different block kinds ($(b.kind) vs $(bc.kind))")
                for f in (:Tz, :Hz, :D)
                    x = getfield(b, f)
                    x === nothing || (x .+= getfield(bc, f))
                end
                op.timings[b.obs_layer] += opc.timings[b.obs_layer]
            end
        end
        opc = nothing; GC.gc()
    end
    return op, t_cache, t_build
end

planes(d, Δ) = collect(((1:round(Int, d / Δ)) .- 0.5) .* Δ)
mb(op) = sum(sum(x === nothing ? 0 : sizeof(x) for x in (b.Tz, b.Hz, b.D)) for b in op.blocks) / 2^20

csv = joinpath(@__DIR__, "logs", "bench_green_operator_loss.csv")
mkpath(dirname(csv))
isfile(csv) || open(io -> println(io, "date,threads,stack,loss,size,Nx,Ny,Ns,nkρ,nφ,tails_stalled,t_grid,t_cache,t_build,t_precompute,MB,t_apply_1,t_apply_Ns,t_apply_all,blocks,skipped"), csv, "w")

# warm-up (compilation) on a tiny case of every stack, not timed
for (_, mk) in stacks
    st = mk(1e-1)
    g = uniform_kxky_grid(L=0.25, Δx=1/8)
    kρ, wρ, φ, wφ, _ = box_kρ_grid(st, ω, 2; box=(1/8, 1/8, 1/8), N_phi=8, rtol=1e-1, kρ_cap=50.0, ρ_far=0.25)
    c = build_cavity_transfer_2d(st, ω, 2, kρ, wρ, φ, wφ)
    z = planes(0.5, 1/8)
    op = build_layer_operator(c, g, z, Dict(o => planes(min(st[o].d, 0.5), 1/8) for o in eachindex(st));
                              box_lateral=(1/8, 1/8), Δz_box=1/8, lateral=:nufft, nufft_eps=NUFFT_EPS)
    apply_operator(op, ones(ComplexF64, g.Nx, g.Ny, length(z), 3))
end

@printf("threads: %d   FFTW threads set per call to nthreads()\n\n", Threads.nthreads())
@printf("kρ: box_kρ_grid rtol=%g kρ_cap=%g maxevals=%d, N_phi=%d (chunks of ≤ %d φ, ≤ %d nodes), nufft_eps=%g, kρ budget %d\n\n",
        BOX_RTOL, BOX_KCAP, BOX_MAXEV, BOX_NPHI, PHI_CHUNK, NODE_BUDGET, NUFFT_EPS, KMAX)
@printf("%-24s %-6s %-4s %8s %4s %11s | %7s %7s %8s %9s %8s | %8s %8s %8s\n",
        "stack", "δ", "size", "Nx×Ny", "Ns", "kρ × φ", "grid s", "cache s", "build s", "precomp s", "MB", "1 src ms", "Ns src", "all ms")
println("-"^140)

skipped = Dict{Tuple{String,Float64},String}()      # (stack, δ) → reason, carried to larger sizes
done = Set{Tuple{String,Float64,String}}()           # (stack, δ, size) already in the CSV (BENCH_RESUME=1)
if get(ENV, "BENCH_RESUME", "0") == "1"
    names = Dict(strip(n) => n for (n, _) in stacks)
    for l in readlines(csv)[2:end]
        f = split(l, ","); haskey(names, f[3]) || continue
        key = (names[f[3]], parse(Float64, f[4]))
        push!(done, (key..., f[5]))
        reason = f[end]
        if !isempty(reason) && !occursin("smaller size", reason)
            skipped[key] = replace(reason, r" \(.*" => "")
        elseif !isempty(f[11]) && parse(Int, f[11]) > 0
            skipped[key] = "tail stalled"
        end
    end
    println("resuming: $(length(done)) cases in the CSV, skips ", skipped, "\n")
end
function record_skip(name, δ, sz, g, Ns, reason, t_grid=NaN, nkρ=0)
    @printf("%-24s %-6.0e %-4s %8s %4d %11s | skipped: %s\n", name, δ, sz, "$(g.Nx)×$(g.Ny)", Ns,
            nkρ > 0 ? "$nkρ × $BOX_NPHI" : "-", reason)
    open(csv, "a") do io
        @printf(io, "%s,%d,%s,%.0e,%s,%d,%d,%d,%d,%d,,%.3f,,,,,,,,,%s\n", Dates.format(now(), "yyyy-mm-ddTHH:MM"),
                Threads.nthreads(), strip(name), δ, sz, g.Nx, g.Ny, Ns, nkρ, BOX_NPHI, t_grid, reason)
    end
    flush(stdout)
end

for sz in size_order, δ in losses, (name, mk) in stacks
    st = mk(δ)
    p = sizes[sz]
    g = uniform_kxky_grid(L=p.L, Δx=p.Δ)
    src = 2
    obs = 1:length(st)
    zsrc = planes(st[src].d, p.Δ)
    zobs = Dict(o => planes(min(st[o].d, 0.5), p.Δ) for o in obs)     # semi-infinite: first 0.5 only
    kw = (box_lateral=(p.Δ, p.Δ), Δz_box=p.Δ, lateral=:nufft, nufft_eps=NUFFT_EPS)

    Ns = length(zsrc)
    (name, parse(Float64, @sprintf("%.0e", δ)), sz) in done && continue
    if haskey(skipped, (name, δ))
        record_skip(name, δ, sz, g, Ns, skipped[(name, δ)] * " (at a smaller size)")
        continue
    end

    GC.gc()
    local kρ, wρ, φ, wφ, ginfo
    t_grid = @elapsed try
        kρ, wρ, φ, wφ, ginfo = box_kρ_grid(st, ω, src; box=(p.Δ, p.Δ, p.Δ), N_phi=BOX_NPHI,
                                           rtol=BOX_RTOL, kρ_cap=BOX_KCAP, ρ_far=p.L, maxevals=BOX_MAXEV)
    catch e
        e isa ErrorException && occursin("did not reach rtol", e.msg) || rethrow()
        kρ = nothing
    end
    if kρ === nothing
        skipped[(name, δ)] = "grid stalled"
        record_skip(name, δ, sz, g, Ns, "grid stalled", t_grid)
        continue
    elseif length(kρ) > KMAX
        skipped[(name, δ)] = "kρ budget"
        record_skip(name, δ, sz, g, Ns, "kρ budget ($(length(kρ)) > $KMAX)", t_grid, length(kρ))
        continue
    end
    op, t_cache, t_build = build_phi_chunked(st, src, kρ, wρ, φ, wφ, g, zsrc, zobs; kw...)
    nkρ, nφ = length(kρ), length(φ)
    nstall = count(q -> q.stalled, ginfo.per_probe_tail)          # > 0: adaptive tail fell back to a fixed band

    P1 = zeros(ComplexF64, g.Nx, g.Ny, Ns, 3); P1[g.Nx ÷ 2, g.Ny ÷ 2, (Ns + 1) ÷ 2, :] .= 1   # one source
    PN = zeros(ComplexF64, g.Nx, g.Ny, Ns, 3)                                               # one per plane
    for k in 1:Ns; PN[rand(1:g.Nx), rand(1:g.Ny), k, :] .= randn(ComplexF64, 3); end
    PA = randn(ComplexF64, g.Nx, g.Ny, Ns, 3)                                               # full volume
    tapp(P) = (apply_operator(op, P); median([(@elapsed apply_operator(op, P)) for _ in 1:nrep]))
    t1, tN, tA = tapp(P1), tapp(PN), tapp(PA)

    t_pre = t_grid + t_cache + t_build
    @printf("%-24s %-6.0e %-4s %8s %4d %11s | %7.1f %7.1f %8.1f %9.1f %8.1f | %8.1f %8.1f %8.1f\n",
            name, δ, sz, "$(g.Nx)×$(g.Ny)", Ns, "$nkρ × $nφ" * (nstall > 0 ? "!" : ""), t_grid, t_cache, t_build, t_pre, mb(op), 1e3t1, 1e3tN, 1e3tA)
    blocks = String[]
    for b in op.blocks
        bmb = sum(x === nothing ? 0 : sizeof(x) for x in (b.Tz, b.Hz, b.D)) / 2^20
        @printf("      %d → %d  %-15s %3d obs planes  %8.1f MB  %7.2f s\n",
                src, b.obs_layer, b.kind, length(b.z_obs), bmb, op.timings[b.obs_layer])
        push!(blocks, @sprintf("%d:%s:%d:%.1fMB:%.2fs", b.obs_layer, b.kind, length(b.z_obs), bmb, op.timings[b.obs_layer]))
    end
    flush(stdout)
    open(csv, "a") do io
        @printf(io, "%s,%d,%s,%.0e,%s,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%.5f,%.5f,%.5f,%s,\n",
                Dates.format(now(), "yyyy-mm-ddTHH:MM"), Threads.nthreads(), strip(name), δ, sz, g.Nx, g.Ny,
                Ns, nkρ, nφ, nstall, t_grid, t_cache, t_build, t_pre, mb(op), t1, tN, tA, join(blocks, " "))
    end
    nstall > 0 && (skipped[(name, δ)] = "tail stalled")
end
println("\n! = at least one adaptive tail stalled (fixed-band fallback, not converged)")
println("rows appended to ", relpath(csv, @__DIR__))
