module KxGrid

export branch_points, build_kphi_grid
export thinnest_finite_layer, evanescent_umax
export adaptive_kx_grid

import ..ModeSolver: AbstractLayer
using FastGaussQuadrature
using LinearAlgebra: norm
using QuadGK: quadgk_segbuf, cachedrule


# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

# Finds branch point to build grid around singularities, or avoid them
function branch_points(stack::Vector{<:AbstractLayer}, ω::Float64)
    ks = Float64[]
    for lay in stack
        εxx = real(lay.ϵ[1,1])
        εyy = real(lay.ϵ[2,2])
        εzz = real(lay.ϵ[3,3])
        εxx > 0 && push!(ks, ω * sqrt(εxx))
        εyy > 0 && push!(ks, ω * sqrt(εyy))
        εzz > 0 && push!(ks, ω * sqrt(εzz))
    end
    return sort(unique(ks))
end


"""
    evanescent_umax(kmax, Leff; target_decay=25.0, kx_ratio_max=80.0)

Effective length scale that sets the decay. In u, for change of variables kρ = kmax * cosh(u), 
the tail decays as exp(-kmax * Leff * sinh(u)).
"""
function evanescent_umax(kmax::Float64, Leff::Float64;
                         target_decay::Float64=25.0, kx_ratio_max::Float64=80.0)
    umax_phys = asinh(target_decay / (kmax * Leff))
    umax_cap  = acosh(max(kx_ratio_max, 1.0 + 1e-10))
    return min(umax_phys, umax_cap)
end


"""
    _evanescent_band_nodes(kmax, umax, N; δ=1e-6)

The `kmax -> ∞` cosh-substituted Gauss-Legendre band itself.
"""
function _evanescent_band_nodes(kmax::Float64, umax::Float64, N::Int; δ::Float64=1e-6)
    umax <= δ && return Float64[], Float64[]
    ξe, we = gausslegendre(N)
    u  = @. (umax - δ)/2 * ξe + (umax + δ)/2
    wu = @. we * (umax - δ)/2
    return kmax .* cosh.(u), wu .* (kmax .* sinh.(u))
end


"""
    thinnest_finite_layer(stack) -> Float64

Minimum thickness among `stack`'s genuinely finite layers.
"""
function thinnest_finite_layer(stack::Vector{<:AbstractLayer})
    N = length(stack)
    N <= 1 && return Inf
    return minimum(stack[j].d for j in 1:N-1)
end


#TODO Still usefull ??? Probe adapted ? 
"""
    _kronrod_reference_rule(x, w, wg) -> (ξ, wt)

Expands QuadGK's own packed Gauss-Kronrod rule representation (`x`, `w`,
`wg`, as returned by `QuadGK.cachedrule` -- the half-rule-by-symmetry storage
`QuadGK.evalrule` itself unpacks inline) into an explicit, full list of
reference abscissas `ξ ∈ [-1,1]` and weights `wt` such that `Σ wt_i *
h(ξ_i) ≈ ∫_{-1}^1 h(ξ) dξ` for the KRONROD (not embedded Gauss) rule --
i.e. the exact rule `QuadGK.evalrule` uses for its own returned segment
integral `Segment.I`. The indexing below is a literal transcription of
`QuadGK.evalrule`'s accumulation loop (`adapt.jl`/`evalrule.jl`), just
collecting `(node, weight)` pairs instead of evaluating `f` at them -- see
`adaptive_kx_grid`'s own docstring for why reproducing QuadGK's EXACT rule
(not a fresh, differently-nodED rule) is the point: a fresh rule risks
completely missing sub-panel structure (a narrow resonance, say) that just
happens to fall between its own node positions even though the panel was
correctly judged converged AT QuadGK's actual sample points.
"""
function _kronrod_reference_rule(x::Vector{Float64}, w::Vector{Float64}, wg::Vector{Float64})
    n1 = 1 - (length(x) & 1)
    ξ  = Float64[]
    wt = Float64[]
    if n1 == 0
        push!(ξ, 0.0); push!(wt, w[end])
    else
        push!(ξ, 0.0);         push!(wt, w[end])
        push!(ξ, x[end-1]);    push!(wt, w[end-1])
        push!(ξ, -x[end-1]);   push!(wt, w[end-1])
    end
    for i in 1:(length(wg) - n1)
        push!(ξ, x[2i]);    push!(wt, w[2i])
        push!(ξ, -x[2i]);   push!(wt, w[2i])
        push!(ξ, x[2i-1]);  push!(wt, w[2i-1])
        push!(ξ, -x[2i-1]); push!(wt, w[2i-1])
    end
    return ξ, wt
end


#TODO Still usefull ??? Probe adapted ? 
"""
    _panels_to_nodes(segs, x, w, wg) -> (nodes, weights)

Materializes an explicit composite quadrature rule from a QuadGK segment
decomposition: places QuadGK's OWN Kronrod rule (`_kronrod_reference_rule`,
from `cachedrule(Float64, kr_order)`'s `(x,w,wg)`) on each accepted panel
`[a,b]`. See `adaptive_kx_grid`'s own docstring for why the frozen rule must
be QuadGK's actual rule, not an independently-chosen one.
"""
function _panels_to_nodes(segs, x::Vector{Float64}, w::Vector{Float64}, wg::Vector{Float64})
    ξ, wt = _kronrod_reference_rule(x, w, wg)
    m = length(ξ)
    n = length(segs)
    nodes = Vector{Float64}(undef, n * m)
    wts   = Vector{Float64}(undef, n * m)
    idx = 1
    for s in segs
        a, b = Float64(s.a), Float64(s.b)
        h = 0.5 * (b - a)
        c = 0.5 * (a + b)
        for i in 1:m
            nodes[idx] = c + h * ξ[i]
            wts[idx]   = h * wt[i]
            idx += 1
        end
    end
    return nodes, wts
end


#TODO Still usefull ??? Probe adapted ? 
"""
    _tail_t_cap(kmax, kx_ratio_max) -> t_cap

Inverts the tail substitution `kρ = kmax + t/(1-t)` to find the `t ∈ (0,1)`
corresponding to `kρ = kmax*kx_ratio_max`, so `adaptive_kx_grid` can
integrate the semi-infinite tail over `t ∈ [0,t_cap]` instead of the full
`[0,1)` -- see `adaptive_kx_grid`'s own docstring (`kx_ratio_max` paragraph)
for why this cap is now applied unconditionally, not just on stall.
"""
function _tail_t_cap(kmax::Float64, kx_ratio_max::Float64)
    A = kmax * (max(kx_ratio_max, 1.0 + 1e-10) - 1.0)
    return A / (1.0 + A)
end


# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    adaptive_kx_grid(fs, kbreaks; rtol=1e-6, atol=0.0, kr_order=7,
                     maxevals=10_000, Leff=1.0, fallback_N=250,
                     target_decay=25.0, kx_ratio_max=80.0, warn_stall=true)
        -> (kx_nodes, wt_nodes, info)

Error-controlled replacement for the old fixed `N_prop`/`N_band`/`N_eva` node
placement (Part 2 of the NUFFT/adaptive-quadrature handoff) -- that fixed grid
(formerly `build_kx_grid`, this module) has since moved to
`mod_Dipz_SemiAnalytical.jl` (2026-09-18), kept only as the semi-analytical
ground-truth comparison's own grid, since this function superseded it
everywhere else in the live pipeline; `evanescent_umax`/`_evanescent_band_nodes`
stay here since this function's own tail-stall fallback (below) still needs
them.
`fs::Vector` holds one or more independent probe integrands, each
`f(kρ::Float64)` a caller-supplied PHYSICAL integrand -- in this project
always a full per-node reciprocal-space solve's contribution to some scalar-
or vector-valued probe quantity (see `mod_ClosedFormGrid.jl`'s probes); `kbreaks` the sorted,
strictly-positive branch points (`branch_points(stack, ω)`).

Converges every probe in `fs` over the SAME domain and returns ONE merged
node/weight rule, correctly -- i.e. NOT by concatenating each probe's own
complete `(nodes,weights)` rule (which double/multiply-counts the shared
domain, silently N-tupling any well-resolved quantity) and NOT by stacking
every probe into one combined vector integrand for a single stacked-vector
adaptive-quadrature run either (which technically avoids double-counting,
but introduces a DIFFERENT, subtler bug: that stacked run's stopping test
compares error against the NORM of the whole stacked vector, so a probe with
much SMALLER natural magnitude than another can have its own relative error
hidden entirely inside a combined tolerance dominated by the larger probe --
confirmed directly: stacking `_radial_probe_value`'s ρ=0 and ρ=ρ_max
outputs, whose magnitudes differ by orders of magnitude near a resonance,
converged the COMBINED norm to `rtol` while the smaller-magnitude probe's
own value was still off by >100%).

The fix used here: run each probe's OWN `QuadGK.quadgk_segbuf` h-adaptive
Gauss-Kronrod refinement completely independently (so each probe is judged
against `rtol` scaled by ITS OWN magnitude -- exactly right, no cross-probe
scale bias), on:
  - the finite bands `(0,k1),(k1,k2),...,(k_{n-1},kmax)`;
  - the semi-infinite tail `(kmax,∞)`, via the explicit substitution
    `kρ = kmax + t/(1-t)`, `t ∈ [0,1)`, Jacobian `dkρ/dt = 1/(1-t)^2` --
    written out here rather than relying on QuadGK's own (unexported)
    infinite-interval transform, so the inverse map back to kρ-space nodes
    and weights is exact and auditable rather than reverse-engineered from
    QuadGK internals.

Then, instead of keeping each probe's own node/weight rule, only the PANEL
BOUNDARIES each probe's refinement settled on are collected, unioned
(sorted, deduplicated) across all probes, and ONE Kronrod rule is placed on
that merged partition (`_panels_to_nodes`, reconstructed from
`QuadGK.cachedrule(Float64, kr_order)`'s packed `(x,w,wg)` via
`_kronrod_reference_rule` -- `quadgk_segbuf` itself only returns panel
boundaries plus integral/error estimates, `QuadGK.Segment` storing just `a`,
`b`, `I`, `E`, not an explicit node list). This is mathematically safe: the
merged partition is a common refinement of every individual probe's own
accepted partition (each probe's own boundaries are a SUBSET of the union),
so the merged rule is at least as fine -- hence at least as accurate -- for
every probe as that probe's own independently-converged rule, while still
being exactly ONE quadrature rule for the shared domain (no multiple
counting, since there is exactly one partition and one rule placed on it).

Reusing QuadGK's OWN exact Kronrod rule (not a fresh, differently-noded rule
of the same order) is not merely convenient, it is necessary for
correctness: an EARLIER version of this module placed a fresh,
independently-chosen Gauss-Legendre rule on each accepted panel instead,
reasoning that "a panel already fine enough for Kronrod is fine enough for
any comparable rule" -- this is FALSE whenever the integrand has structure
narrower than the panel that happens to fall between the fresh rule's node
positions (a Fabry-Perot-type resonance in a layered-media reflection
coefficient, say): the panel was correctly judged converged AT QUADGK'S OWN
SAMPLE POINTS, but a differently-noded rule can completely miss that same
structure, giving a frozen node set that reproduces nothing like the
converged value. Reusing QuadGK's exact rule makes "the frozen nodes
reproduce the converged integral" a mathematical identity (the same weighted
sum, not merely similar), not an assumption -- confirmed empirically during
this handoff's own development (see the module's validation script's
revision history) and guarded ongoing by Part 2's validation harness
(re-summing the frozen nodes through the real per-pixel pipeline and
comparing to the semi-analytical ground truth).

`Leff`/`fallback_N`/`target_decay` are only used if ANY probe's own TAIL
integration stalls (hits `maxevals` without reaching `(rtol,atol)` -- the
oscillatory-large-lateral-offset "hard case" the handoff calls out; should
not arise at the probe pixel itself, which sits at zero lateral offset by
construction, but is handled explicitly rather than silently returning an
under-converged tail). On stall, the tail is replaced -- for ALL probes, one
shared node set -- by this module's own fixed cosh-substitution evanescent
band (`evanescent_umax`/`_evanescent_band_nodes`) past `kmax`,
using the supplied `Leff` -- the caller is expected to pass the actual
source-to-interface distance here, since it knows the probe geometry and
this function does not.

`kx_ratio_max` (2026-08-21 fix, found during Part 3's multi-dipole biaxial
validation -- not merely the stall-fallback's own cap anymore): ALWAYS caps
the semi-infinite tail's substitution domain to `kρ ∈ [kmax, kmax*kx_ratio_max]`
(via `t ∈ [0, t_cap]` in the `kρ = kmax + t/(1-t)` substitution), not just
when the tail integration stalls. Previously the CONVERGED tail was
integrated over the full `t ∈ [0,1)`, i.e. `kρ → ∞`; QuadGK's coarsest
Kronrod rule on that whole interval already samples `t` values extremely
close to 1 (its outermost node, before any subdivision), so a probe that
happens to converge WITHOUT needing subdivision there can still return nodes
at astronomically large kρ (confirmed directly: a probe converging on its
very first Kronrod pass on `[0,1)` returned nodes ~100-1000x `kmax`). This is
harmless for a SINGLE dipole (`Modal_amp`/`Modal_amp_profile` evaluated once,
per-node contributions correctly underflow toward zero at extreme kρ), but
CORRUPTS the multi-dipole within-layer path (`Modal_amp_profile` batching
several sources at DIFFERENT z): the forward/backward mode terms for two
sources separated by `Δz` involve `exp(±kz·Δz)`-type factors that stay
FINITE (not NaN/Inf, so `_finite_or_zero`'s guard does not catch it) but lose
essentially all precision to catastrophic cancellation once `kz·Δz` gets
anywhere near Float64's ~709 overflow threshold -- confirmed directly: two
co-layer dipoles at different z, batched through an adaptive grid whose tail
was capped only on stall, came back >50% wrong, entirely inside the shared
source layer's own rows, while the SAME two dipoles at the SAME z (Δz=0, no
cancellation needed) or a SINGLE dipole through the identical grid matched
the fixed-grid answer to the expected accuracy level. `kx_ratio_max=80`
(same default already used by `evanescent_umax`'s fallback and by
`mod_Dipz_SemiAnalytical.jl`'s `build_kx_grid`, both already cross-validated
against analytic/semi-analytical ground truth in Part 2) is comfortably past
`target_decay=25` e-foldings of evanescent decay, so capping the converged
tail there costs no real accuracy -- content beyond it is already many
orders of magnitude below any reasonable `rtol` for the physical integral,
while removing it is what keeps the multi-source formula numerically sane.

`min_panel_width` (Part C, uniform-extended-source handoff): the FINEST
panel QuadGK settled on among the finite bands, off the MERGED
(union-of-probes) finite-band partition -- i.e. the narrowest feature of the
integrand this adaptive refinement actually resolved (a near-real
guided-mode pole, a branch-point cusp). Exposed so a fixed-Δkx uniform grid
can be judged adequate against it (`Δkx` vs. this width) without re-running
the adaptive probe a second time just to ask "how fine did you need to go".
Finite bands only (the merged `fin_segs`, not the evanescent tail) --
`kbreaks` already brackets every branch point/pole location inside the
finite-band domain, so that is where a narrow feature would show up as a
small accepted panel.

`info` reports `(n_panels_fin, n_nodes, per_probe_fin, per_probe_tail,
any_tail_stalled, min_panel_width)` (`warn_stall=false` silences the per-probe
finite-band stall warning, for a caller that reports stalls itself): `per_probe_fin`/`per_probe_tail` are
per-probe `(E, stalled)` NamedTuples (achieved error estimate and whether
that probe's own refinement stalled), so a caller can tell WHICH probe (if
any) drove the final node count or stalled -- `any_tail_stalled` is the OR
of every probe's own tail-stall flag.
"""
function adaptive_kx_grid(fs::Vector, kbreaks::Vector{Float64};
                          rtol::Float64=1e-6, atol::Float64=0.0,
                          kr_order::Int=7, maxevals::Int=10_000,
                          Leff::Float64=1.0, fallback_N::Int=250,
                          target_decay::Float64=25.0, kx_ratio_max::Float64=80.0, warn_stall::Bool=true)
    isempty(kbreaks) && error("adaptive_kx_grid: kbreaks cannot be empty")
    isempty(fs) && error("adaptive_kx_grid: fs cannot be empty")
    kmax = kbreaks[end]
    x, w, wg = cachedrule(Float64, kr_order)

    fin_breaks = Set{Float64}(kbreaks)
    push!(fin_breaks, 0.0)
    per_probe_fin = Vector{NamedTuple}(undef, length(fs))
    Ifin_by_probe = Vector{Any}(undef, length(fs))
    for (i, f) in enumerate(fs)
        Ifin, Efin, segfin = quadgk_segbuf(f, 0.0, kbreaks...; rtol=rtol, atol=atol, order=kr_order, maxevals=maxevals)
        stalled = Efin > max(atol, rtol * norm(Ifin))
        stalled && warn_stall && @warn "adaptive_kx_grid: probe $i's finite bands did not reach (rtol=$rtol, atol=$atol) within maxevals=$maxevals (achieved error estimate $Efin) -- see this function's own docstring's note on lossless poles."
        for s in segfin
            push!(fin_breaks, Float64(s.a))
            push!(fin_breaks, Float64(s.b))
        end
        per_probe_fin[i] = (E=Efin, stalled=stalled)
        Ifin_by_probe[i] = Ifin
    end
    fin_pts = sort(collect(fin_breaks))
    fin_segs = [(a=fin_pts[i], b=fin_pts[i+1]) for i in 1:length(fin_pts)-1]
    nodes_fin, wts_fin = _panels_to_nodes(fin_segs, x, w, wg)

    t_cap = _tail_t_cap(kmax, kx_ratio_max)
    t_breaks = Set{Float64}((0.0, t_cap))
    per_probe_tail = Vector{NamedTuple}(undef, length(fs))
    any_tail_stalled = false
    for (i, f) in enumerate(fs)
        g(t) = f(kmax + t / (1.0 - t)) * (1.0 / (1.0 - t))^2
        # 2026-09-08 (voxel-path speedup pass, tail-stall investigation): the
        # tail is, by construction, the ALREADY-DECAYING correction past
        # `kmax` -- typically small relative to this SAME probe's own
        # finite/propagating band (`Ifin_by_probe[i]`, just computed above).
        # Demanding `atol=0` (pure RELATIVE `rtol` on the tail's OWN, possibly
        # tiny, value) asks QuadGK to resolve that small correction to a far
        # tighter ABSOLUTE precision than the physical problem needs, and can
        # chase precision past the noise floor of evaluating `f` itself (a
        # full modal/Redheffer solve, not noise-free arithmetic) -- a classic
        # pure-relative-tolerance pathology (this project confirmed the tail
        # stall was real, not just slow -- see the project status memory),
        # not necessarily a sign of genuinely unresolvable content. Anchoring
        # the tail's own `atol` to `rtol * |this probe's finite-band value|`
        # instead says "resolve the tail correction to the SAME absolute
        # precision already demanded of this probe's dominant, finite-band
        # part" -- composes with a caller-supplied `atol` via `max`, never
        # loosens below it.
        atol_tail = max(atol, rtol * norm(Ifin_by_probe[i]))
        Itail, Etail, segtail = quadgk_segbuf(g, 0.0, t_cap; rtol=rtol, atol=atol_tail, order=kr_order, maxevals=maxevals)
        stalled = Etail > max(atol_tail, rtol * norm(Itail))
        any_tail_stalled |= stalled
        for s in segtail
            push!(t_breaks, Float64(s.a))
            push!(t_breaks, Float64(s.b))
        end
        per_probe_tail[i] = (E=Etail, stalled=stalled)
    end

    if any_tail_stalled
        @warn "adaptive_kx_grid: at least one probe's evanescent tail past kmax=$kmax did not reach (rtol=$rtol, atol=$atol) within maxevals=$maxevals -- falling back to the fixed cosh-substitution evanescent band (Leff=$Leff) for ALL probes (a shared node set), per this function's own documented fallback."
        umax = evanescent_umax(kmax, Leff; target_decay=target_decay, kx_ratio_max=kx_ratio_max)
        nodes_tail, wts_tail = _evanescent_band_nodes(kmax, umax, fallback_N)
    else
        t_pts = sort(collect(t_breaks))
        t_segs = [(a=t_pts[i], b=t_pts[i+1]) for i in 1:length(t_pts)-1]
        nodes_t, wts_t = _panels_to_nodes(t_segs, x, w, wg)
        nodes_tail = @. kmax + nodes_t / (1.0 - nodes_t)
        wts_tail   = @. wts_t / (1.0 - nodes_t)^2
    end

    kx_nodes = vcat(nodes_fin, nodes_tail)
    wt_nodes = vcat(wts_fin, wts_tail)

    # `min_panel_width` -- see this function's own docstring; computed off
    # the merged (union-of-probes) finite-band partition `fin_segs`.
    min_panel_width = isempty(fin_segs) ? Inf : minimum(s.b - s.a for s in fin_segs)

    info = (n_panels_fin=length(fin_segs), n_nodes=length(kx_nodes),
            per_probe_fin=per_probe_fin, per_probe_tail=per_probe_tail,
            any_tail_stalled=any_tail_stalled, min_panel_width=min_panel_width)
    return kx_nodes, wt_nodes, info
end


function build_kphi_grid(N_phi::Int)
    phi_nodes   = [2π * p / N_phi for p in 0:(N_phi - 1)]
    phi_weights = fill(2π / N_phi, N_phi)
    return phi_nodes, phi_weights
end


end
