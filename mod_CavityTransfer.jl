"""
mod_CavityTransfer.jl -- the "Layer B" cache of the closed-form (z,z')-separable
rewrite (`PLAN_closed_form_point_source_rewrite.md` §1.2, §4.1).

For one fixed in-plane wavevector node `kt`, the modal amplitude of a point
source at depth `z` in `source_layer`, observed in layer `o`, is exactly

    [Ef_o; Eb_o](kt; z) = T^o(kt) · [ sf ∘ Ψf(z) ; sb ∘ Ψb(z) ]
    Ψf,n(z) = exp(i·Qf,src[n]·(d_src − zpos)),   Ψb,n(z) = exp(−i·Qb,src[n]·zpos)

with `T^o` a 4×4 matrix per node that depends on `(kt, source_layer, o)` only
(never `z`, `z'`, moment, or source shape) and `(sf,sb)` the z-independent
point seed of `Source_projection_Berreman_xyz`. `T^o` is built ONCE per node
from `FieldSolver._cavity_transfer_columns` (the §3 "minimal FieldSolver
extension", option 1 of the plan: FieldSolver's own resummation+walk run on
the 4 unit basis inputs -- no duplicated cavity code here). The observation
depth then enters only through `Φf = e^{iQf,obs·z'}`, `Φb = e^{iQb,obs·(z'−d_obs)}`
(`mod_ClosedFormField.jl`).

Conventions (fixed by the foundation modules):
  - `zpos = d − z` for `source_layer == 1`, `zpos = z` otherwise
    (`FieldSolver._seed_from_sources`).
  - the cached backward seed is `sb = −sib_raw`: `Source_projection_Berreman_xyz`'s
    raw `sib` has the opposite sign from the `sib` `Modal_amp` consumes (`σ_b = −1`,
    confirmed through `Ψ_p = −(iM)⁻¹Js/Δz`, plan §5.3). `sf, sb` below are the
    Modal_amp-convention seeds, one column per canonical moment x̂,ŷ,ẑ.
  - same-layer `T^o` holds the REFLECTED (cavity) part only; the direct term is
    added by the synthesis layer (`mod_ClosedFormField.jl`) with an explicit
    branch on the sign of `z' − zpos`.

Two caches share this construction (§4.1):
  - `CavityTransferCache1D` -- `kt=(kx,0)` radial nodes; point source on an
    isotropic/uniaxial stack only (the J0/J1/J2 fast path).
  - `CavityTransferCache2D` -- `(kρ,φ)` nodes; everything else (box on any
    medium, point on biaxial/gyrotropic). Two fill strategies, by
    `_is_azimuthally_symmetric_stack`: one `Modes_Stack` solve per kρ at φ=0
    plus the exact `_rotate_modes` rotation for every other φ (`T^o` is then
    φ-independent: it depends on the modes only through `Qf,Qb` and
    `Sall/Sint`, all invariant under `Rot4(φ)`), or an independent solve at
    every `(kρ,φ)` node.

Quadrature nodes are an INPUT here: `mod_ClosedFormGrid.jl` builds them
(`point_kx_grid` for the 1D cache, `box_kρ_grid` for the 2D one), or pass any
rule of your own.
"""
module CavityTransfer

export AbstractCavityTransferCache, CavityTransferCache1D, build_cavity_transfer_1d
export CavityTransferCache2D, build_cavity_transfer_2d
export _mode_field_core, _mode_field_eps

import ..ModeSolver: AbstractLayer, Modes, Modes_Stack, _is_azimuthally_symmetric_stack, _rotate_modes
import ..ScatteringMatrix: S_all_ingredients, S_all_Matrix
import ..SourceProjection: Source_projection_Berreman_xyz
import ..FieldSolver: _cavity_transfer_columns

using LinearAlgebra
using StaticArrays
using Base.Threads

const _T44 = SMatrix{4,4,ComplexF64,16}
const _S23 = SMatrix{2,3,ComplexF64,6}

@inline _isfinitec(z::Complex) = isfinite(real(z)) && isfinite(imag(z))
@inline _allfinite(A) = all(_isfinitec, A)

# ─────────────────────────────────────────────────────────────────────────────
#  Mode field basis
# ─────────────────────────────────────────────────────────────────────────────

"""
Layer-only-dependent ϵ-derived quantities `_mode_field_core` needs (`ϵzt`,
`ϵzz`).
"""
function _mode_field_eps(layer::AbstractLayer)
    ϵ = layer.ϵ
    return SVector{2,ComplexF64}(ϵ[3,1], ϵ[3,2]), ComplexF64(ϵ[3,3])
end

"""
Mode field `(Ex, Ey, Ez)` and `kz` of mode `m` in direction `dir ∈ (:f,:b)`.
`Ez` is built from the mode vector
only (no local `Jz/(ωϵzz)` term -- see the plan's §5.3; that term is added
separately by the interior-slab routine, never here). `ϵzt[i]*V[i]` (not
`dot`) is deliberate: a plain bilinear sum, no conjugation.
"""
function _mode_field_core(ϵzt::SVector{2,ComplexF64}, ϵzz::ComplexF64,
                           modes::Modes, ω::Float64, kvec::SVector{2,ComplexF64},
                           dir::Symbol, m::Int)
    if dir === :f
        kz = ComplexF64(modes.Qf[m])
        V  = modes.Vf[:, m]
    elseif dir === :b
        kz = ComplexF64(modes.Qb[m])
        V  = modes.Vb[:, m]
    else
        error("dir must be :f or :b")
    end

    Ex_mode = ComplexF64(V[1])
    Ey_mode = ComplexF64(V[2])
    Ez_mode = -(ϵzt[1]*V[1] + ϵzt[2]*V[2] + (1/ω)*(kvec[1]*V[4] - kvec[2]*V[3])) / ϵzz

    return Ex_mode, Ey_mode, ComplexF64(Ez_mode), kz
end

"""
Per-observation-layer mode field basis on a node set, `Nk×2` matrices
(node, mode): `Exf,Eyf,Ezf,Exb,Eyb,Ezb` and the obs layer's own `KZf,KZb`
(= `Qf_obs, Qb_obs`), plus `d` (= `d_obs`). A node whose basis is not finite
is left at zero.
"""
function _obs_basis(dims::Int...)
    z() = zeros(ComplexF64, dims..., 2)
    return (Exf=z(), Eyf=z(), Ezf=z(), Exb=z(), Eyb=z(), Ezb=z(), KZf=z(), KZb=z())
end

function _fill_obs_basis!(B, k, layer::AbstractLayer, modes::Modes, ω::Float64,
                          kvec::SVector{2,ComplexF64})
    ϵzt, ϵzz = _mode_field_eps(layer)
    for m in 1:2
        exf, eyf, ezf, kzf = _mode_field_core(ϵzt, ϵzz, modes, ω, kvec, :f, m)
        exb, eyb, ezb, kzb = _mode_field_core(ϵzt, ϵzz, modes, ω, kvec, :b, m)
        all(_isfinitec, (exf, eyf, ezf, exb, eyb, ezb, kzf, kzb)) || continue
        B.Exf[k..., m] = exf; B.Eyf[k..., m] = eyf; B.Ezf[k..., m] = ezf
        B.Exb[k..., m] = exb; B.Eyb[k..., m] = eyb; B.Ezb[k..., m] = ezb
        B.KZf[k..., m] = kzf; B.KZb[k..., m] = kzb
    end
    return B
end

"Common supertype of the 1D and 2D transfer caches."
abstract type AbstractCavityTransferCache end

# ─────────────────────────────────────────────────────────────────────────────
#  1D cache: kt = (kx, 0), point source, isotropic/uniaxial stack
# ─────────────────────────────────────────────────────────────────────────────

"""
    CavityTransferCache1D

Everything the closed-form synthesis needs, per radial node `k` of a 1D
`kt=(kx,0)` quadrature, for one `source_layer` and a set of `obs_layers`:

  - `T[o][k]`       -- 4×4 `T^o` (see module docstring), per obs layer `o`
  - `Qf_src[k]`, `Qb_src[k]`, `d_src` -- what `Ψ(z)` needs
  - `sf[k]`, `sb[k]` -- 2×3 point seeds (Modal_amp convention, `sb = −sib_raw`),
                       column q = canonical moment x̂,ŷ,ẑ
  - `basis[o]`      -- mode field basis + `KZf/KZb` (= `Qf_obs,Qb_obs`) of layer `o`
  - `d_obs[o]`
  - `ϵ_src`         -- source layer's 3×3 ϵ (§5.3 interior term, later phases)
  - `valid[k]`      -- false where the mode/S-matrix/projection solve failed or
                       returned a non-finite seed (node contributes nothing at all)
  - `Tvalid[o][k]`  -- false where `T^o` itself is non-finite; only layer `o`'s
                       reflected/transmitted term drops that node (the same-layer
                       direct term still uses it)

Built by `build_cavity_transfer_1d`. Contains no `z`, `z'`, moment, or shape.
"""
struct CavityTransferCache1D{L<:AbstractLayer} <: AbstractCavityTransferCache
    stack        :: Vector{L}
    ω            :: Float64
    source_layer :: Int
    obs_layers   :: Vector{Int}
    kx_nodes     :: Vector{Float64}
    wt_nodes     :: Vector{Float64}
    valid        :: Vector{Bool}
    Tvalid       :: Dict{Int,Vector{Bool}}
    T            :: Dict{Int,Vector{_T44}}
    Qf_src       :: Vector{SVector{2,ComplexF64}}
    Qb_src       :: Vector{SVector{2,ComplexF64}}
    d_src        :: Float64
    sf           :: Vector{_S23}
    sb           :: Vector{_S23}
    basis        :: Dict{Int,NamedTuple}
    d_obs        :: Dict{Int,Float64}
    ϵ_src        :: SMatrix{3,3,ComplexF64,9}
end

"""
    build_cavity_transfer_1d(stack, ω, source_layer, kx_nodes, wt_nodes;
                             obs_layers=1:length(stack), check_symmetry=true)
        -> CavityTransferCache1D

Per node: `Modes_Stack(stack, ω, kx)` → `S_all_ingredients`/`S_all_Matrix` →
`_cavity_transfer_columns` (T^o for every obs layer at once) →
`Source_projection_Berreman_xyz` (point seed) → mode field basis per obs
layer. `O(Nk)` stack solves, once per `(stack, ω, source_layer)`; every source
depth, observation depth and moment is then served from this cache with no
further solve.

The 1D path is valid only for a POINT source on an azimuthally symmetric
(isotropic/uniaxial, isotropic μ) stack; `check_symmetry=true` enforces the
stack half of that.
"""
function build_cavity_transfer_1d(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int,
                                  kx_nodes::AbstractVector{<:Real}, wt_nodes::AbstractVector{<:Real};
                                  obs_layers = 1:length(stack), check_symmetry::Bool = true)
    N = length(stack)
    (1 <= source_layer < N) || error("build_cavity_transfer_1d: source_layer=$source_layer must be in 1:$(N-1) " *
                                     "(the last, semi-infinite layer cannot host a source).")
    length(kx_nodes) == length(wt_nodes) || error("build_cavity_transfer_1d: kx_nodes/wt_nodes length mismatch")
    check_symmetry && !_is_azimuthally_symmetric_stack(stack) &&
        error("build_cavity_transfer_1d: the 1D path requires an isotropic/uniaxial stack " *
              "(`_is_azimuthally_symmetric_stack`); use the 2D (kρ,φ) cache for biaxial/gyrotropic.")

    ωf  = Float64(ω)
    kxs = collect(Float64, kx_nodes)
    wts = collect(Float64, wt_nodes)
    obs = sort!(unique!(collect(Int, obs_layers)))
    all(o -> 1 <= o <= N, obs) || error("build_cavity_transfer_1d: obs_layers out of range 1:$N")
    Nk  = length(kxs)

    valid  = fill(false, Nk)
    Tvalid = Dict(o => fill(false, Nk) for o in obs)
    T      = Dict(o => fill(zero(_T44), Nk) for o in obs)
    Qf_src = fill(zero(SVector{2,ComplexF64}), Nk)
    Qb_src = fill(zero(SVector{2,ComplexF64}), Nk)
    sf     = fill(zero(_S23), Nk)
    sb     = fill(zero(_S23), Nk)
    basis  = Dict{Int,NamedTuple}(o => _obs_basis(Nk) for o in obs)
    ϵ_src  = SMatrix{3,3,ComplexF64}(stack[source_layer].ϵ[i, j] for i in 1:3, j in 1:3)

    @threads :static for k in 1:Nk
        kx = kxs[k]
        try
            modes = Modes_Stack(stack, ωf, kx)
            Sint, Sprop = S_all_ingredients(stack, modes)
            Sall = S_all_Matrix(stack, modes, Sint, Sprop)
            Tk = _cavity_transfer_columns(stack, source_layer, modes, Sall, Sint; obs_layers=obs)
            sif, sib = Source_projection_Berreman_xyz([kx; 0.0], ωf, stack[source_layer].ϵ, modes[source_layer])
            (_allfinite(sif) && _allfinite(sib)) || continue

            for o in obs
                if _allfinite(Tk[o])
                    T[o][k] = Tk[o]
                    Tvalid[o][k] = true
                end
            end
            Qf_src[k] = modes[source_layer].Qf
            Qb_src[k] = modes[source_layer].Qb
            sf[k] = _S23(sif)
            sb[k] = -_S23(sib)
            kvec = SVector{2,ComplexF64}(kx, 0.0)
            for o in obs
                _fill_obs_basis!(basis[o], (k,), stack[o], modes[o], ωf, kvec)
            end
            valid[k] = true
        catch e
            e isa Union{LinearAlgebra.SingularException, ErrorException, DomainError} || rethrow()
        end
    end

    d_obs = Dict(o => Float64(stack[o].d) for o in obs)
    return CavityTransferCache1D(stack, ωf, source_layer, obs, kxs, wts, valid, Tvalid, T,
                                 Qf_src, Qb_src, Float64(stack[source_layer].d), sf, sb,
                                 basis, d_obs, ϵ_src)
end


# ─────────────────────────────────────────────────────────────────────────────
#  2D cache: (kρ, φ) nodes -- box on any medium, point on biaxial/gyrotropic
# ─────────────────────────────────────────────────────────────────────────────

"""
    CavityTransferCache2D

Same content as `CavityTransferCache1D`, per `(kρ,φ)` node: every per-node
field is an `Nk×Np` matrix (`basis[o]` arrays are `Nk×Np×2`), node
`(k,p)` ↔ `kt = (kρ_k cosφ_p, kρ_k sinφ_p)`. `rotation` records which fill
strategy was used (see module docstring).
"""
struct CavityTransferCache2D{L<:AbstractLayer} <: AbstractCavityTransferCache
    stack        :: Vector{L}
    ω            :: Float64
    source_layer :: Int
    obs_layers   :: Vector{Int}
    kρ_nodes     :: Vector{Float64}
    kρ_weights   :: Vector{Float64}
    phi_nodes    :: Vector{Float64}
    phi_weights  :: Vector{Float64}
    rotation     :: Bool
    valid        :: Matrix{Bool}
    Tvalid       :: Dict{Int,Matrix{Bool}}
    T            :: Dict{Int,Matrix{_T44}}
    Qf_src       :: Matrix{SVector{2,ComplexF64}}
    Qb_src       :: Matrix{SVector{2,ComplexF64}}
    d_src        :: Float64
    sf           :: Matrix{_S23}
    sb           :: Matrix{_S23}
    basis        :: Dict{Int,NamedTuple}
    d_obs        :: Dict{Int,Float64}
    ϵ_src        :: SMatrix{3,3,ComplexF64,9}
end

"""
    build_cavity_transfer_2d(stack, ω, source_layer, kρ_nodes, kρ_weights, phi_nodes, phi_weights;
                             obs_layers=1:length(stack), rotation=:auto) -> CavityTransferCache2D

`rotation=:auto` uses the φ-rotation fill iff `_is_azimuthally_symmetric_stack(stack)`;
`true` forces it (error on a non-symmetric stack), `false` forces an independent
solve at every node (useful to validate the rotation branch). The rotation branch
solves at φ=0 and applies `_rotate_modes(modes0, φ_p)` for every node, so it needs
no assumption on `phi_nodes[1]`. The point seed is recomputed at every node from
the ACTUAL `kt` and the rotated modes (cheap: one 4×4 solve).
"""
function build_cavity_transfer_2d(stack::Vector{<:AbstractLayer}, ω::Real, source_layer::Int,
                                  kρ_nodes::AbstractVector{<:Real}, kρ_weights::AbstractVector{<:Real},
                                  phi_nodes::AbstractVector{<:Real}, phi_weights::AbstractVector{<:Real};
                                  obs_layers = 1:length(stack), rotation = :auto)
    N = length(stack)
    (1 <= source_layer < N) || error("build_cavity_transfer_2d: source_layer=$source_layer must be in 1:$(N-1) " *
                                     "(the last, semi-infinite layer cannot host a source).")
    length(kρ_nodes) == length(kρ_weights) || error("build_cavity_transfer_2d: kρ nodes/weights length mismatch")
    length(phi_nodes) == length(phi_weights) || error("build_cavity_transfer_2d: φ nodes/weights length mismatch")
    sym = _is_azimuthally_symmetric_stack(stack)
    use_rot = rotation === :auto ? sym : Bool(rotation)
    use_rot && !sym && error("build_cavity_transfer_2d: rotation=true requires an azimuthally symmetric stack")

    ωf  = Float64(ω)
    kρs = collect(Float64, kρ_nodes); wρ = collect(Float64, kρ_weights)
    φs  = collect(Float64, phi_nodes); wφ = collect(Float64, phi_weights)
    obs = sort!(unique!(collect(Int, obs_layers)))
    all(o -> 1 <= o <= N, obs) || error("build_cavity_transfer_2d: obs_layers out of range 1:$N")
    Nk, Np = length(kρs), length(φs)

    valid  = fill(false, Nk, Np)
    Tvalid = Dict(o => fill(false, Nk, Np) for o in obs)
    T      = Dict(o => fill(zero(_T44), Nk, Np) for o in obs)
    Qf_src = fill(zero(SVector{2,ComplexF64}), Nk, Np)
    Qb_src = fill(zero(SVector{2,ComplexF64}), Nk, Np)
    sf     = fill(zero(_S23), Nk, Np)
    sb     = fill(zero(_S23), Nk, Np)
    basis  = Dict{Int,NamedTuple}(o => _obs_basis(Nk, Np) for o in obs)
    ϵ_src  = SMatrix{3,3,ComplexF64}(stack[source_layer].ϵ[i, j] for i in 1:3, j in 1:3)
    ok(e) = e isa Union{LinearAlgebra.SingularException, ErrorException, DomainError}

    # Fill node (k,p) from an already-solved, continuity-matched mode set and its T^o.
    function fill_node!(k, p, modes, Tk)
        kρ, φ = kρs[k], φs[p]
        kx, ky = kρ * cos(φ), kρ * sin(φ)
        sif, sib = Source_projection_Berreman_xyz([kx; ky], ωf, stack[source_layer].ϵ, modes[source_layer])
        (_allfinite(sif) && _allfinite(sib)) || return
        for o in obs
            if _allfinite(Tk[o])
                T[o][k, p] = Tk[o]
                Tvalid[o][k, p] = true
            end
        end
        Qf_src[k, p] = modes[source_layer].Qf
        Qb_src[k, p] = modes[source_layer].Qb
        sf[k, p] = _S23(sif)
        sb[k, p] = -_S23(sib)
        kvec = SVector{2,ComplexF64}(kx, ky)
        for o in obs
            _fill_obs_basis!(basis[o], (k, p), stack[o], modes[o], ωf, kvec)
        end
        valid[k, p] = true
    end

    if use_rot
        @threads :static for k in 1:Nk
            try
                modes0 = Modes_Stack(stack, ωf, kρs[k], 0.0)
                Sint, Sprop = S_all_ingredients(stack, modes0)
                Sall = S_all_Matrix(stack, modes0, Sint, Sprop)
                Tk = _cavity_transfer_columns(stack, source_layer, modes0, Sall, Sint; obs_layers=obs)
                for p in 1:Np
                    try
                        fill_node!(k, p, _rotate_modes(modes0, φs[p]), Tk)
                    catch e
                        ok(e) || rethrow()
                    end
                end
            catch e
                ok(e) || rethrow()
            end
        end
    else
        @threads :static for idx in 1:(Nk * Np)
            k = (idx - 1) ÷ Np + 1
            p = (idx - 1) % Np + 1
            try
                modes = Modes_Stack(stack, ωf, kρs[k], φs[p])
                Sint, Sprop = S_all_ingredients(stack, modes)
                Sall = S_all_Matrix(stack, modes, Sint, Sprop)
                Tk = _cavity_transfer_columns(stack, source_layer, modes, Sall, Sint; obs_layers=obs)
                fill_node!(k, p, modes, Tk)
            catch e
                ok(e) || rethrow()
            end
        end
    end

    d_obs = Dict(o => Float64(stack[o].d) for o in obs)
    return CavityTransferCache2D(stack, ωf, source_layer, obs, kρs, wρ, φs, wφ, use_rot, valid, Tvalid, T,
                                 Qf_src, Qb_src, Float64(stack[source_layer].d), sf, sb, basis, d_obs, ϵ_src)
end

end
