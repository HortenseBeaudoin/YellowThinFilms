module SpectralCachedResponses

export AbstractCavityTransferCache, CavityTransferCache1D, build_cavity_transfer_1d
export CavityTransferCache2D, build_cavity_transfer_2d
export _mode_field_core, _mode_field_eps

import ..LayeredMaterials: AbstractLayer, _is_azimuthally_symmetric_stack
import ..BerremanModes: Modes, Modes_Stack, _rotate_modes
import ..ScatteringMatrices: S_all_ingredients, S_all_Matrix
import ..ExcitationModeProjections: Source_projection_Berreman_xyz
import ..StackResponses: _cavity_transfer_columns

using LinearAlgebra
using StaticArrays
using Base.Threads

# ─────────────────────────────────────────────────────────────────────────────
#  Constant and structures declaration
# ─────────────────────────────────────────────────────────────────────────────

const _T44 = SMatrix{4,4,ComplexF64,16}
const _S23 = SMatrix{2,3,ComplexF64,6}

"Common supertype of the 1D and 2D transfer caches."
abstract type AbstractCavityTransferCache end

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────


@inline _isfinitec(z::Complex) = isfinite(real(z)) && isfinite(imag(z))
@inline _allfinite(A) = all(_isfinitec, A)

"""
ϵ-derived quantities for Ez computation (`ϵzt`, `ϵzz`).
"""
function _mode_field_eps(layer::AbstractLayer)
    ϵ = layer.ϵ
    return SVector{2,ComplexF64}(ϵ[3,1], ϵ[3,2]), ComplexF64(ϵ[3,3])
end

"""
Mode field `(Ex, Ey, Ez)` and `kz` of mode `m` in direction `dir ∈ (:f,:b)`.
`Ez` is built from the mode vector only. 
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
Per-observation-layer mode field basis on a node set. `Nkx2` matrices
(node, mode): `Exf,Eyf,Ezf,Exb,Eyb,Ezb` and the obs layer's own `KZf,KZb`.
No fixed size, depends on the number of observation layers. 
"""
function _obs_basis(dims::Int...)
    z() = zeros(ComplexF64, dims..., 2)
    return (Exf=z(), Eyf=z(), Ezf=z(), Exb=z(), Eyb=z(), Ezb=z(), KZf=z(), KZb=z())
end

"""
Fills a basis for a given kₜ and observation layer.
"""
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

# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

# TODO Length of the stack suddenly L, where in other modules its N ? A reason for that ? 
# TODO Uniformize the kx/kt/krho notation everywhere perhaps. The more general is to use kt. 
#  1D cache: kt = (kx, 0), point source, isotropic/uniaxial stack
"""
    CavityTransferCache1D(stack, ω, source_layer, obs_layers, kx_nodes, wt_nodes, valid,
                          Tvalid, T, Qf_src, Qb_src, d_src, sf, sb, basis, d_obs, ϵ_src)

Values needed for the closed-form, per radial node `k` of a 1D `kt=(kx,0)` quadrature.

# Fields
- `stack::Vector{L}`: vector of length L, with L the number of layers.
- `ω::Float64`: frequency. 
- `source_layer::Int`: source layer id.
- `obs_layers::Vector{Int}`: vector with all the observation layer's id. 
- `kx_nodes::Vector{Float64}`: integration kₜ nodes.
- `wt_nodes::Vector{Float64}`: integration weight of each node.
- `valid::Vector{Bool}`: false when the mode/S-matrix/projection solve failed or
  returned a non-finite seed
- `Tvalid::Dict{Int,Vector{Bool}}`: false when `T^o` is non-finite. 
- `T::Dict{Int,Vector{_T44}}`: 4x4 `T^o` per obs layer `o`. See doc for details.
- `Qf_src::Vector{SVector{2,ComplexF64}}`: eigenvalues of the forward modes in the source layer.
- `Qb_src::Vector{SVector{2,ComplexF64}}`: eigenvalues of the backward modes in the source layer.
- `d_src::Float64`: thickness of the source layer. 
- `sf::Vector{_S23}`: 2x3 forward seed for each unit moment. 
- `sb::Vector{_S23}`: 2x3 backward seed for each unit moment. 
- `basis::Dict{Int,NamedTuple}`: basis [ex ey ez] + kxf/kxb of the observation layer.
- `d_obs::Dict{Int,Float64}`: thickness of the observation layer.
- `ϵ_src::SMatrix{3,3,ComplexF64,9}`: permittivity tensor of the source layer. 

Built by `build_cavity_transfer_1d`. Contains no `z`, `z'`, moment, or shape.
"""
struct CavityTransferCache1D{L<:AbstractLayer} <: AbstractCavityTransferCache
    stack::Vector{L}
    ω::Float64
    source_layer::Int
    obs_layers::Vector{Int}
    kx_nodes::Vector{Float64}
    wt_nodes::Vector{Float64}
    valid::Vector{Bool}
    Tvalid::Dict{Int,Vector{Bool}}
    T::Dict{Int,Vector{_T44}}
    Qf_src::Vector{SVector{2,ComplexF64}}
    Qb_src::Vector{SVector{2,ComplexF64}}
    d_src::Float64
    sf::Vector{_S23}
    sb::Vector{_S23}
    basis::Dict{Int,NamedTuple}
    d_obs::Dict{Int,Float64}
    ϵ_src::SMatrix{3,3,ComplexF64,9}
end

# TODO : uniformize source_layer vs src_layer vs other notation throughout the whole code. 
"""
    build_cavity_transfer_1d(stack, ω, source_layer, kx_nodes, wt_nodes;
                             obs_layers=1:length(stack), check_symmetry=true)
        -> CavityTransferCache1D

Calls, per mode, `Modes_Stack(stack, ω, kx)` → `S_all_ingredients`/`S_all_Matrix` →
`_cavity_transfer_columns` (T^o for every obs layer at once) →
`Source_projection_Berreman_xyz` (point seed) → mode field basis per obs
layer.

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length L, with L the number of layers.
- `ω::Real`: frequency. 
- `source_layer::Int`: source layer id.
- `kx_nodes::AbstractVector{<:Real}`: integration kₜ nodes.
- `wt_nodes::AbstractVector{<:Real}`: integration weight of each node.

# Returns 
- `CavityTransferCache1D`: see struct description.
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

    @threads for k in 1:Nk
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
    CavityTransferCache2D(stack, ω, source_layer, obs_layers, kρ_nodes, wρ_nodes, phi_nodes,
                          phi_weights, rotation, valid, Tvalid, T, Qf_src, Qb_src, d_src, sf, 
                          sb, basis, d_obs, ϵ_src)

Values needed for the closed-form, per radial node `k` of a 1D `kt=(kx,0)` quadrature.

# Fields
- `stack::Vector{L}`: vector of length L, with L the number of layers.
- `ω::Float64`: frequency. 
- `source_layer::Int`: source layer id.
- `obs_layers::Vector{Int}`: vector with all the observation layer's id. 
- `kρ_nodes::Vector{Float64}`: integration kₜ nodes.
- `wρ_nodes::Vector{Float64}`: integration weight of each kₜ node.
- `phi_nodes::Vector{Float64}`: integration φ nodes.
- `phi_weights::Vector{Float64}`: integration weight of each φ node.
- `rotation::Bool`: yes if azimuthally cylindric stack and a single solve per kₜ. 
- `valid::Vector{Bool}`: false when the mode/S-matrix/projection solve failed or
  returned a non-finite seed
- `Tvalid::Dict{Int,Vector{Bool}}`: false when `T^o` is non-finite. 
- `T::Dict{Int,Vector{_T44}}`: 4x4 `T^o` per obs layer `o`. See doc for details.
- `Qf_src::Vector{SVector{2,ComplexF64}}`: eigenvalues of the forward modes in the source layer.
- `Qb_src::Vector{SVector{2,ComplexF64}}`: eigenvalues of the backward modes in the source layer.
- `d_src::Float64`: thickness of the source layer. 
- `sf::Vector{_S23}`: 2x3 forward seed for each unit moment. 
- `sb::Vector{_S23}`: 2x3 backward seed for each unit moment. 
- `basis::Dict{Int,NamedTuple}`: basis [ex ey ez] + kxf/kxb of the observation layer.
- `d_obs::Dict{Int,Float64}`: thickness of the observation layer.
- `ϵ_src::SMatrix{3,3,ComplexF64,9}`: permittivity tensor of the source layer. 

Built by `build_cavity_transfer_2d`. Contains no `z`, `z'`, moment, or shape.
"""
struct CavityTransferCache2D{L<:AbstractLayer} <: AbstractCavityTransferCache
    stack:: Vector{L}
    ω::Float64
    source_layer::Int
    obs_layers::Vector{Int}
    kρ_nodes::Vector{Float64}
    kρ_weights::Vector{Float64}
    phi_nodes::Vector{Float64}
    phi_weights::Vector{Float64}
    rotation::Bool
    valid::Matrix{Bool}
    Tvalid::Dict{Int,Matrix{Bool}}
    T::Dict{Int,Matrix{_T44}}
    Qf_src::Matrix{SVector{2,ComplexF64}}
    Qb_src::Matrix{SVector{2,ComplexF64}}
    d_src::Float64
    sf::Matrix{_S23}
    sb::Matrix{_S23}
    basis::Dict{Int,NamedTuple}
    d_obs::Dict{Int,Float64}
    ϵ_src::SMatrix{3,3,ComplexF64,9}
end

"""
    build_cavity_transfer_2d(stack, ω, source_layer, kρ_nodes, kρ_weights, phi_nodes, phi_weights;
                             obs_layers=1:length(stack), rotation=:auto) -> CavityTransferCache2D

Calls, per mode, `Modes_Stack(stack, ω, kx)` → `S_all_ingredients`/`S_all_Matrix` →
`_cavity_transfer_columns` (T^o for every obs layer at once) →
`Source_projection_Berreman_xyz` (point seed) → mode field basis per obs
layer. If the stack is azimuthally symmetric, a single solve per kρ and the other 
results are obtained by rotation. 

# Arguments
- `stack::Vector{<:AbstractLayer}`: vector of length L, with L the number of layers.
- `ω::Real`: frequency. 
- `source_layer::Int`: source layer id.
- `kρ_nodes::AbstractVector{<:Real}`: integration kₜ nodes.
- `kρ_weights::AbstractVector{<:Real}`: integration weight of each node.
- `phi_nodes::Vector{Float64}`: integration φ nodes.
- `phi_weights::Vector{Float64}`: integration weight of each φ node.

# Returns 
- `CavityTransferCache2D`: see struct description.
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
        @threads for k in 1:Nk
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
        @threads for idx in 1:(Nk * Np)
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
