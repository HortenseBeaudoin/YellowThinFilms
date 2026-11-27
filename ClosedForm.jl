# ClosedForm.jl -- loads the closed-form (z,z')-separable solver, in dependency order.
#
#   include("ClosedForm.jl")
#
# Foundation: LayeredMaterials (ε/μ tensors, layers), BerremanModes (Berreman eigenmodes), ScatteringMatrices (Redheffer),
# Dipoles (Dipole), ExcitationModeProjections (point seed),
# StackResponses (cavity resummation + stack walk, `_cavity_transfer_columns`),
# SpectralSingularities (light lines, real-axis poles, cut-off: where the integrand has structure),
# QuadratureRules (engine choice, adaptive Gauss–Kronrod and graded Gauss–Legendre rules), CurrentSheets (Sheet + its uniform lateral pixel grid).
# Closed form: SpectralCachedResponses (T^o caches, 1D and 2D), SpectralIntegrals (synthesis:
# point, box exterior/interior), ClosedFormAssembly (multi-source FFT assembly),
# SpectralQuadratureGrids (point / box-aware adaptive grids), GreenOperator (precomputed
# z-structured operator of one source layer: Toeplitz/Hankel same layer, dense otherwise).
# See PLAN_closed_form_point_source_rewrite.md and STATUS_closed_form_rewrite.md.

for f in ("LayeredMaterials.jl", "BerremanModes.jl", "ScatteringMatrices.jl", "Dipoles.jl", "ExcitationModeProjections.jl",
          "StackResponses.jl", "SpectralSingularities.jl", "QuadratureRules.jl", "CurrentSheets.jl",
          "SpectralCachedResponses.jl", "SpectralIntegrals.jl", "mod_ClosedFormAssembly.jl", "SpectralQuadratureGrids.jl",
          "mod_GreenOperator.jl")
    include(joinpath(@__DIR__, f))
end
using .LayeredMaterials, .BerremanModes, .ScatteringMatrices, .Dipoles, .ExcitationModeProjections, .StackResponses, .SpectralSingularities, .QuadratureRules, .CurrentSheets
using .SpectralCachedResponses, .SpectralIntegrals, .ClosedFormAssembly, .SpectralQuadratureGrids, .GreenOperator
