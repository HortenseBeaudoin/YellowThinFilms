# ClosedForm.jl -- loads the closed-form (z,z')-separable solver, in dependency order.
#
#   include("ClosedForm.jl")
#
# Foundation: Materials (ε/μ tensors, layers), ModeSolver (Berreman eigenmodes), ScatteringMatrix (Redheffer),
# SourceHandling (Dipole), SourceProjection (point seed),
# FieldSolver (cavity resummation + stack walk, `_cavity_transfer_columns`),
# Features (light lines, real-axis poles, cut-off: where the integrand has structure),
# Quadrature (engine choice, adaptive Gauss–Kronrod and graded Gauss–Legendre rules), Sheets (Sheet + its uniform lateral pixel grid).
# Closed form: CavityTransfer (T^o caches, 1D and 2D), ClosedFormField (synthesis:
# point, box exterior/interior), ClosedFormAssembly (multi-source FFT assembly),
# ClosedFormGrid (point / box-aware adaptive grids), GreenOperator (precomputed
# z-structured operator of one source layer: Toeplitz/Hankel same layer, dense otherwise).
# See PLAN_closed_form_point_source_rewrite.md and STATUS_closed_form_rewrite.md.

for f in ("mod_Materials.jl", "mod_ModeSolver.jl", "mod_ScatteringMatrix.jl", "mod_Sources.jl", "mod_SourceProj.jl",
          "mod_FieldSolver.jl", "mod_Features.jl", "mod_Quadrature.jl", "mod_Sheets.jl",
          "mod_CavityTransfer.jl", "mod_ClosedFormField.jl", "mod_ClosedFormAssembly.jl", "mod_ClosedFormGrid.jl",
          "mod_GreenOperator.jl")
    include(joinpath(@__DIR__, f))
end
using .Materials, .ModeSolver, .ScatteringMatrix, .SourceHandling, .SourceProjection, .FieldSolver, .Features, .Quadrature, .Sheets
using .CavityTransfer, .ClosedFormField, .ClosedFormAssembly, .ClosedFormGrid, .GreenOperator
