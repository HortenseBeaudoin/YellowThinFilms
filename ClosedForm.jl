# ClosedForm.jl -- loads the closed-form (z,z')-separable solver, in dependency order.
#
#   include("ClosedForm.jl")
#
# Foundation: ModeSolver (Berreman eigenmodes), ScatteringMatrix (Redheffer),
# SourceHandling (Dipole/Sheet/ModalSeed), SourceProjection (point seed),
# FieldSolver (cavity resummation + stack walk, `_cavity_transfer_columns`),
# KxGrid (adaptive Gauss–Kronrod quadrature), UniformSource (uniform-DFT grid/FFT).
# Closed form: CavityTransfer (T^o caches, 1D and 2D), ClosedFormField (synthesis:
# point, box exterior/interior), ClosedFormAssembly (multi-source FFT assembly),
# ClosedFormGrid (point / box-aware adaptive grids).
# See PLAN_closed_form_point_source_rewrite.md and STATUS_closed_form_rewrite.md.

for f in ("mod_ModeSolver.jl", "mod_ScatteringMatrix.jl", "mod_Sources.jl", "mod_SourceProj.jl",
          "mod_FieldSolver.jl", "mod_KxGrid.jl", "mod_UniformSource.jl",
          "mod_CavityTransfer.jl", "mod_ClosedFormField.jl", "mod_ClosedFormAssembly.jl", "mod_ClosedFormGrid.jl")
    include(joinpath(@__DIR__, f))
end
using .ModeSolver, .ScatteringMatrix, .SourceHandling, .SourceProjection, .FieldSolver, .KxGrid, .UniformSource
using .CavityTransfer, .ClosedFormField, .ClosedFormAssembly, .ClosedFormGrid
