include("ClosedForm.jl")
using LinearAlgebra, StaticArrays, Printf

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Foundation_Validation.jl -- the §3 foundation extension.
#
# `_cavity_transfer_columns` runs FieldSolver's cavity resummation + stack walk
# on the 4 unit edge seeds, giving the z-independent 4×4 map T^o per obs layer.
# The physical seed [S₊;S₋] pushed through T^o must reproduce:
#   - `Modal_amp`'s Ef/Eb in every other layer (cross-layer);
#   - `Modal_amp_profile`'s reflected part in the source layer (its value at
#     z'=d minus the forward direct jumps, at z'=0 minus the backward ones).
# Roundoff, not bit-for-bit: T·u sums 4 columns in a different order. Isotropic,
# uniaxial, biaxial and gyrotropic stacks; propagating to evanescent kt.
# ─────────────────────────────────────────────────────────────────────────────

const FS = FieldSolver
lay(m, d) = FlatOpticalLayer(m, d)
m_vac = IsotropicMaterial(1.0 + 0.001im)
m_gl  = IsotropicMaterial(2.25 + 0.001im)
m_hi  = IsotropicMaterial(6.0 + 0.3im)
m_uni = UniaxialMaterial(2.0 + 0.01im, 3.5 + 0.01im)
m_bi  = BiaxialMaterial(2.0 + 0.01im, 3.0 + 0.01im, 4.0 + 0.01im)
m_gyr = GyrotropicMaterial(ComplexF64[4.0+0.01im 0.3im 0; -0.3im 4.0+0.01im 0; 0 0 3.0+0.01im])

stacks = [
    ("iso 3-layer vac/glass/vac", [lay(m_vac, 2.0), lay(m_gl, 2.0), lay(m_vac, 2.0)]),
    ("iso 5-layer lossy",        [lay(m_vac, 1.0), lay(m_gl, 0.7), lay(m_hi, 0.3), lay(m_gl, 1.1), lay(m_vac, 1.0)]),
    ("uniaxial 4-layer",         [lay(m_vac, 1.0), lay(m_uni, 0.8), lay(m_gl, 0.5), lay(m_vac, 1.0)]),
    ("biaxial 4-layer",          [lay(m_vac, 1.0), lay(m_bi, 0.8), lay(m_gl, 0.5), lay(m_vac, 1.0)]),
    ("gyrotropic 4-layer",       [lay(m_gyr, 1.0), lay(m_gl, 0.6), lay(m_gyr, 0.9), lay(m_vac, 1.0)]),
]
ω = 2π
kts = [(0.3, 0.0), (1.2, 0.4), (2.9, -1.1), (8.0, 3.0)]
relerr(a, b) = norm(a - b) / max(norm(b), 1e-300)

worst_cross = 0.0; worst_same = 0.0; ncase = 0
for (label, stack) in stacks
    N = length(stack)
    for (kx, ky) in kts
        modes = Modes_Stack(stack, ω, [kx, ky])
        Sint, Sprop = S_all_ingredients(stack, modes)
        Sall = S_all_Matrix(stack, modes, Sint, Sprop)
        for layer in 1:(N - 1)
            d = stack[layer].d
            sif, sib = Source_projection_Berreman_xyz([kx, ky], ω, stack[layer].ϵ, modes[layer])
            T = FS._cavity_transfer_columns(stack, layer, modes, Sall, Sint)
            _, _, _, Qf, Qb, _ = FS._R_cavity(stack, layer, modes, Sall)
            for q in 1:3, zf in (0.13, 0.5, 0.91)
                z = zf * d
                seed = ModalSeed(layer, z, sif[:, q], -sib[:, q])      # Modal_amp convention: −sib
                full = Modal_amp(stack, SourceDef[seed], layer, modes, Sall, Sint)
                S₊, S₋, _ = FS._seed_from_sources(SourceDef[seed], layer, N, Qf, Qb, d)
                u = vcat(S₊, S₋)
                for o in 1:N
                    global ncase += 1
                    v = T[o] * u
                    if o == layer
                        zpos = layer == 1 ? d - z : z
                        p0 = Modal_amp_profile(stack, SourceDef[seed], layer, modes, Sall, Sint, [0.0, d])
                        Ef_refl = p0[2].Ef - seed.sif .* FS._SProp(-zpos, Qf)
                        Eb_refl = p0[1].Eb - seed.sib .* FS._SProp(d - zpos, Qb)
                        e = norm(v - vcat(Ef_refl, Eb_refl)) / max(norm(vcat(p0[2].Ef, p0[1].Eb)), 1e-300)
                        global worst_same = max(worst_same, e)
                    else
                        global worst_cross = max(worst_cross, relerr(v, vcat(full[o].Ef, full[o].Eb)))
                    end
                end
            end
        end
    end
end

@printf("T^o·[S₊;S₋] vs Modal_amp (cross-layer):          worst rel err %.3e  (%d cases)\n", worst_cross, ncase)
@printf("T^o·[S₊;S₋] vs Modal_amp_profile reflected part: worst rel err %.3e\n", worst_same)
println(worst_cross < 1e-11 && worst_same < 1e-11 ? "FOUNDATION: PASS" : "FOUNDATION: FAIL")
