include("ClosedForm.jl")
using LinearAlgebra, StaticArrays, Printf, Random

# ─────────────────────────────────────────────────────────────────────────────
# Gz_ClosedForm_Foundation_Validation.jl -- the §3 foundation, checked on the physics.
#
# `_cavity_transfer_columns` runs StackResponses' cavity resummation + stack walk
# on the 4 unit edge seeds, giving the z-independent 4×4 map T^o per obs layer.
# No second solver is used as reference: the amplitudes `T^o·[S₊;S₋]` (plus the
# direct term in the source layer) must satisfy the conditions that define the
# solution uniquely:
#   - tangential (Ex, Ey, Hx, Hy) = Vf·Ef(z) + Vb·Eb(z) continuous at every interface;
#   - nothing incoming from the half-spaces (no Ef in layer 1, no Eb in layer N,
#     apart from the source's own direct term);
#   - for `plane_wave_amp`, the incident amplitude found back at the stack face.
# The jump at the source plane is built into the direct term. Sources: the point
# seed of `Source_projection_Berreman_xyz` (x̂, ŷ, ẑ) at three depths of every
# finite and top layer, and plane waves from both sides. Isotropic, uniaxial,
# biaxial and gyrotropic stacks; propagating to evanescent kt.
# ─────────────────────────────────────────────────────────────────────────────

const FS = StackResponses
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

# Tangential field of layer k at local depth z from ModalAmp-convention amplitudes
# (Ef at z=0, Eb at z=d); also returns the size of the two terms, the roundoff scale.
function ψ(modes, stack, k, Ef, Eb, z)
    m = modes[k]
    tf = m.Vf * (Ef .* FS._SProp(z, m.Qf))
    tb = m.Vb * (Eb .* FS._SProp(z - stack[k].d, m.Qb))
    return tf + tb, norm(tf) + norm(tb)
end

# Worst continuity mismatch over all interfaces. `amp(k, side)` gives (Ef, Eb) of layer k
# valid at its edge `side` (:left = z=0, :right = z=d). Mismatch is relative to the larger
# of the local field and `floor` (fields far down an evanescent tail sit at roundoff).
function continuity_err(modes, stack, amp, floor)
    worst = 0.0
    for k in 1:length(stack)-1
        L, sL = ψ(modes, stack, k, amp(k, :right)..., stack[k].d)
        R, sR = ψ(modes, stack, k + 1, amp(k + 1, :left)..., 0.0)
        worst = max(worst, norm(L - R) / max(sL, sR, floor))
    end
    return worst
end

worst_pt = 0.0; worst_pt_rad = 0.0; worst_pw = 0.0; worst_pw_bc = 0.0; ncase = 0
for (label, stack) in stacks
    N = length(stack)
    for (ki, (kx, ky)) in enumerate(kts)
        modes = Modes_Stack(stack, ω, [kx, ky])
        Sint, Sprop = S_all_ingredients(stack, modes)
        Sall = S_all_Matrix(stack, modes, Sint, Sprop)

        # ── point seed in every layer but the last ──
        for layer in 1:(N - 1)
            d = stack[layer].d
            Qf, Qb = modes[layer].Qf, modes[layer].Qb
            sif, sib = Source_projection_Berreman_xyz([kx, ky], ω, stack[layer].ϵ, modes[layer])
            T = FS._cavity_transfer_columns(stack, layer, modes, Sall, Sint)
            for q in 1:3, zf in (0.13, 0.5, 0.91)
                global ncase += 1
                sf, sb = SVector{2}(sif[:, q]), SVector{2}(-sib[:, q])   # edge-seed convention: −sib
                zpos = layer == 1 ? d - zf * d : zf * d
                u = vcat(sf .* FS._SProp(d - zpos, Qf), sb .* FS._SProp(-zpos, Qb))
                A = Dict(o => (SVector{2}((T[o] * u)[1:2]), SVector{2}((T[o] * u)[3:4])) for o in 1:N)
                # source layer: add the direct term on the side of the edge asked for
                amp(k, side) = k != layer ? A[k] :
                    side === :right ? (A[k][1] + sf .* FS._SProp(-zpos, Qf), A[k][2]) :
                                      (A[k][1], A[k][2] + sb .* FS._SProp(d - zpos, Qb))
                scale = norm(vcat(sf, sb))
                global worst_pt = max(worst_pt, continuity_err(modes, stack, amp, 1e-12 * scale))
                # radiation: no reflected wave coming in from layer 1's or layer N's outer side
                global worst_pt_rad = max(worst_pt_rad, norm(A[1][1]) / scale, norm(A[N][2]) / scale)
            end
        end

        # ── plane wave from each side ──
        for side in (:top, :bottom)
            a = SVector{2,ComplexF64}(randn(Xoshiro(17ki + length(label)), ComplexF64, 2))
            P = plane_wave_amp(stack, modes, Sall, Sint, a; side = side)
            amp(k, _) = (P[k].Ef, P[k].Eb)
            global worst_pw = max(worst_pw, continuity_err(modes, stack, amp, 1e-12 * norm(a)))
            # incident wave at the stack face, nothing incoming from the other side
            inc, none = side === :top ?
                (P[1].Ef .* FS._SProp(stack[1].d, modes[1].Qf), P[N].Eb) :
                (P[N].Eb .* FS._SProp(-stack[N].d, modes[N].Qb), P[1].Ef)
            global worst_pw_bc = max(worst_pw_bc, norm(inc - a) / norm(a), norm(none) / norm(a))
        end
    end
end

@printf("point seed, T^o: interface continuity       worst rel err %.3e  (%d cases)\n", worst_pt, ncase)
@printf("point seed, T^o: no incoming wave           worst rel     %.3e\n", worst_pt_rad)
@printf("plane_wave_amp:  interface continuity       worst rel err %.3e\n", worst_pw)
@printf("plane_wave_amp:  incident at face / no back worst rel err %.3e\n", worst_pw_bc)
ok = max(worst_pt, worst_pt_rad, worst_pw, worst_pw_bc) < 1e-10
println(ok ? "FOUNDATION: PASS" : "FOUNDATION: FAIL")
