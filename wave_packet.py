#!/usr/bin/env python3
"""
Implementation of Wavepacket Preparation and Scattering in Ising Field Theory
Based on Farrell et al. (arXiv:2505.03111v2)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter


def get_ising_hamiltonian_obc(L, gx, gz):
    """
    Constructs the Hamiltonian for 1D Ising Field Theory with Open Boundary Conditions.

    Rotated Hamiltonian (X <-> Z):
    H_sim = -0.5 * sum_{n=0}^{L-2} (X_n X_{n+1}) - gx * sum_{n=0}^{L-1} Z_n + gz * sum_{n=0}^{L-1} X_n
    """
    op_list = []

    # Interaction term: -0.5 * X_n X_{n+1}
    for n in range(L - 1):
        label = ["I"] * L
        label[L - 1 - n] = "X"
        label[L - 1 - (n + 1)] = "X"
        op_list.append(("".join(label), -0.5))

    # Transverse field (Mass): -gx * Z_n
    for n in range(L):
        label = ["I"] * L
        label[L - 1 - n] = "Z"
        op_list.append(("".join(label), -gx))

    # Longitudinal field: -gz * X_n
    for n in range(L):
        label = ["I"] * L
        label[L - 1 - n] = "X"
        op_list.append(("".join(label), -gz))

    return SparsePauliOp.from_list(op_list)


def prepare_w_state_circuit(n_qubits, qubits, k0, x0, sigma):
    """
    Implements the unitary circuit to prepare a Wavepacket (W-state-like).
    """
    qc = QuantumCircuit(n_qubits)

    d = len(qubits)
    coeffs = []
    normalization = 0.0

    for n in range(d):
        c = np.exp(-((n - x0)**2) / (4 * sigma**2))
        coeffs.append(c)
        normalization += c**2

    coeffs = np.array(coeffs) / np.sqrt(normalization)

    # 1. Initialize excitation at the first qubit of the subset
    qc.x(qubits[0])

    current_amp = 1.0

    for i in range(d - 1):
        q_src = qubits[i]
        q_dst = qubits[i+1]

        target_c = coeffs[i]

        if abs(current_amp) < 1e-9:
            theta = 0
        else:
            ratio = target_c / current_amp
            if ratio > 1.0:
                ratio = 1.0
            if ratio < -1.0:
                ratio = -1.0
            theta = 2 * np.arccos(ratio)

        current_amp = current_amp * np.sin(theta/2)

        # RBS Gate equivalent: CRY + CNOT
        qc.cry(theta, q_src, q_dst)
        qc.cx(q_dst, q_src)

    # 3. Apply Phases (Momentum k0)
    for n in range(d):
        phase = k0 * n
        qc.p(phase, qubits[n])

    return qc


def run_simulation():
    # --- Parameters ---
    L = 18

    # IFT Parameters from Paper (Section III)
    gx = 1.25
    gz = 0.15

    # Wavepacket Parameters
    k0 = 0.32 * np.pi
    sigma = 1.13

    mid = L // 2
    x_L = mid - 5
    x_R = mid + 5

    qubits_L = list(range(0, mid))
    qubits_R = list(range(mid, L))

    # Adjust centers relative to subsets
    x_L_local = x_L
    x_R_local = x_R - mid

    # --- Hamiltonian ---
    H_op = get_ising_hamiltonian_obc(L, gx, gz)

    # --- State Preparation ---
    qc = QuantumCircuit(L)

    qc_L = prepare_w_state_circuit(L, qubits_L, k0, x_L_local, sigma)
    qc.compose(qc_L, inplace=True)

    qc_R = prepare_w_state_circuit(L, qubits_R, -k0, x_R_local, sigma)
    qc.compose(qc_R, inplace=True)

    # --- Time Evolution ---
    psi = Statevector(qc)

    t_max = 25.0
    dt = 1/8
    steps = int(t_max / dt)

    density_profile = []

    print(f"Starting simulation on L={L} sites...")
    print(f"Hamiltonian: Ising Field Theory (OBC), gx={gx}, gz={gz}")
    print(f"Wavepackets: k0={k0/np.pi:.2f}pi, sigma={sigma}")

    # Evolution Circuit
    evo_gate = PauliEvolutionGate(
        H_op, time=dt, synthesis=SuzukiTrotter(order=2))
    step_circuit = QuantumCircuit(L)
    step_circuit.append(evo_gate, range(L))
    # Decomposing once speeds up statevector evolution (native gates vs opaque gate)
    step_circuit = step_circuit.decompose()

    current_psi = psi

    for step in range(steps + 1):
        row = []
        for i in range(L):
            p1 = current_psi.probabilities([i])[1]
            row.append(p1)

        density_profile.append(row)

        # Evolve
        if step < steps:
            current_psi = current_psi.evolve(step_circuit)

        print(f"Step {step}/{steps} completed.", end="\r")

    print("\nSimulation complete.")

    # --- Plotting ---
    density_profile = np.array(density_profile)

    plt.figure(figsize=(8, 6))
    plt.imshow(density_profile, aspect='auto', origin='lower',
               extent=[0, L, 0, t_max], cmap='magma')
    plt.colorbar(label="Particle Density")
    plt.xlabel("Lattice Site n")
    plt.ylabel("Time t")
    plt.title(f"Scattering in Ising Field Theory (L={L})\n$g_x={
              gx}, g_z={gz}, k_0={k0/np.pi:.2f}\\pi$")
    plt.savefig("scattering_density.png")
    print("Plot saved to scattering_density.png")


if __name__ == "__main__":
    run_simulation()
