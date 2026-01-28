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
from os import makedirs


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


def get_schwinger_hamiltonian(L, w, m, J, theta=0.0):
    """
    Constructs the Hamiltonian for the Schwinger Model (1+1D QED) on a lattice.

    Args:
        L (int): Number of lattice sites (qubits).
        w (float): Hopping coefficient.
        m (float): Mass coefficient.
        J (float): Coupling coefficient (usually g^2/2).
        theta (float): Topological angle.

    Returns:
        SparsePauliOp: The Hamiltonian operator.
    """
    op_list = []

    # 1. Hopping Term: 0.5 * w * (X_n X_{n+1} + Y_n Y_{n+1}) + Mass corrections
    for n in range(L - 1):
        term_coeff = 0.5 * (w - ((-1.0)**(n + 1) * m * np.sin(theta) * 0.5))

        # XX
        label_x = ["I"] * L
        label_x[L - 1 - n] = "X"
        label_x[L - 1 - (n + 1)] = "X"
        op_list.append(("".join(label_x), term_coeff))

        # YY
        label_y = ["I"] * L
        label_y[L - 1 - n] = "Y"
        label_y[L - 1 - (n + 1)] = "Y"
        op_list.append(("".join(label_y), term_coeff))

    # 2. Mass Term: m * (-1)^n * Z_n
    for n in range(L):
        mass_sign = (-1.0)**(n + 1)
        term_coeff = m * mass_sign * np.cos(theta) * 0.5

        label_z = ["I"] * L
        label_z[L - 1 - n] = "Z"
        op_list.append(("".join(label_z), term_coeff))

    # 3. Electric Field / Interaction
    # Term 1: - (J/2) * sum_{n} (if n+1 odd) sum_{l <= n} Z_l
    for n in range(L - 1):
        if (n + 1) % 2 == 1:
            for l in range(n + 1):
                label = ["I"] * L
                label[L - 1 - l] = "Z"
                op_list.append(("".join(label), -J / 2.0))

    # Term 2: (J/2) * sum_{n} sum_{l <= n} sum_{k < l} Z_k Z_l
    for n in range(L - 1):
        for l in range(n + 1):
            for k in range(l):
                label = ["I"] * L
                label[L - 1 - k] = "Z"
                label[L - 1 - l] = "Z"
                op_list.append(("".join(label), J / 2.0))

    return SparsePauliOp.from_list(op_list).simplify()


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

    # Schwinger Parameters
    w = 0.5
    m = 1.0
    J = 0.5
    theta = 0

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
    Ising = False
    Schwinger = False
    # H_op, Ising = get_ising_hamiltonian_obc(L, gx, gz), True
    H_op, Schwinger = get_schwinger_hamiltonian(L, w, m, J, theta=theta), True

    # --- State Preparation ---
    qc = QuantumCircuit(L)

    qc_L = prepare_w_state_circuit(L, qubits_L, k0, x_L_local, sigma)
    qc.compose(qc_L, inplace=True)

    qc_R = prepare_w_state_circuit(L, qubits_R, -k0, x_R_local, sigma)
    qc.compose(qc_R, inplace=True)

    # --- Time Evolution ---
    psi = Statevector(qc)

    t_max = 25.0
    dt = 0.55
    steps = int(t_max / dt)

    density_profile = []

    print(f"Starting simulation on L={L} sites...")
    if Schwinger:
        print(
            f"Hamiltonian: Schwinger Model (1+1D QED), w={w}, m={m}, J={J}, theta={theta}")
    if Ising:
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
    if Schwinger:
        save_dir = "schwinger"
    if Ising:
        save_dir = "ising"
    makedirs(save_dir, exist_ok=True)

    density_profile = np.array(density_profile)

    plt.figure(figsize=(6, 10))
    plt.imshow(density_profile, aspect='auto', origin='lower',
               extent=[0, L, 0, t_max], cmap='magma')
    plt.colorbar(label="Particle Density")
    plt.xlabel("Lattice Site n")
    plt.ylabel("Time t")
    if Schwinger:
        plt.title(f"Scattering in Schwinger Model (L={L})\n$w={
                  w}, m={m}, J={J}, k_0={k0/np.pi:.2f}\\pi$")
    if Ising:
        plt.title(f"Scattering in Ising Field Theory (L={L})\n$k_0={
                  k0/np.pi:.2f}\\pi, \\sigma={sigma}$")
    plt.savefig(f"{save_dir}/scattering_density.png")
    print(f"Plot saved to {save_dir}/scattering_density.png")

    # plot density profiles at times 0, 5 and 10
    ts = [0, 5, 13.5]
    steps = [int(t / dt) for t in ts]

    for t, step in zip(ts, steps):
        plt.figure(figsize=(8, 6))
        plt.plot(density_profile[step])
        plt.xlabel("Lattice Site n")
        plt.ylabel("Particle Density")
        if Schwinger:
            plt.title(f"Scattering in Schwinger Model (L={L})\n$w={
                      w}, m={m}, J={J}, k_0={k0/np.pi:.2f}\\pi$")
        if Ising:
            plt.title(f"Scattering in Ising Field Theory (L={L})\n$k_0={
                      k0/np.pi:.2f}\\pi, \\sigma={sigma}$")
        plt.savefig(f"{save_dir}/scattering_density_{t}.png")
        print(f"Plot saved to {save_dir}/scattering_density_{t}.png")


if __name__ == "__main__":
    run_simulation()
