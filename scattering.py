#!/usr/bin/env python3
"""
Main Simulation Script for Scattering in Ising Field Theory and Schwinger Model.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from os import makedirs
from sys import argv

# Import from split modules
from ising import get_ising_hamiltonian_obc
from schwinger import get_schwinger_hamiltonian
from wave_packet import prepare_w_state_circuit


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

    # --- Hamiltonian Selection ---
    # Uncomment the model you want to simulate
    model_name = "Ising"
    if argv[1].lower() == "ising":
        model_name = "Ising"  # Default
    if argv[1].lower() == "schwinger":
        model_name = "Schwinger"

    if model_name == "Ising":
        print("Initializing Ising Field Theory Hamiltonian...")
        H_op = get_ising_hamiltonian_obc(L, gx, gz)
        save_dir = "ising"
        title_str = f"Scattering in Ising Field Theory (L={L})\\n$k_0={
            k0/np.pi:.2f}\\pi, \\sigma={sigma}$"
    elif model_name == "Schwinger":
        print("Initializing Schwinger Model Hamiltonian...")
        H_op = get_schwinger_hamiltonian(L, w, m, J, theta=theta)
        save_dir = "schwinger"
        title_str = f"Scattering in Schwinger Model (L={L})\\n$w={w}, m={m}, J={
            J}, k_0={k0/np.pi:.2f}\\pi, \\sigma={sigma}$"
    else:
        raise ValueError("Unknown model name")

    # --- State Preparation ---
    print("Preparing Wavepackets...")
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
    print(f"Time steps: {steps}, dt: {dt}, t_max: {t_max}")

    # Evolution Circuit
    evo_gate = PauliEvolutionGate(
        H_op, time=dt, synthesis=SuzukiTrotter(order=2))
    step_circuit = QuantumCircuit(L)
    step_circuit.append(evo_gate, range(L))

    # Decomposing once speeds up statevector evolution (native gates vs opaque gate)
    # Note: For very large L, decomposition might be slow, but for 18 it's fine.
    step_circuit = step_circuit.decompose()

    current_psi = psi

    for step in range(steps + 1):
        row = []
        for i in range(L):
            # Probability of |1> at site i
            p1 = current_psi.probabilities([i])[1]
            row.append(p1)

        density_profile.append(row)

        # Evolve
        if step < steps:
            current_psi = current_psi.evolve(step_circuit)

        print(f"Step {step}/{steps} completed.", end="\r")

    print("\nSimulation complete.")

    # --- Plotting ---
    makedirs(save_dir, exist_ok=True)

    density_profile = np.array(density_profile)

    plt.figure(figsize=(6, 10))
    plt.imshow(density_profile, aspect='auto', origin='lower',
               extent=[0, L, 0, t_max], cmap='magma')
    plt.colorbar(label="Particle Density")
    plt.xlabel("Lattice Site n")
    plt.ylabel("Time t")
    plt.title(title_str)
    plt.savefig(f"{save_dir}/scattering_density.png")
    print(f"Plot saved to {save_dir}/scattering_density.png")

    # plot density profiles at specific times
    ts = [0, 5, 13.5]

    # Find closest steps
    target_steps = [int(t / dt) for t in ts]

    for t, step_idx in zip(ts, target_steps):
        if step_idx < len(density_profile):
            plt.figure(figsize=(8, 6))
            plt.plot(density_profile[step_idx])
            plt.xlabel("Lattice Site n")
            plt.ylabel("Particle Density")
            plt.text(0.05, 0.95, f"t={t}", ha='left',
                     va='top', transform=plt.gca().transAxes)
            plt.title(title_str)
            plt.savefig(f"{save_dir}/scattering_density_{t}.png")
            print(f"Plot saved to {save_dir}/scattering_density_{t}.png")


if __name__ == "__main__":
    run_simulation()
