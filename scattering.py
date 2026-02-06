#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from os import makedirs

# Reuse the Ising W-state preparation as a base
from wave_packet import prepare_w_state_circuit_ising
from adapt_vqe import apply_adapt_vqe_layer, run_adapt_vqe


def get_schwinger_hamiltonian_obc(L, m0, g, w=1.0):
    """
    Constructs the Schwinger Model Hamiltonian under Open Boundary Conditions (OBC)
    using the Jordan-Wigner transformation.

    H = H_mass + H_hop + H_gauge

    H_mass = (m0/2) * sum_{n=0}^{L-1} (-1)^n Z_n
    H_hop  = (w/2)  * sum_{n=0}^{L-2} (X_n X_{n+1} + Y_n Y_{n+1})
    H_gauge = (g^2/2) * sum_{n=0}^{L-2} (E_n)^2

    where E_n = sum_{k=0}^n Q_k
          Q_k = (Z_k + (-1)^k)/2  (Charge operator)
    """
    mass_ops = []
    for n in range(L):
        label = ["I"] * L
        label[L - 1 - n] = "Z"
        coeff = (m0 / 2) * ((-1)**n)
        mass_ops.append(("".join(label), coeff))

    hop_ops = []
    for n in range(L - 1):
        lbl_xx = ["I"] * L
        lbl_xx[L - 1 - n] = "X"
        lbl_xx[L - 1 - (n+1)] = "X"
        hop_ops.append(("".join(lbl_xx), w/2))

        lbl_yy = ["I"] * L
        lbl_yy[L - 1 - n] = "Y"
        lbl_yy[L - 1 - (n+1)] = "Y"
        hop_ops.append(("".join(lbl_yy), w/2))

    gauge_ops = {}  # Map Pauli string to coefficient

    def add_term(pauli_str, val):
        if pauli_str in gauge_ops:
            gauge_ops[pauli_str] += val
        else:
            gauge_ops[pauli_str] = val

    for n in range(L - 1):  # Sum over links

        indices = range(n + 1)
        prefactor = (g**2 / 2) * 0.25

        for k in indices:
            for p in indices:

                if k == p:
                    add_term("I" * L, prefactor)
                else:
                    lbl = ["I"] * L
                    lbl[L - 1 - k] = "Z"
                    lbl[L - 1 - p] = "Z"
                    add_term("".join(lbl), prefactor)

                lbl_k = ["I"] * L
                lbl_k[L - 1 - k] = "Z"
                add_term("".join(lbl_k), prefactor * ((-1)**p))

                lbl_p = ["I"] * L
                lbl_p[L - 1 - p] = "Z"
                add_term("".join(lbl_p), prefactor * ((-1)**k))

                add_term("I" * L, prefactor * ((-1)**(k+p)))

    gauge_op_list = [(k, v) for k, v in gauge_ops.items() if abs(v) > 1e-9]

    all_ops = mass_ops + hop_ops + gauge_op_list
    return SparsePauliOp.from_list(all_ops).simplify()


def prepare_schwinger_wavepacket(L, qubits_indices, k0, x0, sigma):
    """
    Prepares a Schwinger model wavepacket using the W-state strategy.

    Strategy:
    1. The system has L staggered sites (L qubits).
       Actually, standard notation: L spatial sites -> 2L staggered sites (qubits).
       Let's assume the input L is the number of QUBITS (2 * Spatial Sites).

    2. We treat pairs (2i, 2i+1) as sites for the W-state.
       We select the even qubits [q0, q2, q4, ...] to prepare the W-state.

    3. Expand 1 -> 11: CNOT(2i, 2i+1).

    4. Apply X to all even sites to set the correct vacuum background.
       (Assuming vacuum is |0101...> and 11 excitation becomes 01, 00 becomes 10).
    """
    qc = QuantumCircuit(L)

    even_qubits = [q for q in qubits_indices if q % 2 == 0]

    w_qc = prepare_w_state_circuit_ising(L, even_qubits, k0, x0, sigma)
    qc.compose(w_qc, inplace=True)

    for q_even in even_qubits:
        if q_even + 1 < L:
            qc.cx(q_even, q_even + 1)

    for i in range(0, L, 2):
        qc.x(i)

    return qc


def run_simulation():
    L_spatial = 8
    L = 2 * L_spatial
    m0 = 0.5
    g = 0.5
    w = 1.0
    k0 = 0.1 * np.pi
    sigma = 1.25

    mid_spatial = L_spatial // 2

    x_L_spatial = 2

    x_R_spatial = L_spatial - 3

    qubits_L = list(range(0, L // 2))
    qubits_R = list(range(L // 2, L))

    print("Initializing Schwinger Model Hamiltonian...")
    H_op = get_schwinger_hamiltonian_obc(L, m0, g, w)
    save_dir = "schwinger"
    title_str = f"Scattering in Schwinger Model (L={L})\\n$k_0={
        k0/np.pi:.2f}\\pi, \\sigma={sigma}$"

    print("Preparing Wavepackets...")

    # Prepare Single Wavepacket with positive momentum
    # We use the full lattice for the single wavepacket
    q_idxs = list(range(0, L))

    # Place it on the left side to observe propagation to the right
    x_start_spatial = 2

    qc_combined = prepare_schwinger_wavepacket(
        L, q_idxs, k0, x_start_spatial, sigma)

    # --- ADAPT-VQE Energy Minimization ---
    print("Running ADAPT-VQE to minimize energy...")
    # The pool "Schwinger_1" corresponds to kinetic terms which are good for this.
    # We might want to expand the pool for better results, but this demonstrates the method.
    adapt_ansatz = run_adapt_vqe(
        H_op, qc_combined, L, max_steps=2, pool=["Schwinger_1"])

    print("Applying optimized ADAPT-VQE layers...")
    apply_adapt_vqe_layer(qc_combined, L, adapt_ansatz)

    # --- Vacuum Preparation ---
    print("Preparing Vacuum State...")
    vac_qc = QuantumCircuit(L)
    # Initialize staggered vacuum |1010...>
    for i in range(0, L, 2):
        vac_qc.x(i)

    print("Running ADAPT-VQE for Vacuum...")
    vac_ansatz = run_adapt_vqe(
        H_op, vac_qc, L, max_steps=2, pool=["Schwinger_1"])
    apply_adapt_vqe_layer(vac_qc, L, vac_ansatz)

    # --- Time Evolution ---
    psi = Statevector(qc_combined)
    vac_psi = Statevector(vac_qc)

    t_max = 15.0
    dt = 0.2
    steps = int(t_max / dt)

    density_profile = []

    print(f"Starting simulation on L={L} qubits...")

    title_str = f"Vacuum-Subtracted Charge Density (L={L})\\n$k_0={
        k0/np.pi:.2f}\\pi, \\sigma={sigma}$"
    evo_gate = PauliEvolutionGate(
        H_op, time=dt, synthesis=SuzukiTrotter(order=2))
    step_circuit = QuantumCircuit(L)
    step_circuit.append(evo_gate, range(L))
    step_circuit = step_circuit.decompose()

    current_psi = psi
    current_vac = vac_psi

    for step in range(steps + 1):
        row = []
        for n in range(L):
            # WP Density
            probs_wp = current_psi.probabilities([n])
            z_exp_wp = probs_wp[0] - probs_wp[1]
            rho_wp = (z_exp_wp + (-1)**n) / 2.0

            # Vacuum Density
            probs_vac = current_vac.probabilities([n])
            z_exp_vac = probs_vac[0] - probs_vac[1]
            rho_vac = (z_exp_vac + (-1)**n) / 2.0

            row.append(rho_wp - rho_vac)

        density_profile.append(row)

        if step < steps:
            current_psi = current_psi.evolve(step_circuit)
            current_vac = current_vac.evolve(step_circuit)

        print(f"Step {step}/{steps} completed.", end="\r")

    print("\nSimulation complete.")

    makedirs(save_dir, exist_ok=True)
    density_profile = np.array(density_profile)

    plt.figure(figsize=(6, 10))
    plt.imshow(density_profile, aspect='auto', origin='lower',
               extent=[0, L, 0, t_max], cmap='magma')
    plt.colorbar(label="Charge Density")
    plt.xlabel("Staggered Site n")
    plt.ylabel("Time t")
    plt.title(title_str)
    plt.savefig(f"{save_dir}/scattering_density.png")
    print(f"Plot saved to {save_dir}/scattering_density.png")

    ts = [0, 5, 10, 15]
    for t in ts:
        step_idx = int(t / dt)
        if step_idx < len(density_profile):
            plt.figure(figsize=(8, 6))
            plt.plot(density_profile[step_idx], label="Charge Density")
            plt.xlabel("Staggered Site n")
            plt.ylabel("Density")
            plt.title(f"{title_str} (t={t})")
            plt.legend()
            plt.savefig(f"{save_dir}/density_t{t}.png")


if __name__ == "__main__":
    run_simulation()
