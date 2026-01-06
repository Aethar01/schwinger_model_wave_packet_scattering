#!/usr/bin/env python

from collections import defaultdict
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate, hamiltonian_variational_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


# --- 1. SETUP PARAMETERS AND HAMILTONIAN CONSTRUCTION ---

# Simulation Parameters
N_QUBITS = 8           # Number of qubits/lattice sites
MASS = 1.0             # Fermion mass (m)
INITIAL_MASS = 0.5
COUPLING = 1         # Interaction coupling (g)
TIME_MAX = 2.0        # Maximum evolution time
# TIME_STEPS = 200 // 10    # Number of time steps for data collection
TIME_STEPS = 10
# Number of Trotter steps per unit time (per unit time t=1)
TROTTER_STEPS = 5
THETA = 0

# Define the Parameter for time
time_param = Parameter('t')


def schwinger_hamiltonian(N, w, m, m0, J, theta) -> SparsePauliOp:
    """
    Generates the Pauli string terms for the U(1) Schwinger model Hamiltonian in 1+1D
    after Jordan-Wigner and gauge-field elimination, returning them as a Qiskit SparsePauliOp.

    The Hamiltonian terms are: H = H_hop + H_mass + H_elec
    - H_hop (Hopping/Kinetic): O(X_n X_{n+1} + Y_n Y_{n+1})
    - H_mass (Mass): O((-1)^n Z_n)
    - H_elec (Electric Field/Long-Range Interaction): O(L_n^2), where L_n is the electric field.

    Args:
        N (int): Number of qubits (sites).
        w (float): Hopping coefficient.
        m (float): Mass term coefficient.
        J (float): Electric field / Interaction coefficient.

    Returns:
        SparsePauliOp: A SparsePauliOp object representing the Hamiltonian.
    """
    if N < 2:
        print("Error: Schwinger model requires at least 2 sites (N >= 2).")
        # Return an empty SparsePauliOp
        return SparsePauliOp([], [])

    hamiltonian_terms = defaultdict(float)

    # A helper function to add terms and simplify the string representation
    def add_term(coeff, operators_and_indices):
        """Creates the string, aggregates the coefficient."""
        def create_pauli_string(N, operators_and_indices):
            """
            Creates a full Pauli string (e.g., 'IIXIIZ') for an N-qubit system.

            Args:
                N (int): Total number of qubits.
                operators_and_indices (list of tuples): List of (operator, index),
                                                        where operator is 'X', 'Y', or 'Z', and index is 0 to N-1.

            Returns:
                str: The full Pauli string, padded with 'I' (Identity).
            """
            # Start with all Identity operators
            string = ['I'] * N

            # Place the specified operators
            for op, idx in operators_and_indices:
                if 0 <= idx < N:
                    string[idx] = op
                else:
                    raise IndexError(
                        f"Qubit index {idx} out of range [0, {N-1}]")

            return "".join(string)
        try:
            pauli_string = create_pauli_string(N, operators_and_indices)
            hamiltonian_terms[pauli_string] += coeff
        except IndexError as e:
            print(f"Skipping term due to index error: {e}")

    # --- 1. Hopping Term (H_+-) ---
    coeff_w = w
    for n in range(N - 1):
        coeff = 0.5 * (coeff_w - ((-1.0)**(n+1) * m * np.sin(theta) * 0.5))
        # print(coeff)
        add_term(coeff, [('X', n), ('X', n + 1)])
        add_term(coeff, [('Y', n), ('Y', n + 1)])

    # --- 2. Mass Term and Field Term (H_Z) ---
    coeff_mass = m
    for n in range(N):
        mass_sign = (-1.0)**(n+1)
        add_term(coeff_mass * mass_sign * np.cos(theta) * 0.5, [('Z', n)])

    coeff_J = - J / 2
    for n in range(N-1):
        if (n+1) % 2 == 1:
            for l_0 in range(n+1):
                add_term(coeff_J, [('Z', l_0)])

    # --- 3. H_zz(Long-Range interaction) - --
    coeff_ZZ = J / 2
    for n in range(0, N - 1):
        for l in range(n+1):
            for k in range(l):
                add_term(coeff_ZZ, [('Z', k), ('Z', l)])

    labels = []
    coeffs = []

    # Filter out zero-coefficient terms and prepare the lists
    for pauli_str, coeff in hamiltonian_terms.items():
        if abs(coeff) > 1e-9:
            labels.append(pauli_str)
            coeffs.append(coeff)

    # Return the SparsePauliOp object
    # The .simplify() method aggregates identical Pauli strings and sorts them.
    return SparsePauliOp(labels, coeffs).simplify()


def build_thirring_hamiltonian(N, m, g):
    """
    Constructs the Massive Thirring Model Hamiltonian (H) in the staggered lattice
    formulation using the Jordan-Wigner transformation.
    """
    pauli_terms = []

    # 1. Kinetic Term (H_kin): Nearest-neighbor hopping
    for n in range(N - 1):
        # -1/4 * X_{n+1}Y_n
        term_xy = ['I'] * N
        term_xy[n] = 'Y'
        term_xy[n+1] = 'X'
        pauli_terms.append((f"{''.join(term_xy[::-1])}", -0.25))

        # +1/4 * Y_{n+1}X_n
        term_yx = ['I'] * N
        term_yx[n] = 'X'
        term_yx[n+1] = 'Y'
        pauli_terms.append((f"{''.join(term_yx[::-1])}", 0.25))

    # 2. Mass Term (H_mass): Staggered mass
    for n in range(N):
        coefficient = -m / 2 * (-1)**n
        term_z = ['I'] * N
        term_z[n] = 'Z'
        pauli_terms.append((f"{''.join(term_z[::-1])}", coefficient))

    # 3. Interaction Term (H_int): Nearest-neighbor interaction
    for n in range(N - 1):
        coeff = g/4
        pauli_terms.append(('I' * N, coeff))
        term_zn = ['I'] * N
        term_zn[n] = 'Z'
        pauli_terms.append((f"{''.join(term_zn[::-1])}", -coeff))
        term_znp1 = ['I'] * N
        term_znp1[n+1] = 'Z'
        pauli_terms.append((f"{''.join(term_znp1[::-1])}", -coeff))
        term_zz = ['I'] * N
        term_zz[n] = 'Z'
        term_zz[n+1] = 'Z'
        pauli_terms.append((f"{''.join(term_zz[::-1])}", coeff))

    H = SparsePauliOp.from_list(pauli_terms)
    H = H.chop(1e-12).simplify()
    return H


def vqe_ground_state(hamiltonian):
    """ Performs VQE to find the ground state of the Hamiltonian. """
    print("--- 2. VQE Ground State Preparation ---")

    initial_layer = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        if i % 2 == 0:
            initial_layer.x(i)

    # ansatz_su2 = efficient_su2(N_QUBITS, entanglement='reverse_linear', reps=2)
    ansatz_HVA = hamiltonian_variational_ansatz(hamiltonian, reps=2)
    ansatz = initial_layer.compose(ansatz_HVA)

    # optimizer = SPSA(maxiter=1000)
    optimizer = COBYLA(maxiter=3000)
    estimator = StatevectorEstimator()

    # Capture energy values for plotting
    energy_values = []

    def callback(eval_count, parameters, mean, std):
        energy_values.append(mean)
        print(f"VQE Iteration {eval_count}: Energy = {mean:.5f}", end="\r")

    vqe = VQE(estimator, ansatz, optimizer=optimizer, callback=callback)
    vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
    print(f"\nVQE Result: Ground State Energy = {
          vqe_result.eigenvalue.real:.4f}")
    H_matrix = hamiltonian.to_matrix(sparse=False)
    from scipy.linalg import eigh
    eigenvalues = eigh(H_matrix, eigvals_only=True)
    E_exact = np.min(eigenvalues).real
    print(f"Exact Ground State Energy from Diagonalization = {E_exact:.4f}")

    # Plotting VQE Convergence
    plt.figure(figsize=(6, 6))
    plt.plot(energy_values, label='VQE Energy')
    plt.axhline(y=E_exact, color='r', linestyle='--',
                label=f'Exact Energy = {E_exact:.4f}')
    plt.xlabel('Optimization Cycles')
    plt.ylabel('Energy')
    # plt.title('VQE Energy Convergence')
    plt.legend()
    # plt.grid(True)
    plt.show()
    # plt.savefig('vqe_convergence.png')
    # print("VQE convergence plot saved to 'vqe_convergence.png'")

    ground_state_circuit = ansatz.assign_parameters(
        vqe_result.optimal_parameters)
    return ground_state_circuit


def wave_packet_preparation(N, packet_width_sigma=1.0, momentum_k_magnitude=np.pi/4):
    """
    Creates a quantum circuit preparing two Gaussian wave packets with opposing momenta
    using the coefficients derived from the free theory analytical solution (Eq. 13 in paper).
    Args:
        N (int): Number of qubits/lattice sites.
        packet_width_sigma (float): Width parameter. 
                                    Note: Paper uses sigma_k. We assume input is spatial sigma, so sigma_k ~ 1/sigma.
                                    Or we follow the paper's example sigma_k = 2pi/N.
        momentum_k_magnitude (float): Momentum k.
    Returns:
        QuantumCircuit: A circuit preparing the initial wave packet state.
    """

    print(f"\n--- 3. Initial Wave Packet Preparation (Analytic Coeffs): sigma={
          packet_width_sigma}, k={momentum_k_magnitude:.2f} ---")
    wp_circuit = QuantumCircuit(N)
    # Packet Centers (Position space)
    mu_n_c = int(N * 0.25)
    if mu_n_c % 2 != 0:
        mu_n_c += 1
    mu_n_d = int(N * 0.75)
    if mu_n_d % 2 == 0:
        mu_n_d -= 1
    # Momentum space grid
    # k in 2pi/N * {-N/2, ..., N/2 - 1}
    # Range of integer indices for k
    k_indices = np.arange(-int(N/2), int(N/2))
    k_values = 2 * np.pi / N * k_indices
    # Momentum width
    # If packet_width_sigma is spatial width, sigma_k = 1/packet_width_sigma
    # Paper uses sigma_k = 2pi/N for their plots.
    sigma_k = 1.0 / packet_width_sigma
    # --- Calculate Momentum Space Coefficients (Eq 7) ---
    # Particle (c) centered at +k, position mu_n_c
    phi_c_k = np.exp(-1j * k_values * mu_n_c) * \
        np.exp(-(k_values - momentum_k_magnitude)**2 / (4 * sigma_k**2))
    # Normalize
    phi_c_k /= np.linalg.norm(phi_c_k)
    # Antiparticle (d) centered at -k, position mu_n_d
    phi_d_k = np.exp(-1j * k_values * mu_n_d) * np.exp(-(k_values -
                                                         (-momentum_k_magnitude))**2 / (4 * sigma_k**2))
    # Normalize
    phi_d_k /= np.linalg.norm(phi_d_k)
    # --- Calculate Position Space Coefficients (Eq 13) ---
    # Need w_k and v_k
    # m is global MASS
    m = MASS
    w_k = np.sqrt(m**2 + np.sin(k_values)**2)
    v_k = np.sin(k_values) / (m + w_k)
    phi_tilde_c = np.zeros(N, dtype=complex)
    phi_tilde_d = np.zeros(N, dtype=complex)
    for n in range(N):
        # Projectors
        Pi_n0 = 1 if n % 2 == 0 else 0
        Pi_n1 = 1 if n % 2 == 1 else 0
        # Terms in sum
        # term_c = phi_c_k * sqrt(...) * exp(ikn) * (Pi_n0 + v_k * Pi_n1)
        factor_c = np.sqrt((m + w_k) / w_k) * np.exp(1j *
                                                     k_values * n) * (Pi_n0 + v_k * Pi_n1)
        val_c = np.sum(phi_c_k * factor_c)
        phi_tilde_c[n] = val_c / np.sqrt(N)  # 1/sqrt(N) from Eq 13
        # term_d = phi_d_k * sqrt(...) * exp(ikn) * (Pi_n1 + v_k * Pi_n0)
        factor_d = np.sqrt((m + w_k) / w_k) * np.exp(1j *
                                                     k_values * n) * (Pi_n1 + v_k * Pi_n0)
        val_d = np.sum(phi_d_k * factor_d)
        phi_tilde_d[n] = val_d / np.sqrt(N)
    for n in range(N):
        if n % 2 == 0:
            # Even Site. Vacuum is |1>.
            # We want to create Particle (excitation) => Amplitude on |0>.
            # phi_tilde_c[n] is the amplitude of the particle.
            # State: sqrt(1-p)*|1> + sqrt(p)*e^{i phase}*|0>.
            amp = phi_tilde_c[n]
            mag = np.abs(amp)
            phase = np.angle(amp)
            if mag > 1.0:
                mag = 1.0

            # Ry(theta) |1> = cos(theta/2)|1> - sin(theta/2)|0>.
            # We want coefficient of |0> to be `mag`.
            # -sin(theta/2) = mag  => sin(theta/2) = -mag.
            # theta/2 = arcsin(-mag) = -arcsin(mag).
            # theta = -2 * arcsin(mag).
            # Then coeff of |0> is -(-mag) = mag. Positive.
            # We also need phase on |0>.
            # P(phi) on |0> does nothing relative to |1> usually?
            # Standard P gate: |0> -> |0>, |1> -> e^{i phi} |1>.
            # If we have `mag |0> + C |1>`, applying P(phi) gives `mag |0> + C e^{i phi} |1>`.
            # We want `mag e^{i phase} |0> + C |1>`.
            # Equivalent to `mag |0> + C e^{-i phase} |1>` (up to global phase).
            # So apply P(-phase).

            theta = -2 * np.arcsin(mag)
            wp_circuit.ry(theta, n)
            wp_circuit.p(-phase, n)

        else:
            # Odd Site. Vacuum is |0>.
            # We want to create Antiparticle (excitation) => Amplitude on |1>.
            # phi_tilde_d[n] is amplitude.
            # State: sqrt(1-p)|0> + sqrt(p) e^{i phase} |1>.
            amp = phi_tilde_d[n]
            mag = np.abs(amp)
            phase = np.angle(amp)
            if mag > 1.0:
                mag = 1.0

            # Ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>.
            # sin(theta/2) = mag.
            # theta = 2 * arcsin(mag).
            # Coeff of |1> is mag.
            # Apply P(phase) => mag e^{i phase} |1>.

            theta = 2 * np.arcsin(mag)
            wp_circuit.ry(theta, n)
            wp_circuit.p(phase, n)

    wp_circuit.barrier(label='Analytic Prep')
    return wp_circuit


def simulate_scattering(H, U0, Uwp):
    """
    Time-evolves the state using PauliEvolutionGate with manual Trotter steps and measures particle density.
    This version is optimized by batching all circuits into a single estimator job for performance.
    """
    print("\n--- 4. Adiabatic Time Evolution and Measurement ---")

    # Prepare initial state circuit, which is the same for all time steps
    initial_state_circuit = QuantumCircuit(N_QUBITS, name="Initial State")
    initial_state_circuit.compose(U0, inplace=True)
    initial_state_circuit.compose(Uwp, inplace=True)

    time_points = np.linspace(0, TIME_MAX, TIME_STEPS)

    # Define observables (Density: (I - Z_j)/2)
    observables = []
    for j in range(N_QUBITS):
        z_string = ('I' * (N_QUBITS - j - 1)) + 'Z' + ('I' * j)
        density_op = SparsePauliOp.from_list(
            [('I' * N_QUBITS, 0.5), (z_string, -0.5)]).simplify()
        observables.append((f'Density_Site_{j}', density_op))

    # --- 1. Build all circuits for all time steps ---
    circuits_to_run = []
    delta_t_fixed = 1.0 / TROTTER_STEPS
    for t in time_points:
        evolution_circuit = QuantumCircuit(N_QUBITS, name=f"Ev(t={t:.2f})")
        if t > 0:
            # Calculate the number of Trotter steps and the step size `dt` for this time point
            num_trotter_steps = max(1, int(round(t / delta_t_fixed)))
            dt = t / num_trotter_steps

            if num_trotter_steps > 0:
                # Create a single Trotter step as a circuit
                trotter_step_gate = PauliEvolutionGate(H, time=dt)
                single_trotter_circuit = QuantumCircuit(N_QUBITS)
                single_trotter_circuit.append(
                    trotter_step_gate, range(N_QUBITS))

                # Use the efficient .repeat() method instead of a Python loop for composition
                evolution_circuit = single_trotter_circuit.repeat(
                    num_trotter_steps)

        # Compose the fixed initial state with the time-dependent evolution
        full_circuit = initial_state_circuit.compose(evolution_circuit)
        circuits_to_run.append(full_circuit)

    # --- 2. Prepare all Program-Unit-of-Work (PUBs) for the estimator ---
    # Each PUB is a (circuit, observable) pair.
    # We create a flat list of all circuit-observable pairs to run in one job.
    pubs = [(circuit, obs)
            for circuit in circuits_to_run for _, obs in observables]

    # --- 3. Run the estimator once with all PUBs ---
    print(f"Submitting {
          len(pubs)} circuits to the estimator in a single batch job...")
    estimator = StatevectorEstimator()
    job = estimator.run(pubs)
    raw_results = job.result()
    print("Estimator job finished.")

    # --- 4. Process the results ---
    # The results will be a flat list of expectation values. Access via the .values attribute.
    results_flat = np.array([res.data.evs.real for res in raw_results])

    # Reshape the results: (num_time_steps, num_observables)
    results_reshaped = results_flat.reshape(TIME_STEPS, len(observables))

    density_dynamics = {key: [] for key, _ in observables}
    for j, (key, _) in enumerate(observables):
        density_dynamics[key] = results_reshaped[:, j].tolist()

    return time_points, density_dynamics


def measure_static_density(circuit, N_QUBITS):
    """Measures particle density for a static circuit."""
    observables = []
    for j in range(N_QUBITS):
        z_string = ('I' * (N_QUBITS - j - 1)) + 'Z' + ('I' * j)
        density_op = SparsePauliOp.from_list(
            [('I' * N_QUBITS, 0.5), (z_string, -0.5)]).simplify()
        observables.append((f'Density_Site_{j}', density_op))

    pubs = [(circuit, obs) for _, obs in observables]
    estimator = StatevectorEstimator()
    job = estimator.run(pubs)
    raw_results = job.result()

    results = {}
    for i, (name, _) in enumerate(observables):
        results[name] = raw_results[i].data.evs.real
    return results


def main():
    # --- 5. MAIN EXECUTION AND PLOTTING ---
    print("\n--- 5. MAIN EXECUTION AND PLOTTING ---")

    H = schwinger_hamiltonian(N_QUBITS, 1/2, MASS,
                              INITIAL_MASS, (COUPLING**2)*0.5, THETA)
    # H = build_thirring_hamiltonian(N_QUBITS, MASS, COUPLING)
    print(f"Hamiltonian (N={N_QUBITS}, m={MASS}, g={COUPLING}):\n{H}")

    if len(sys.argv) < 2:
        sys.argv.append("cache")
    if os.path.exists("cache/density_data.pkl") and sys.argv[1] != "new":
        T, density_data, vacuum_densities = pickle.load(
            open("cache/density_data.pkl", "rb"))
    else:
        U_ground = vqe_ground_state(H)
        vacuum_densities = measure_static_density(U_ground, N_QUBITS)
        # print("Vacuum Densities (Expectation of (I-Z)/2):")
        # for k, v in vacuum_densities.items():
        #     print(f"  {k}: {v:.4f}")

        U_wavepacket = wave_packet_preparation(
            N_QUBITS, packet_width_sigma=3, momentum_k_magnitude=np.pi/4)
        T, density_data = simulate_scattering(H, U_ground, U_wavepacket)
        os.makedirs("cache", exist_ok=True)
        pickle.dump((T, density_data, vacuum_densities),
                    open("cache/density_data.pkl", "wb"))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for key, values in density_data.items():
        site_index = int(key.split('_')[-1])
        # Even sites (0, 2, ...) correspond to particles; Odd sites (1, 3, ...) correspond to antiparticles.
        label_text = f'Site n={
            site_index} ({"Particle" if site_index % 2 == 0 else "Antiparticle"})'
        plt.plot(T, values, label=label_text, marker='o', markersize=3)

    plt.xlabel('Time (t)')
    plt.ylabel('Particle Density $\\langle n_j \\rangle$')
    # plt.title(f'Fermion Site Scattering (N={N_QUBITS}, g={
    #           COUPLING}, Trotter Steps={TROTTER_STEPS} per unit time)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    delta_density = np.zeros((N_QUBITS, T.shape[0]))
    new_density_data = np.zeros((N_QUBITS, T.shape[0]))
    for rawkey, value in density_data.items():
        density_data[rawkey] = np.array(value)
        key = int(rawkey.split('_')[-1])
        new_density_data[key] = density_data[rawkey]
        vac_dens = vacuum_densities[rawkey]
        delta_density[key] = density_data[rawkey] - vac_dens

    site_indices = np.arange(N_QUBITS)
    plt.figure(figsize=(10, 6))
    plt.bar(site_indices, delta_density.T[0], color='skyblue')
    plt.xlabel('Site Index')
    plt.ylabel('Initial Particle Density $\\langle n_j \\rangle_{t=0}$')
    # plt.title(f'Initial State Particle Density (t=0) (N={N_QUBITS})')
    # plt.xticks(site_indices)
    plt.xticks(np.arange(0, N_QUBITS, 1), np.arange(1, N_QUBITS+1, 1))
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 10))

    im = ax.imshow(
        delta_density.T,
        origin='lower',
        aspect='auto',
        cmap='seismic',
        interpolation='none',
        vmin=-1,
        vmax=1,
        extent=[-0.5, N_QUBITS - 0.5, T[0], T[-1]]
    )

    ax.set_xlabel('Site Index $n$')
    ax.set_ylabel('Time $t$')
    # ax.set_title(f'Fermion Site Density ($\Delta \\langle n_j \\rangle_t$) \n Schwinger Model (N={
                 # N_QUBITS}, m={MASS}, g={COUPLING}, $\\theta={THETA}$)')

    ax.set_xticks(np.arange(0, N_QUBITS, 1), np.arange(1, N_QUBITS+1, 1))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('$\\Delta \\langle n_j \\rangle_t$')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
