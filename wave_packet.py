import numpy as np
from qiskit import QuantumCircuit


def prepare_w_state_circuit(n_qubits, qubits, k0, x0, sigma):
    """
    Implements the unitary circuit to prepare a Wavepacket (W-state-like).

    Args:
        n_qubits (int): Total number of qubits in the circuit.
        qubits (list): List of qubit indices to apply the W-state preparation on.
        k0 (float): Momentum of the wavepacket.
        x0 (float): Center position of the wavepacket.
        sigma (float): Width of the wavepacket.

    Returns:
        QuantumCircuit: The circuit preparing the wavepacket.
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
