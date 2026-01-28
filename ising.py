from qiskit.quantum_info import SparsePauliOp


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
