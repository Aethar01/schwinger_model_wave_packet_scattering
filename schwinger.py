import numpy as np
from qiskit.quantum_info import SparsePauliOp


def get_schwinger_hamiltonian(L, w, m, J, theta=0.0):
    """
    Constructs the Hamiltonian for the Schwinger Model (1+1D QED) on a lattice.

    Based on the Hamiltonian in Farrell et al. (arXiv:2505.03111v2).

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

    # 1. Hopping Term: -0.25 * w_eff * (X_n X_{n+1} + Y_n Y_{n+1})
    # Corresponds to -1/2 * sum (sigma+ sigma- + h.c.)
    for n in range(L - 1):
        # Effective hopping parameter including theta-dependent mass correction if applicable
        w_eff = (w - ((-1.0)**(n + 1) * m * np.sin(theta) * 0.5))
        term_coeff = -0.25 * w_eff

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

    # 2. Mass Term: - (m/2) * (-1)^n * Z_n
    for n in range(L):
        # mass_sign is (-1)^(n+1) = - (-1)^n
        mass_sign = (-1.0)**(n + 1)
        term_coeff = m * mass_sign * np.cos(theta) * 0.5

        label_z = ["I"] * L
        label_z[L - 1 - n] = "Z"
        op_list.append(("".join(label_z), term_coeff))

    # 3. Electric Field / Interaction (Coulomb Potential after integrating out gauge fields)
    # H_el = (g^2/2) * sum L_n^2
    # With Q_n = -0.5 * (Z_n + (-1)^n)

    # Term 1: Linear Z terms
    # + (J/2) * sum_{n even} sum_{l <= n} Z_l
    for n in range(L - 1):
        if (n + 1) % 2 == 1:  # n is even (since n starts at 0, n+1=odd => n=0, 2, ...)
            for l in range(n + 1):
                label = ["I"] * L
                label[L - 1 - l] = "Z"
                op_list.append(("".join(label), J / 2.0))

    # Term 2: Quadratic Z terms (ZZ)
    # + (J/2) * sum_{n} sum_{k < l <= n} Z_k Z_l
    for n in range(L - 1):
        for l in range(n + 1):
            for k in range(l):
                label = ["I"] * L
                label[L - 1 - k] = "Z"
                label[L - 1 - l] = "Z"
                op_list.append(("".join(label), J / 2.0))

    return SparsePauliOp.from_list(op_list).simplify()
