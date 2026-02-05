import numpy as np
from qiskit.quantum_info import SparsePauliOp

def get_schwinger_hamiltonian(L_spatial, w, m, g):
    """
    Constructs the Hamiltonian for the Schwinger Model (1+1D QED) on a lattice
    with Open Boundary Conditions (OBC) and staggered fermions.
    
    L_spatial: Number of spatial sites. Total qubits = 2 * L_spatial.
    w: Hopping parameter (1/(2a)).
    m: Mass parameter.
    g: Coupling constant.
    
    H = H_kin + H_m + H_el
    
    H_kin = -w * sum_{j=0}^{2L-2} (sigma^+_j sigma^-_{j+1} + h.c.)
          = - (w/2) * sum_{j=0}^{2L-2} (X_j X_{j+1} + Y_j Y_{j+1})
          
    H_m = (m/2) * sum_{j=0}^{2L-1} (-1)^j Z_j
          (Constant term dropped)
          
    H_el = (g^2 / 2) * sum_{n=0}^{2L-2} (E_n)^2
    where E_n = sum_{k=0}^{n} Q_k
    and Q_k = (Z_k + (-1)^k) / 2
    """
    
    num_qubits = 2 * L_spatial
    op_list = []
    
    # 1. Kinetic Term (Hopping)
    # -w/2 * (XX + YY)
    for j in range(num_qubits - 1):
        # XX
        label_x = ["I"] * num_qubits
        label_x[num_qubits - 1 - j] = "X"
        label_x[num_qubits - 1 - (j + 1)] = "X"
        op_list.append(("".join(label_x), -0.5 * w))
        
        # YY
        label_y = ["I"] * num_qubits
        label_y[num_qubits - 1 - j] = "Y"
        label_y[num_qubits - 1 - (j + 1)] = "Y"
        op_list.append(("".join(label_y), -0.5 * w))
        
    # 2. Mass Term
    # (m/2) * (-1)^j * Z_j
    for j in range(num_qubits):
        sign = (-1)**j
        label_z = ["I"] * num_qubits
        label_z[num_qubits - 1 - j] = "Z"
        op_list.append(("".join(label_z), 0.5 * m * sign))
        
    # 3. Electric Term
    # H_el = (g^2 / 2) * sum_{n=0}^{2L-2} (sum_{k=0}^n Q_k)^2
    # Q_k = 0.5 * (Z_k + (-1)^k)
    
    # Let's expand the square: (sum Q_k)^2 = sum Q_k^2 + 2 sum_{k<l} Q_k Q_l
    
    # We will construct the operator symbolicly by adding terms.
    # It's easier to iterate and accumulate coefficients.
    
    # We use a dictionary to store coefficients for Pauli strings
    # Keys are tuples of (qubit_index, pauli_type), e.g. ((0, 'Z'), (1, 'Z'))
    # Since we only have Z and I terms in H_el, we can just use a dictionary mapping
    # indices tuple to coeff.
    
    # However, SparsePauliOp.from_list handles duplicates by summing them if we simplify.
    # So we can just append to op_list.
    
    J = (g**2) / 2.0
    
    for n in range(num_qubits - 1): # E_n is defined on links. Links 0 to 2L-2.
        # E_n = sum_{k=0}^{n} Q_k
        # We need E_n^2
        
        # Q_k = 0.5 * Z_k + 0.5 * (-1)^k
        
        # E_n = 0.5 * sum_{k=0}^n Z_k + C_n
        # where C_n = 0.5 * sum_{k=0}^n (-1)^k
        
        # C_n:
        # k=0: +0.5
        # k=1: -0.5. Sum=0
        # k=2: +0.5. Sum=0.5
        # So C_n = 0.5 if n is even, 0 if n is odd.
        
        C_n = 0.5 if (n % 2 == 0) else 0.0
        
        # E_n^2 = (0.5 * sum Z_k + C_n)^2
        #       = 0.25 * (sum Z_k)^2 + C_n^2 + C_n * sum Z_k
        #       = 0.25 * (sum Z_k^2 + 2 sum_{k<l} Z_k Z_l) + C_n^2 + C_n * sum Z_k
        #       = 0.25 * ( (n+1) * I + 2 sum_{k<l} Z_k Z_l ) + C_n^2 + C_n * sum Z_k
        
        # Note: Z_k^2 = I
        
        # We add these terms to op_list with factor J
        
        # Constant term (usually dropped, but we can keep it or not)
        # J * (0.25 * (n+1) + C_n^2)
        # We'll skip pure Identity terms for efficiency unless needed for exact energy values.
        # But for dynamics, global phase doesn't matter.
        
        # Linear Z terms: J * C_n * Z_k
        if abs(C_n) > 1e-9:
            for k in range(n + 1):
                label = ["I"] * num_qubits
                label[num_qubits - 1 - k] = "Z"
                op_list.append(("".join(label), J * C_n))
                
        # Quadratic ZZ terms: J * 0.5 * Z_k Z_l (factor 0.25 * 2 = 0.5)
        for l in range(n + 1):
            for k in range(l):
                label = ["I"] * num_qubits
                label[num_qubits - 1 - k] = "Z"
                label[num_qubits - 1 - l] = "Z"
                op_list.append(("".join(label), J * 0.5))
                
    return SparsePauliOp.from_list(op_list).simplify()