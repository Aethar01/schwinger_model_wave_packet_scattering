from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
import numpy as np

def get_op_pool_operator(L, op_name):
    """
    Generates the translationally invariant operator for a given name 
    under Open Boundary Conditions (OBC).
    
    Based on Eq. (17) in Farrell et al.
    """
    op_list = []
    
    if op_name == "Y":
        # Sum_n Y_n
        for n in range(L):
            label = ["I"] * L
            label[L - 1 - n] = "Y"
            op_list.append(("".join(label), 1.0))
            
    elif op_name == "Z":
        # Sum_n Z_n
        for n in range(L):
            label = ["I"] * L
            label[L - 1 - n] = "Z"
            op_list.append(("".join(label), 1.0))

    elif op_name == "YZ":
        # Sum_n (Y_n Z_{n+1} + Z_n Y_{n+1})
        for n in range(L - 1):
            # Term Y Z
            l1 = ["I"] * L
            l1[L - 1 - n] = "Y"
            l1[L - 1 - (n+1)] = "Z"
            op_list.append(("".join(l1), 1.0))
            
            # Term Z Y
            l2 = ["I"] * L
            l2[L - 1 - n] = "Z"
            l2[L - 1 - (n+1)] = "Y"
            op_list.append(("".join(l2), 1.0))

    elif op_name == "YX":
        # Sum_n (Y_n X_{n+1} + X_n Y_{n+1})
        for n in range(L - 1):
            # Term Y X
            l1 = ["I"] * L
            l1[L - 1 - n] = "Y"
            l1[L - 1 - (n+1)] = "X"
            op_list.append(("".join(l1), 1.0))
            
            # Term X Y
            l2 = ["I"] * L
            l2[L - 1 - n] = "X"
            l2[L - 1 - (n+1)] = "Y"
            op_list.append(("".join(l2), 1.0))

    elif op_name == "ZXY":
        # Sum_n (Z_n X_{n+1} Y_{n+2} + Y_n X_{n+1} Z_{n+2})
        for n in range(L - 2):
            # Term Z X Y
            l1 = ["I"] * L
            l1[L - 1 - n] = "Z"
            l1[L - 1 - (n+1)] = "X"
            l1[L - 1 - (n+2)] = "Y"
            op_list.append(("".join(l1), 1.0))
            
            # Term Y X Z
            l2 = ["I"] * L
            l2[L - 1 - n] = "Y"
            l2[L - 1 - (n+1)] = "X"
            l2[L - 1 - (n+2)] = "Z"
            op_list.append(("".join(l2), 1.0))
            
    elif op_name == "ZYZ":
        # Sum_n Z_n Y_{n+1} Z_{n+2}
        for n in range(L - 2):
            label = ["I"] * L
            label[L - 1 - n] = "Z"
            label[L - 1 - (n+1)] = "Y"
            label[L - 1 - (n+2)] = "Z"
            op_list.append(("".join(label), 1.0))

    else:
        raise ValueError(f"Operator {op_name} not implemented in pool.")

    return SparsePauliOp.from_list(op_list)

def apply_adapt_vqe_layer(qc, L, parameters):
    """
    Applies the ADAPT-VQE unitary layers to the quantum circuit.
    
    Args:
        qc (QuantumCircuit): The circuit to append to.
        L (int): System size.
        parameters (list of tuple): List of (op_name, theta) tuples.
    """
    # Use Trotter synthesis to decompose the exponential of Pauli sums into gates.
    # This avoids constructing the full dense matrix which causes memory errors for L=18.
    synth = SuzukiTrotter(order=2, reps=1)

    for op_name, theta in parameters:
        if abs(theta) < 1e-6:
            continue
            
        op = get_op_pool_operator(L, op_name)
        
        # PauliEvolutionGate(op, time=t) implements exp(-i * t * op)
        # To get exp(i * theta * O), we set t = -theta.
        evo = PauliEvolutionGate(op, time=-theta, synthesis=synth)
        
        # Decompose immediately to ensure we have standard gates in the circuit
        qc.compose(evo.definition, range(L), inplace=True)

