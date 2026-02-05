from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from scipy.optimize import minimize
import numpy as np


def get_op_pool_operator(L, op_name):
    """
    Generates the translationally invariant operator for a given name 
    under Open Boundary Conditions (OBC).
    """
    op_list = []

    if op_name == "Schwinger_1":
        # Sum_n (-1)^n (X_n Y_{n+1} - Y_n X_{n+1})
        # This is derived from [H_mass, H_kin] ~ [Sum (-1)^n Z_n, Sum (XX+YY)]
        for n in range(L - 1):
            sign = (-1)**n
            
            # X_n Y_{n+1}
            l1 = ["I"] * L
            l1[L - 1 - n] = "X"
            l1[L - 1 - (n+1)] = "Y"
            op_list.append(("".join(l1), sign * 1.0))

            # - Y_n X_{n+1}
            l2 = ["I"] * L
            l2[L - 1 - n] = "Y"
            l2[L - 1 - (n+1)] = "X"
            op_list.append(("".join(l2), -1.0 * sign))

    elif op_name == "Y":
        for n in range(L):
            label = ["I"] * L
            label[L - 1 - n] = "Y"
            op_list.append(("".join(label), 1.0))

    elif op_name == "Z":
        for n in range(L):
            label = ["I"] * L
            label[L - 1 - n] = "Z"
            op_list.append(("".join(label), 1.0))

    elif op_name == "YZ":
        for n in range(L - 1):
            l1 = ["I"] * L
            l1[L - 1 - n] = "Y"
            l1[L - 1 - (n+1)] = "Z"
            op_list.append(("".join(l1), 1.0))

            l2 = ["I"] * L
            l2[L - 1 - n] = "Z"
            l2[L - 1 - (n+1)] = "Y"
            op_list.append(("".join(l2), 1.0))

    elif op_name == "YX":
        for n in range(L - 1):
            l1 = ["I"] * L
            l1[L - 1 - n] = "Y"
            l1[L - 1 - (n+1)] = "X"
            op_list.append(("".join(l1), 1.0))

            l2 = ["I"] * L
            l2[L - 1 - n] = "X"
            l2[L - 1 - (n+1)] = "Y"
            op_list.append(("".join(l2), 1.0))

    elif op_name == "ZXY":
        for n in range(L - 2):
            l1 = ["I"] * L
            l1[L - 1 - n] = "Z"
            l1[L - 1 - (n+1)] = "X"
            l1[L - 1 - (n+2)] = "Y"
            op_list.append(("".join(l1), 1.0))

            l2 = ["I"] * L
            l2[L - 1 - n] = "Y"
            l2[L - 1 - (n+1)] = "X"
            l2[L - 1 - (n+2)] = "Z"
            op_list.append(("".join(l2), 1.0))

    else:
        raise ValueError(f"Operator {op_name} not implemented in pool.")

    return SparsePauliOp.from_list(op_list)


def apply_adapt_vqe_layer(qc: QuantumCircuit, L, parameters):
    """
    Applies the ADAPT-VQE unitary layers to the quantum circuit.
    """
    synth = SuzukiTrotter(order=2, reps=1)

    for op_name, theta in parameters:
        if abs(theta) < 1e-6:
            continue

        op = get_op_pool_operator(L, op_name)
        # exp(i * theta * O)
        evo = PauliEvolutionGate(op, time=-theta, synthesis=synth)
        qc.compose(evo.definition, range(L), inplace=True)


def run_adapt_vqe(H, initial_state_qc, L, max_steps=1, pool=["Schwinger_1"]):
    """
    Runs a simplified ADAPT-VQE loop.
    
    Args:
        H (SparsePauliOp): Hamiltonian.
        initial_state_qc (QuantumCircuit): Circuit preparing the initial state.
        L (int): System size.
        max_steps (int): Number of ADAPT layers to add.
        pool (list): List of operator names to consider.
        
    Returns:
        list: Optimized ansatz [(op_name, theta), ...]
    """
    ansatz = []
    
    # Precompute pool operators
    pool_ops = {name: get_op_pool_operator(L, name) for name in pool}
    
    current_params = []
    
    for step in range(max_steps):
        print(f"  ADAPT-VQE Step {step+1}/{max_steps}...")
        
        # 1. Prepare current state
        qc = initial_state_qc.copy()
        apply_adapt_vqe_layer(qc, L, ansatz)
        psi = Statevector(qc)
        
        # 2. Measure gradients: |<psi| [H, A] |psi>|
        # Since A is anti-Hermitian (i*P), [H, A] is Hermitian?
        # A_k = i * P_k. 
        # Gradient is dE/dtheta = <psi| [H, A_k] |psi>
        # [H, i P_k] = i (H P_k - P_k H).
        # This is an observable.
        
        best_op = None
        max_grad = -1.0
        
        for name, op in pool_ops.items():
            # Commutator C = [H, op]
            # Since 'op' in pool is usually defined as the Hermitian part P, and U = exp(i theta P).
            # The generator is i P. 
            # Grad = <psi | [H, i P] | psi > = i <psi| [H, P] |psi>
            #      = i <psi| (HP - PH) |psi> = i ( <psi|HP|psi> - <psi|PH|psi> )
            #      = i ( <psi|HP|psi> - <psi|HP|psi>* )  (since H, P Hermitian)
            #      = i ( 2i Im( <psi|HP|psi> ) ) = -2 Im( <psi|HP|psi> )
            
            # We can compute <psi| H @ op |psi>
            # Note: SparsePauliOp matmul is '@'.
            
            val = psi.expectation_value(H @ op)
            grad = abs(2 * val.imag)
            
            if grad > max_grad:
                max_grad = grad
                best_op = name
        
        print(f"    Best operator: {best_op} (Grad: {max_grad:.6f})")
        
        if max_grad < 1e-3:
            print("    Gradient too small, stopping.")
            break
            
        ansatz.append((best_op, 0.0))
        current_params.append(0.0)
        
        # 3. Optimize parameters
        def cost_fn(params):
            qc_opt = initial_state_qc.copy()
            # Reconstruct ansatz with new params
            current_ansatz = []
            for i, (name, _) in enumerate(ansatz):
                current_ansatz.append((name, params[i]))
            
            apply_adapt_vqe_layer(qc_opt, L, current_ansatz)
            psi_opt = Statevector(qc_opt)
            return psi_opt.expectation_value(H).real
        
        res = minimize(cost_fn, current_params, method='COBYLA', tol=1e-3)
        current_params = list(res.x)
        
        # Update ansatz with optimized params
        for i in range(len(ansatz)):
            ansatz[i] = (ansatz[i][0], current_params[i])
            
        print(f"    New Energy: {res.fun:.6f}")
        
    return ansatz