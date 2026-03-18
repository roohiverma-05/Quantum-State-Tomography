import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit.result import Result
from typing import Dict

# Data to be collected from Qubits 
def get_tomography_data(target_state_circuit: QuantumCircuit, shots = 1024):
# Simulates measurements in X, Y, and Z bases through POVM. 
    simulator = AerSimulator()
    data = {}

    bases = ['X', 'Y', 'Z']

    # Defining the Operators that will perform measurement on X,Y,Z. 
    for basis in bases:
        qc = target_state_circuit.copy()

        if basis == 'X':
            qc.h(0)
        elif basis == 'Y':
            qc.sdg(0)
            qc.h(0)
        # No gate needed for Z basis. 

        qc.measure_all()

        job = simulator.run(transpile(qc, simulator), shots = shots)
        result : Result = job.result()
        counts = dict(result.get_counts())
        
        n_0 = counts.get('0', 0)
        n_1 = counts.get('1', 0)
        data[basis] = {'0':n_0, '1':n_1}

    return data

# Creation of Density Matrix 
def param_to_rho(t_params):
    # Generation of the Cholesky Matrix for testing likelihood of the data gathered. 
    t1, t2, t3, t4 = t_params
    T = np.array([
        [t1, 0],
        [t3 + 1j*t4, t2]
        ], dtype=complex)
    # Creation of T dagger*T 
    T_dag_T = T.conj().T@T

    # Normalization of the Hermitian Matrix:
    norm_T_dag_T = np.trace(T_dag_T)

    # Density Matrix --> rho 
    rho = T_dag_T / norm_T_dag_T

    return rho 

#To calculate the Maximum Likelihood estimate
def neg_log_likelihood(t_params, data: Dict[str, Dict[str, int]]):
    
    rho = param_to_rho(t_params)
    loss = 0 

    # Z basis projectors for collapsing to 0 or 1 state:
    Pi_Z0 = np.array([[1,0], [0,0]])
    Pi_Z1 = np.array([[0,0], [0,1]])

    # X basis projectors for collapsing to 0 or 1 state:
    Pi_X0 = 0.5 * np.array([[1,1],[1,1]])
    Pi_X1 = 0.5 * np.array([[1,-1], [-1,1]])

    # Y basis projectors:
    Pi_Y0 = 0.5 * np.array([[1,-1j],[1j,1]] , dtype = complex)
    Pi_Y1 = 0.5 * np.array([[1,1j],[-1j,1]] , dtype = complex)

    projectors = {
        'X':{'0': Pi_X0, '1': Pi_X1},
        'Y':{'0': Pi_Y0, '1': Pi_Y1},
        'Z':{'0': Pi_Z0, '1': Pi_Z1}
    }

    
# Calculate L(t)= Sum [n_b,i * log(p_b(i))]
    for basis in ['X','Y','Z']:
        for outcome in ['0', '1']:
            n_count = data[basis][outcome]
        
            if n_count > 0:
                Pi = projectors[basis][outcome]
                prob = np.real(np.trace(rho @ Pi))
                if prob <= 1e-9: 
                    prob = 1e-9
                loss -= n_count * np.log(prob)
    return loss

qc = QuantumCircuit(1)
qc.h(0)
qc.t(0) # Rotation around Z axis, creates a complex state. 

print("Run simulation to gather tomography data:")
shots = 1000
data = get_tomography_data(qc, shots=shots)

print("Data gathered:" , data)

initial_guess = [1.0, 1.0, 0.0, 0.0]

print("Now running MLE")
opt_result = minimize(
    neg_log_likelihood, 
    initial_guess, 
    args=(data,), 
    method='BFGS'
)

optimized_params = opt_result.x
estimated_rho = param_to_rho(optimized_params)

#Calculation of Ideal state and Fidelity. 
ideal_rho = DensityMatrix.from_instruction(qc).data
fidelity = state_fidelity(ideal_rho, estimated_rho)

print("Tomography Results=")
print("Optimization success:",opt_result.success)

print('Ideal Density Matrix:')
print(np.round(ideal_rho, 4))

print("\nReconstructed Density Matrix (MLE):")
print(np.round(estimated_rho, 4))

print(f"\nState Fidelity: {fidelity:.6f}")