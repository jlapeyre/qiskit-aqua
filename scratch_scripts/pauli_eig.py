from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.phase_estimators import PhaseEstimator, PhaseEstimationScale, HamiltonianPE
from qiskit.aqua.algorithms.phase_estimators.hamiltonian_pe import TempPauliEvolve
from qiskit.aqua.operators import *
from qiskit.aqua.operators.evolutions import Trotter
from qiskit.visualization import plot_histogram

def all_phases(hamiltonian, state_preparation=None, num_evaluation_qubits = 10,
#             backend = Aer.get_backend('statevector_simulator'),
               backend = Aer.get_backend('qasm_simulator'),
             ):
    """ Run phase hamiltonian estimation
    """
    qi = QuantumInstance(backend=backend, shots=100000)
    phase_est = HamiltonianPE(num_evaluation_qubits=num_evaluation_qubits,
                              hamiltonian=hamiltonian,
                           quantum_instance=qi,
                              state_preparation=state_preparation,
                              evolution=TempPauliEvolve())
    result = phase_est.run()
    return result

def paulisum():
    a1 = 0.5
    a2 = 1.0
    a3 = 1.0
    hamiltonian = (a1 * X) + (a2 * Y) + (a3 * Z)
#    state_preparation = H.to_circuit()
    state_preparation = None
    result = all_phases(hamiltonian, state_preparation)
    return result

import numpy as np
import scipy
def byhand():
    a1 = 0.5
    a2 = 1.0
    a3 = 1.0
    op = (a1 * X) + (a2 * Y) + (a3 * Z)
    m = op.to_matrix()
    em = scipy.linalg.expm(1j * m)
    return np.log(scipy.linalg.eigvals(em)).imag
