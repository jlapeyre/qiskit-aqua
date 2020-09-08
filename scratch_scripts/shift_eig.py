from qiskit.aqua.algorithms.phase_estimators import PhaseEstimator, PhaseEstimationScale, PhaseEstimatorResult
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.quantum_info import Operator
from qiskit.aqua.operators import *
from qiskit.visualization import plot_histogram
import numpy as np
import scipy
import qiskit
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
_DECOMPOSER1Q = OneQubitEulerDecomposer('U3')

def matrix_to_circuit_1q(matrix):
    theta, phi, lam, global_phase = _DECOMPOSER1Q.angles_and_phase(matrix)
    q = QuantumRegister(1, "q")
    qc = qiskit.QuantumCircuit(1)
    qc._append(U3Gate(theta, phi, lam), [q[0]], [])
    qc.global_phase = global_phase
    return qc

def all_phases(unitary_circuit, state_preparation=None, num_evaluation_qubits = 8,
#             backend = Aer.get_backend('statevector_simulator'),
               backend = Aer.get_backend('qasm_simulator'),
             ):
    """ Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return the bit string with the largest amplitude.
    """
    qi = QuantumInstance(backend=backend, shots=100000)
    phase_est = PhaseEstimator(num_evaluation_qubits=num_evaluation_qubits,
                           unitary=unitary_circuit,
                           quantum_instance=None,
                           state_preparation=state_preparation)
    result = phase_est.run(qi)
    return result

def Zshift():
    t = PhaseEstimationScale(1.1)
#    unitary_circuit = (t.scale * Z).exp_i().to_matrix_op().to_circuit()
    unitary_circuit = (t.scale * Z).exp_i().to_circuit()
    state_preparation = H.to_circuit()
    result = all_phases(unitary_circuit, state_preparation)
    return result, t

def psum(bound_factor=1.2):
    a1 = 0.5
    a2 = 1.0
    op = (a1 * X) + (a2 * Z)
    bound = a1 + a2
    peig = PhaseEstimationScale(bound_factor *  bound)
    unitary_circuit = matrix_to_circuit_1q((peig.scale * op).exp_i().to_matrix())
    state_preparation = H.to_circuit()
    result = all_phases(unitary_circuit, state_preparation)
    return result, peig

def eigfilter(qpe_result, peig, cutoff=0.001):
    ph = qpe_result.filter_phases(cutoff, as_float=True)
    eigs = peig.eigenvalues(ph)
    return eigs

def byhand():
    a1 = 0.5
    a2 = 1.0
    op = (a1 * X) + (a2 * Z)
    m = op.to_matrix()
    em = scipy.linalg.expm(1j * m)
    return np.log(scipy.linalg.eigvals(em)).imag
