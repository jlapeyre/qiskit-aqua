import numpy as np
import qiskit as qk
import scipy
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from qiskit.aqua.operators import *
from qiskit.circuit.library import PhaseEstimation, RYGate

from qiskit.aqua.algorithms.phase_estimators import PhaseEstimator

def eigtester(op_circ, state_preparation=None, n_eval_qubits = 8):
    be = Aer.get_backend('qasm_simulator')
    # be = Aer.get_backend('statevector_simulator')
    qi = QuantumInstance(backend=be, shots=100000)
    pec = PhaseEstimation(n_eval_qubits, op_circ)
    p_est = PhaseEstimator(num_evaluation_qubits=n_eval_qubits,
                           pe_circuit = pec,
                           num_unitary_qubits=op_circ.num_qubits, quantum_instance=qi,
                           state_preparation=state_preparation)
    p_est.run()
    phase = p_est.single_phase()
    return phase

def phasestester(op_circ, state_preparation=None, n_eval_qubits = 8,
                 backend = Aer.get_backend('qasm_simulator')):
    qi = QuantumInstance(backend=backend, shots=100000)
    p_est = PhaseEstimator(num_evaluation_qubits=n_eval_qubits,
                           unitary = op_circ,
                           quantum_instance=qi,
                           state_preparation=state_preparation)
    # pec = PhaseEstimation(n_eval_qubits, op_circ)
    # p_est = PhaseEstimator(num_evaluation_qubits=n_eval_qubits,
    #                        pe_circuit = pec,
    #                        num_unitary_qubits=op_circ.num_qubits, quantum_instance=qi,
    #                    state_preparation=state_preparation)

    p_est.run()
    return p_est

def phasetest1():
    phi = 2 * np.pi / 3
    op_circ = (phi * Z).exp_i().to_circuit() # already a CircuitOp
    state_preparation = None
#    state_preparation = X.to_circuit()
    p_est = phasestester(op_circ, state_preparation, n_eval_qubits=8)
    return p_est

def phasetest2(backend = Aer.get_backend('qasm_simulator')):
    op_circ = Z.to_circuit()
    state_preparation = H.to_circuit()
    p_est = phasestester(op_circ, state_preparation, n_eval_qubits=8, backend=backend)
    return p_est

def eigtest1():
    op_circ = Z.to_circuit()
    state_preparation = None
    phase = eigtester(op_circ, state_preparation)
    assert phase == 0.0

def eigtest2():
    op_circ = Z.to_circuit()
    state_preparation = X.to_circuit()
    phase = eigtester(op_circ, state_preparation)
    assert phase == 0.5

def eigtest3():
    op_circ = X.to_circuit()
    state_preparation = H.to_circuit()
    phase = eigtester(op_circ, state_preparation)
    assert phase == 0.0
    return phase

def eigtest4():
    op_circ = X.to_circuit()
    state_preparation = QuantumCircuit(1)
    state_preparation.append(X.to_circuit(), [0])
    state_preparation.append(H.to_circuit(), [0])
    phase = eigtester(op_circ, state_preparation)
    assert phase == 0.5
    return phase

def eigtest5():
    op_circ = X.to_circuit()
    state_preparation = QuantumCircuit(1)
    state_preparation.append(H.to_circuit(), [0])
    phase = eigtester(op_circ, state_preparation)
    assert phase == 0.0
    return phase

def eigtest6():
    op_circ = X.to_matrix_op().to_instruction().definition
#    op_circ = X.to_matrix_op().to_circuit()  # fails due to bug in decompose
    state_preparation = QuantumCircuit(1)
    state_preparation.append(H.to_circuit(), [0])
    phase = eigtester(op_circ, state_preparation)
    assert phase == 0.0
    return phase

def eigtest7():
    op_circ = Z.exp_i().to_circuit() # already a CircuitOp
    state_preparation = None
    phase = eigtester(op_circ, state_preparation)
    assert np.isclose(phase, 1 - 1 / (2 * np.pi), rtol=1e-2)
    return phase

def eigtest8():
    phi = 0.42
    op_circ = (phi * Z).exp_i().to_circuit() # already a CircuitOp
    state_preparation = X.to_circuit()
    phase = eigtester(op_circ, state_preparation)
    assert np.isclose(phase, phi / (2 * np.pi), rtol=5e-2)
    return phase

def run_eigtests():
    eigtest1()
    eigtest2()
    eigtest3()
    eigtest4()
    eigtest5()
    eigtest6()
    eigtest7()
