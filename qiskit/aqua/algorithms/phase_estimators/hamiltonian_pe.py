import numpy
from .phase_estimator import PhaseEstimator
from typing import Optional, Union
from qiskit.aqua import QuantumInstance
from qiskit.providers import BaseBackend
from . import phase_estimation_scale
from .hamiltonian_pe_result import HamiltonianPEResult


import qiskit
from qiskit.circuit.library.standard_gates import U3Gate
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
_DECOMPOSER1Q = OneQubitEulerDecomposer('U3')

class TempPauliEvolve():
    """Class for working around a bug in terra.

    Hopefully, this can be removed soon.
    """

    def __init__(self):
        pass

    def convert(self, pauli_operator):
        matrix = pauli_operator.exp_i().to_matrix()
        return self._matrix_to_circuit_1q(matrix)

    def _matrix_to_circuit_1q(self, matrix):
        theta, phi, lam, global_phase = _DECOMPOSER1Q.angles_and_phase(matrix)
        q = QuantumRegister(1, "q")
        qc = qiskit.QuantumCircuit(1)
        qc._append(U3Gate(theta, phi, lam), [q[0]], [])
        qc.global_phase = global_phase
        return qc


class HamiltonianPE(PhaseEstimator):

    def __init__(self,
                 num_evaluation_qubits,
                 hamiltonian,
                 evolution = None,
                 state_preparation = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None):


        self._hamiltonian = hamiltonian
        self._evolution = evolution

        unitary = self._get_unitary()

        super().__init__(num_evaluation_qubits,
                         unitary=unitary,
                         pe_circuit = None,
                         num_unitary_qubits = None,
                         state_preparation = state_preparation,
                         quantum_instance = quantum_instance)


    @property
    def phase_estimation_scale(self):
        return self._pe_scale

    def _get_unitary(self):
        pe_scale = phase_estimation_scale.from_pauli_sum(self._hamiltonian)
        self._pe_scale = pe_scale
        unitary = self._evolution.convert(pe_scale.scale * self._hamiltonian)
        if not isinstance(unitary, qiskit.QuantumCircuit):
            return unitary.to_circuit()
        else:
            return unitary


    def _run(self):
        """Run the circuit and return and return `PhaseEstimatorResult`.
        """

        self._add_classical_register()
        circuit_result = self._quantum_instance.execute(self._pe_circuit)
        phases = self._compute_phases(circuit_result)
        if isinstance(phases, numpy.ndarray):
            return HamiltonianPEResult(self._num_evaluation_qubits, phase_array = phases, circuit_result=circuit_result,
                                       phase_estimation_scale=self._pe_scale)
        else:
            return HamiltonianPEResult(self._num_evaluation_qubits, phase_dict = phases, circuit_result=circuit_result,
                                       phase_estimation_scale=self._pe_scale)
