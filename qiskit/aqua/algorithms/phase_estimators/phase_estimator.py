"""The Quantum Phase Estimation Algorithm."""

from typing import Dict, Optional, Union
from qiskit.circuit import QuantumCircuit
import qiskit
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit import execute
from qiskit.aqua.utils import get_subsystem_density_matrix
from .phase_estimator_result import PhaseEstimatorResult, _sort_phases
import numpy

class PhaseEstimator(QuantumAlgorithm):
    """The Quantum Phase Estimation algorithm.
    """

    def __init__(self,
                 num_evaluation_qubits,
                 unitary = None,
                 pe_circuit = None,
                 num_unitary_qubits = None,
                 state_preparation = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None):

        """

        Args:
            num_evaluation_qubits: The number of qubits used in estimating the phase. The
                                   phase will be estimated as a binary string with this many
                                   bits.
            pe_circuit: The phase estimation circuit.
            num_unitary_qubits: Must agree with the number of qubits in the unitary in `pe_circuit` if
                                `pe_circuit` is passed. This parameter will be set from
                                `unitary` if `unitary` is passed.
            unitary: The circuit representing the unitary operator whose eigenvalues (via phase) will
                     be measured. Exactly one of `pe_circuit` and `unitary` must be passed.
            state_preparation: The circuit that prepares the state whose eigenphase will be measured.
                                 If this parameter is ommited, no preparation circuit will be run and
                                 input state will be the all-zero state in the computational basis.
            quantum_instance: The quantum instance on which the circuit will be run.
        """

        # Determine if user passed a unitary, or the entire QPE circuit with the unitary already embedded,
        # and set properties.
        if unitary is None:
            if pe_circuit is None:
                raise ValueError('Only one of `unitary` and `pe_circuit` may be `None`.')
            else:
                if num_unitary_qubits is None:
                    raise ValueError('`num_unitary_qubits` is required when passing `pe_circuit`.')
                self._pe_circuit = pe_circuit
                self._num_unitary_qubits = num_unitary_qubits
        else:
            if not (num_unitary_qubits is None or num_unitary_qubits == unitary.num_qubits):
                raise ValueError('`num_unitary_qubits` disagrees with size of `unitary`.')
            pe_circuit = PhaseEstimation(num_evaluation_qubits, unitary)
            self._num_unitary_qubits = unitary.num_qubits
            self._pe_circuit = pe_circuit

        self._num_evaluation_qubits = num_evaluation_qubits

        if not state_preparation is None:
            self._pe_circuit = pe_circuit.compose(
                state_preparation,
                qubits=range(num_evaluation_qubits, num_evaluation_qubits + self._num_unitary_qubits),
                front=True)
        else:
            self._pe_circuit = pe_circuit

        super().__init__(quantum_instance)
        self._add_classical_register()

    def _add_classical_register(self):
        """Explicitly add measurement instructions only if we are using a state vector simulator."""
        if not self._quantum_instance.is_statevector:
            # Measure only the evaluation qubits.
            regname = 'meas'
            circ = self._pe_circuit
            if not regname in [reg.name for reg in circ.cregs]:
                creg = ClassicalRegister(self._num_evaluation_qubits, regname)
                circ.add_register(creg)
                circ.barrier()
                circ.measure(range(self._num_evaluation_qubits), range(self._num_evaluation_qubits))


    @property
    def pe_circuit(self):
        """ Return the phase estimation circuit. """
        return self._pe_circuit

    def _compute_phases(self, circuit_result):
        if self._quantum_instance.is_statevector:
            state_vec = circuit_result.get_statevector()
            evaluation_density_matrix = get_subsystem_density_matrix(
                state_vec,
                range(self._num_evaluation_qubits, self._num_evaluation_qubits + self._num_unitary_qubits)
            )
            phases = evaluation_density_matrix.diagonal().real # The diagonal is real
        else:
            # return counts with keys sorted numerically
            num_shots = circuit_result.results[0].shots
            counts = circuit_result.get_counts()
            phases = {k[::-1] : counts[k] / num_shots for k in counts.keys()}
            phases = _sort_phases(phases)
            phases = qiskit.result.Counts(phases, memory_slots=counts.memory_slots, creg_sizes=counts.creg_sizes)

        return phases


    def _run(self):
        """
        Run the circuit and return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding
        to a phase of 2pi.
        """

        circuit_result = self._quantum_instance.execute(self._pe_circuit)
        phases = self._compute_phases(circuit_result)
        if isinstance(phases, numpy.ndarray):
            return PhaseEstimatorResult(self._num_evaluation_qubits, phase_array = phases, circuit_result=circuit_result)
        else:
            return PhaseEstimatorResult(self._num_evaluation_qubits, phase_dict = phases, circuit_result=circuit_result)
