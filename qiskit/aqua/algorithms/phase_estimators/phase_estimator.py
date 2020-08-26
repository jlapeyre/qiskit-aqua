"""The Quantum Phase Estimation Algorithm."""

from typing import Dict, Optional, Union
from qiskit.circuit import QuantumCircuit
from qiskit.result import Counts
#from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit import execute
from qiskit.aqua.utils import get_subsystem_density_matrix
import numpy as np


class PhaseEstimator(QuantumAlgorithm):
    """The Quantum Phase Estimation algorithm.

    Blah
    """

    def __init__(self,
                 pe_circuit: QuantumCircuit,
                 num_evaluation_qubits,
                 num_unitary_qubits,
                 input_state_circuit = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None):


        self._pe_circuit = pe_circuit
        self._num_evaluation_qubits = num_evaluation_qubits
        self._num_unitary_qubits = num_unitary_qubits

        if not input_state_circuit is None:
            self._pe_circuit = pe_circuit.compose(input_state_circuit,
                                                  qubits=range(num_evaluation_qubits, num_evaluation_qubits + self._num_unitary_qubits),
                                                  front=True)
        else:
            self._pe_circuit = pe_circuit

        super().__init__(quantum_instance)
        self._add_classical_register()

    def _add_classical_register(self):
        """Explicitly add measurement instructions only if we are using a state vector simulator."""
        circ = self._pe_circuit
        if not self._quantum_instance.is_statevector:
            # Measure only the evaluation qubits.
            regname = 'meas'
            if not regname in [reg.name for reg in circ.cregs]:
                creg = ClassicalRegister(self._num_evaluation_qubits, regname)
                circ.add_register(creg)
                circ.barrier()
                circ.measure(range(self._num_evaluation_qubits), range(self._num_evaluation_qubits))


    def compute_phases(self):
        num_evaluation_qubits = self._num_evaluation_qubits
        if self._quantum_instance.is_statevector:
            state_vec = self._result.get_statevector()
            evaluation_density_matrix = get_subsystem_density_matrix(
                state_vec,
                range(num_evaluation_qubits, num_evaluation_qubits + self._num_unitary_qubits)
            )
            diag = evaluation_density_matrix.diagonal().real # The diagonal is real
            phases = diag
        else:
            # return counts with keys sorted numerically
            counts = self._result.get_counts()
            ck = list(counts.keys())
            ck.sort() # Sorts in order integer encoded by binary string
            phases = {k : counts[k] for k in ck}
            phases = Counts(phases, memory_slots=counts.memory_slots, creg_sizes=counts.creg_sizes)
#            phases = counts

        return phases

    # If we reversed the bit order of evaluation register circuit (including the iqft) then we would avoid
    # the binary calculations in the case of the statevector simulator.
    def _compute_phase(self):
        """Return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding to a
        phase of 2pi. It is assumed that the input vector is an eigenvecter of the unitary so that
        the probability density is concentrated on one string (in the noiseless case).
        """
        num_evaluation_qubits = self._num_evaluation_qubits
        if self._quantum_instance.is_statevector:
            state_vec = self._result.get_statevector()
            # FIXME: Following digs into the implementation of PhaseEstimation circuit.
            evaluation_density_matrix = get_subsystem_density_matrix(
                state_vec,
                range(num_evaluation_qubits, num_evaluation_qubits + self._num_unitary_qubits)
            )
            diag = evaluation_density_matrix.diagonal()
            idx = np.argmax(abs(diag)) # np.argmax ignores complex part of number. But, we take abs anyway
            binary_phase_string = np.binary_repr(idx, num_evaluation_qubits)

        else:
            counts = self._result.get_counts()
            binary_phase_string = max(counts, key=counts.get)

        phase = index_to_phase(binary_phase_string)

        return phase

    def _run(self) -> Dict:
        """Run the circuit and return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding
        to a phase of 2pi.
        """

        result = self._quantum_instance.execute(self._pe_circuit)
        self._result = result
        phase = self._compute_phase()

        return phase, result


def index_to_phase(binary_string):
    n_qubits = len(binary_string)
    return int(binary_string[::-1], 2) / (2 ** n_qubits)
