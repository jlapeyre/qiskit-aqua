"""The Quantum Phase Estimation Algorithm."""

from typing import Dict, Optional, Union
from qiskit.circuit import QuantumCircuit
import qiskit
#from qiskit.result import Counts
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
            self._pe_circuit = pe_circuit.compose(
                input_state_circuit,
                qubits=range(num_evaluation_qubits, num_evaluation_qubits + num_unitary_qubits),
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

    def _state_vector_phases(self):
        state_vec = self._result.get_statevector()
        evaluation_density_matrix = get_subsystem_density_matrix(
            state_vec,
            range(self._num_evaluation_qubits, self._num_evaluation_qubits + self._num_unitary_qubits)
        )
        phases = evaluation_density_matrix.diagonal().real # The diagonal is real
        return phases

    @property
    def phases(self):
        """ Returns the all phases and their frequencies """
        return self._phases

    def _compute_phases(self):
        if self._quantum_instance.is_statevector:
            phases = self._state_vector_phases()
        else:
            # return counts with keys sorted numerically
            counts = self._result.get_counts()
            phases = {k[::-1] : counts[k] for k in counts.keys()}
            phases = _sort_phases(phases)
            phases = qiskit.result.Counts(phases, memory_slots=counts.memory_slots, creg_sizes=counts.creg_sizes)

        self._phases = phases

    # If we reversed the bit order of evaluation register circuit (including the iqft) then we would avoid
    # the binary calculations in the case of the statevector simulator.
    def single_phase(self):
        """Return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding to a
        phase of 2pi. It is assumed that the input vector is an eigenvecter of the unitary so that
        the probability density is concentrated on one string (in the noiseless case).
        """
        if self._quantum_instance.is_statevector:
            idx = np.argmax(abs(self._phases)) # np.argmax ignores complex part of number. But, we take abs anyway
            binary_phase_string = np.binary_repr(idx, self._num_evaluation_qubits)[::-1]

        else:
            binary_phase_string = max(self._phases, key=self._phases.get)

        phase = bit_string_to_phase(binary_phase_string)
        return phase

    def _run(self):
        """Run the circuit and return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding
        to a phase of 2pi.
        """

        result = self._quantum_instance.execute(self._pe_circuit)
        self._result = result
        self._compute_phases()

    def filter_phases(self, cutoff):
        """ Return a dict of phases and frequencies (counts) only for frequencies (counts)
        larger than cutoff
        """
        if isinstance(self._phases, qiskit.result.Counts):
            counts = self._phases
            phases = {k : counts[k] for k in counts.keys() if counts[k] > cutoff}

        else:
            phases = {}
            for idx, amplitude in enumerate(self._phases):
                if amplitude > cutoff:
                    binary_phase_string = np.binary_repr(idx, self._num_evaluation_qubits)[::-1]
                    phases[binary_phase_string] = amplitude
            phases = _sort_phases(phases)

        return phases


def bit_string_to_phase(binary_string):
    n_qubits = len(binary_string)
    return int(binary_string, 2) / (2 ** n_qubits)
#    return int(binary_string[::-1], 2) / (2 ** n_qubits)


def _sort_phases(phases):
    ck = list(phases.keys())
    ck.sort(reverse=False) # Sorts in order integer encoded by binary string
    phases = {k : phases[k] for k in ck}
    return phases
