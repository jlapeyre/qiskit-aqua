"""The Quantum Phase Estimation Algorithm."""

from typing import Dict, Optional, Union
from qiskit.circuit import QuantumCircuit
import qiskit
#from qiskit.result import Counts
from qiskit.circuit.library import PhaseEstimation
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
                 num_evaluation_qubits,
                 pe_circuit = None,
                 num_unitary_qubits = None,
                 unitary = None,
                 input_state_circuit = None,
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
            input_state_circuit: The circuit that prepares the state whose eigenphase will be measured.
                                 If this parameter is ommited, no preparation circuit will be run and
                                 input state will be the all-zero state in the computational basis.
            quantum_instance: The quantum instance on which the circuit will be run.
        """

        if unitary is None:
            if pe_circuit is None:
                raise ValueError('One one of `unitary` and `pe_circuit` may be `None`.')
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

        if not input_state_circuit is None:
            self._pe_circuit = pe_circuit.compose(
                input_state_circuit,
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
    def phases(self):
        """
        Returns the all phases and their frequencies. This is either a dict whose
        keys are bit strings and values are counts, or an array whose values correspond
        to weights on bit strings.
        """
        return self._phases

    def _compute_phases(self):
        if self._quantum_instance.is_statevector:
            state_vec = self._result.get_statevector()
            evaluation_density_matrix = get_subsystem_density_matrix(
                state_vec,
                range(self._num_evaluation_qubits, self._num_evaluation_qubits + self._num_unitary_qubits)
            )
            phases = evaluation_density_matrix.diagonal().real # The diagonal is real
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
        """
        Return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding to a
        phase of 2pi. It is assumed that the input vector is an eigenvecter of the unitary so that
        the peak of the probability density occurs at the bit string that most closely approximates
        the true phase.
        """
        if self._quantum_instance.is_statevector:
            idx = np.argmax(abs(self._phases)) # np.argmax ignores complex part of number. But, we take abs anyway
            binary_phase_string = np.binary_repr(idx, self._num_evaluation_qubits)[::-1]

        else:
            binary_phase_string = max(self._phases, key=self._phases.get)

        phase = bit_string_to_phase(binary_phase_string)
        return phase

    def _run(self):
        """
        Run the circuit and return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding
        to a phase of 2pi.
        """

        result = self._quantum_instance.execute(self._pe_circuit)
        self._result = result
        self._compute_phases()

    def filter_phases(self, cutoff, as_float=False):
        """
        Return a dict whose keys are phases and values are frequencies (counts)
        keeping only frequencies (counts) larger than cutoff. It is assumed that
        the `run` method has been called so that the phases have been computed.
        When using a noiseless, shot-based simulator to read a single phase that can
        be represented exactly by `num_evaluation_qubits`, all the weight will
        be concentrated on a single phase. In all other cases, many, or all, bit
        strings will have non-zero weight. This method is useful for filtering
        out these uninteresting bit strings.

        Args:
            cutoff: minimum weight of number of counts required to keep a bit string.
            as_float: If `True`, returned keys are floats in `[0.0, 1)`. If `False`
                      returned keys are bit strings.
        """
        if isinstance(self._phases, qiskit.result.Counts):
            counts = self._phases
            if as_float:
                phases = {bit_string_to_phase(k) : counts[k] for k in counts.keys() if counts[k] > cutoff}
            else:
                phases = {k : counts[k] for k in counts.keys() if counts[k] > cutoff}

        else:
            phases = {}
            if as_float:
                for idx, amplitude in enumerate(self._phases):
                    if amplitude > cutoff:
                        binary_phase_string = np.binary_repr(idx, self._num_evaluation_qubits)[::-1]
                        phases[bit_string_to_phase(binary_phase_string)] = amplitude
            else:
                for idx, amplitude in enumerate(self._phases):
                    if amplitude > cutoff:
                        binary_phase_string = np.binary_repr(idx, self._num_evaluation_qubits)[::-1]
                        phases[binary_phase_string] = amplitude

            phases = _sort_phases(phases)

        return phases


def bit_string_to_phase(binary_string):
    """
    Convert bit string to phase. It is assumed that the bit string is correctly padded
    and that the order of the bits has been reversed relative to their order when the counts
    were recorded.
    """
    n_qubits = len(binary_string)
    return int(binary_string, 2) / (2 ** n_qubits)

def _sort_phases(phases):
    """
    Sort dict whose keys are bit strings representing phases and whose values are frequencies by bit string.
    The bit strings are sorted according to increasing phase.
    This relies on python preserving insertion order when building dicts.
    """
    ck = list(phases.keys())
    ck.sort(reverse=False) # Sorts in order integer encoded by binary string
    phases = {k : phases[k] for k in ck}
    return phases
