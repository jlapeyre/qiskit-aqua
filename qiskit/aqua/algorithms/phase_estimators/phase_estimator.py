"""The Quantum Phase Estimation Algorithm."""

from typing import Dict, Optional, Union
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

    Blah
    """

    def __init__(self,
                 phase_estimation: PhaseEstimation,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None):

        self._phase_estimation = phase_estimation
        super().__init__(quantum_instance)

    # If we reversed the bit order of evaluation register circuit (including the iqft) then we would avoid
    # the binary calculations in the case of the statevector simulator.
    def _compute_phase(self, result):
        """Return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding to a
        phase of 2pi. It is assumed that the input vector is an eigenvecter of the unitary so that
        the probability density is concentrated on one string (in the noiseless case).
        """
        num_evaluation_qubits = self._phase_estimation.num_evaluation_qubits
        if self._quantum_instance.is_statevector:
            state_vec = result.get_statevector()
            evaluation_density_matrix = get_subsystem_density_matrix(
                state_vec,
                range(num_evaluation_qubits, num_evaluation_qubits + self._phase_estimation.num_unitary_qubits)
            )
            diag = evaluation_density_matrix.diagonal()
            idx = np.argmax(abs(diag)) # np.argmax ignores complex part of number. But, we take abs anyway
            binary_phase_string = np.binary_repr(idx, num_evaluation_qubits)

        else:
            counts = dict(result.get_counts())
            binary_phase_string = max(counts, key=counts.get)

        phase =  int(binary_phase_string[::-1], 2) / (2 ** num_evaluation_qubits)

        return phase

    def _run(self) -> Dict:
        """Run the circuit and return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding
        to a phase of 2pi.
        """
        circ = self._phase_estimation
        if not self._quantum_instance.is_statevector:
            # Measure only the evaluation qubits.
            creg = ClassicalRegister(circ.num_evaluation_qubits, 'meas')
            circ.add_register(creg)
            circ.barrier()
            circ.measure(range(circ.num_evaluation_qubits), range(circ.num_evaluation_qubits))

        result = self._quantum_instance.execute(circ)
        phase = self._compute_phase(result)

        return phase, result
