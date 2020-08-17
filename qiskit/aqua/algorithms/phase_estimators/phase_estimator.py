"""The Quantum Phase Estimation Algorithm."""

from typing import Dict, Optional, Union
from qiskit.circuit.library import PhaseEstimation
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
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]]):

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
            bin_frac = np.binary_repr(idx, num_evaluation_qubits)

        else:
            counts = dict(result.get_counts())
            bin_frac = max(counts, key=counts.get)

        phase =  int(bin_frac[::-1], 2) / (2 ** num_evaluation_qubits)

        return phase

    def _run(self) -> Dict:
        """Run the circuit and return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding
        to a phase of 2pi.
        """
        circ = self._phase_estimation
        if not self._quantum_instance.is_statevector:
            # Measure only the evaluation qubits.
            new_creg = circ._create_creg(circ.num_evaluation_qubits, 'meas')
            circ.add_register(new_creg)
            circ.barrier()
            for b in range(circ.num_evaluation_qubits):
                circ.measure(b, b)

        job = execute(circ, self.backend)
        phase = self._compute_phase(job.result())

        return phase, job.result()
