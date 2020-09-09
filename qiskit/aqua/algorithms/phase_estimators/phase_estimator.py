# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


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
    """Run the Quantum Phase Estimation algorithm.

    This runs a version of QPE with a multi-qubit register for reading the phase [1]. The main inputs are
    the number of qubits in the phase-reading register, a state preparation circuit to prepare an input state,
    and either
    1) A unitary that will act on the the input state, or
    2) A quantum-phase-estimation circuit in which the unitary is already embedded.
    In case 1), an instance of `qiskit.circuit.PhaseEstimation`, a QPE circuit, containing the input unitary
    will be constructed. After construction, the QPE circuit is run on a backend via the `run` method, and
    the frequencies or counts of the phases represented by bitstrings are recorded. The results are returned
    as an instance of qiskit.algorithms.phase_estimator_result.PhaseEstimatorResult.

    If the input state is an eigenstate of the unitary, then in the ideal case, all probability is concentrated
    on the bitstring corresponding to the eigenvalue of the input state. If the input state is a superposition
    of eigenstates, then each bitstring is measured with a probability corresponding to its weight in the
    superposition. In addition, if the phase is not representable exactly by the phase-reading register, the
    probability will be spread across bitstrings, with an amplitude that decreases with distance from the
    bitstring most closely approximating the phase.

    **Reference:**

    [1]: Michael A. Nielsen and Isaac L. Chuang. 2011.
         Quantum Computation and Quantum Information: 10th Anniversary Edition (10th ed.).
         Cambridge University Press, New York, NY, USA.
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
        """Return the phase estimation circuit. """
        return self._pe_circuit

    def _compute_phases(self, circuit_result):
        """Compute frequencies/counts of phases from the result of running the QPE circuit.

        How the frequencies are computed depends on whether the backend computes amplitude or
        samples outcomes.

        1) If the backend is a statevector simulator, then the reduced density
        matrix of the phase-reading register is computed from the combined phase-reading-
        and input-state registers. The elements of the diagonal `(i, i)` give the
        probability to measure the each of the states `i`. The index `i` expressed as a binary
        integer with the LSB rightmost gives the state of the phase-reading register with
        the LSB leftmost when interpreted as a phase. In order to maintain the compact
        representation, the phases are maintained as decimal integers.
        They may be converted to other forms via the results object, `PhaseEstimatorResult`
        or `HamiltonianPEResult`.

         2) If the backend samples bitstrings, then the counts are first retreived as a dict.
        The binary strings (the keys) are then reversed so that the LSB is rightmost and the counts
        are converted to frequencies. Then the keys are sorted according to increasing phase,
        so that they can be easily understood when displaying or plotting a histogram.

        Args:
            circuit_result: the result object returned by the backend that rant the QPE circuit.

        Returns:
               Either a dict or numpy.ndarray representing the frequencies of the phases.
        """
        if self._quantum_instance.is_statevector:
            state_vec = circuit_result.get_statevector()
            evaluation_density_matrix = get_subsystem_density_matrix(
                state_vec,
                range(self._num_evaluation_qubits, self._num_evaluation_qubits + self._num_unitary_qubits)
            )
            phases = evaluation_density_matrix.diagonal().real # The diagonal is real, so imaginary part should be zero.
        else:
            # return counts with keys sorted numerically
            num_shots = circuit_result.results[0].shots
            counts = circuit_result.get_counts()
            phases = {k[::-1] : counts[k] / num_shots for k in counts.keys()}
            phases = _sort_phases(phases)
            phases = qiskit.result.Counts(phases, memory_slots=counts.memory_slots, creg_sizes=counts.creg_sizes)

        return phases


    def _run(self):
        """Run the circuit and return and return `PhaseEstimatorResult`.


        Returns:
               An instance of qiskit.algorithms.phase_estimator_result.PhaseEstimatorResult.
        """

        self._add_classical_register()
        circuit_result = self._quantum_instance.execute(self._pe_circuit)
        phases = self._compute_phases(circuit_result)
        if isinstance(phases, numpy.ndarray):
            return PhaseEstimatorResult(self._num_evaluation_qubits, phase_array = phases, circuit_result=circuit_result)
        else:
            return PhaseEstimatorResult(self._num_evaluation_qubits, phase_dict = phases, circuit_result=circuit_result)
