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


# TODO: allow passing the bound on eigenvalues as input
class HamiltonianPE(PhaseEstimator):
    """Run the Quantum Phase Estimation algorithm to find the eigenvalues of a Hermitian operator.

    This class is nearly the same as `PhaseEstimator` differing only in that the input in that class
    is a unitary operator, whereas here is is a Hermitian operator from which a unitary will be obtained
    by scaling and exponentiating. The scaling is performed in order to prevent the phases from
    wrapping around `2pi`. This class uses and works together with `PhaseEstimationScale` manage scaling
    the Hamiltonian and the phases that are obtained by the QPE algorithm. This includes setting, or
    computing a bound on the eigenvalues of the operator, using this bound to obtaine a scale factor,
    scaling the operator, and shifting and scaling the measured phases to recover the eigenvalues.
    """
    def __init__(self,
                 num_evaluation_qubits,
                 hamiltonian,
                 evolution = None,
                 state_preparation = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None):

        """
        Args:
            num_evaluation_qubits:
            hamiltonian: a Hamiltonian or Hermitian operator
            state_preparation:
            quantum_instance:
        """


        self._hamiltonian = hamiltonian
        self._evolution = evolution

        unitary = self._get_unitary()

        super().__init__(num_evaluation_qubits,
                         unitary=unitary,
                         pe_circuit = None,
                         num_unitary_qubits = None,
                         state_preparation = state_preparation,
                         quantum_instance = quantum_instance)


    # Do we need this ?
    # @property
    # def phase_estimation_scale(self):
    #     return self._pe_scale


    def _get_unitary(self):
        pe_scale = phase_estimation_scale.from_pauli_sum(self._hamiltonian)
        self._pe_scale = pe_scale
        unitary = self._evolution.convert(pe_scale.scale * self._hamiltonian)
        if not isinstance(unitary, qiskit.QuantumCircuit):
            return unitary.to_circuit()
        else:
            return unitary


    def _run(self):
        """Run the circuit and return and return `HamiltonianPEResult`.
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
