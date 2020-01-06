# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Univariate Variational Distribution.
"""

from typing import Union, List
import numpy as np

from qiskit import ClassicalRegister
from qiskit.aqua.components.variational_forms import VariationalForm
from .univariate_distribution import UnivariateDistribution


class UnivariateVariationalDistribution(UnivariateDistribution):
    """
    The Univariate Variational Distribution.
    """

    def __init__(self,
                 num_qubits: int,
                 var_form: VariationalForm,
                 params: [Union[List[float], np.ndarray]],
                 low: float = 0,
                 high: float = 1) -> None:
        self._num_qubits = num_qubits
        self._var_form = var_form
        self.params = params
        if isinstance(num_qubits, int):
            probabilities = np.zeros(2 ** num_qubits)
        elif isinstance(num_qubits, float):
            probabilities = np.zeros(2 ** int(num_qubits))
        else:
            probabilities = np.zeros(2 ** sum(num_qubits))
        super().__init__(num_qubits, probabilities, low, high)

    def build(self, qc, q, q_ancillas=None, params=None):
        circuit_var_form = self._var_form.construct_circuit(self.params)
        qc.append(circuit_var_form.to_instruction(), q)

    def set_probabilities(self, quantum_instance):
        """
        Set Probabilities
        Args:
            quantum_instance (QuantumInstance): Quantum instance
        """
        qc_ = self._var_form.construct_circuit(self.params)

        # q_ = QuantumRegister(self._num_qubits)
        # qc_ = QuantumCircuit(q_)
        # self.build(qc_, None)

        if quantum_instance.is_statevector:
            pass
        else:
            c__ = ClassicalRegister(self._num_qubits, name='c')
            qc_.add_register(c__)
            qc_.measure(qc_.qregs[0], c__)
        result = quantum_instance.execute(qc_)
        if quantum_instance.is_statevector:
            result = result.get_statevector(qc_)
            values = np.multiply(result, np.conj(result))
            values = list(values.real)
        else:
            result = result.get_counts(qc_)
            keys = list(result)
            values = list(result.values())
            values = [float(v) / np.sum(values) for v in values]
            values = [x for _, x in sorted(zip(keys, values))]

        probabilities = values
        self._probabilities = np.array(probabilities)
