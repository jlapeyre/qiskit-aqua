# -*- coding: utf-8 -*-

"""The Quantum Phase Estimation Algorithm."""

from typing import Dict, Optional, Union
from qiskit.circuit.library import PhaseEstimation
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm


class PhaseEstimator(QuantumAlgorithm):
    """The Quantum Phase Estimation algorithm.

    Blah
    """

    def __init__(self,
                 phase_estimation: PhaseEstimation,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None):

        self._phase_estimation = phase_estimation
        super().__init__(quantum_instance)


    def _run(self) -> Dict:
        return {}
