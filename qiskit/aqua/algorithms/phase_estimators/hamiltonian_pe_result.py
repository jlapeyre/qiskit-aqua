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

"""Result from running HamiltonianPE"""

#from hamiltonian_pe import HamiltonianPE

from typing import Dict, List, Union
from qiskit.result import Result
from .phase_estimator_result import PhaseEstimatorResult
from .phase_estimation_scale import PhaseEstimationScale

class HamiltonianPEResult(PhaseEstimatorResult):
    """Store and manipulate results from running `HamiltonianPE`.

    This API of this class is nearly the same as `PhaseEstimatorResult`, differing only in
    the presence of an additional keyword argument in the methods. If `scaled`
    is `False`, then the phases are not translated and scaled to recover the
    eigenvalues of the Hamiltonian. Instead `phi` in `[0, 1)` is returned
    as is the case when then unitary is not derived from a Hamiltonian.
    """
    def __init__(self, num_evaluation_qubits: int, circuit_result: Result,
                 phase_estimation_scale: PhaseEstimationScale,
                 phase_array: List = None, phase_dict: Dict = None) -> None:

        self._phase_estimation_scale = phase_estimation_scale

        super().__init__(num_evaluation_qubits, circuit_result, phase_array, phase_dict)

    # pylint: disable=arguments-differ
    def filter_phases(self, cutoff: float = 0.0, scaled: bool = True,# type: ignore
                      as_float: bool = True) -> Union[Dict, List]: # returns a dict
        """Filter phases as does `PhaseEstimatorResult.filter_phases`, with
        the addition that `phi` is shifted and translated to return eigenvalues
        of the Hamiltonian.

        Args:
            cutoff (float): Minimum weight of number of counts required to keep a bit string.
                          The default value is `0.0`.
            scaled (bool): If False, return `phi` in `[0, 1)` rather than the eigenvalues of
                         the Hamiltonian.
            as_float (bool): If `True`, returned keys are floats in `[0.0, 1)`. If `False`
                      returned keys are bit strings.

        Returns:
              A dict of filtered phases.
        """
        phases = super().filter_phases(cutoff, as_float=as_float)
        if scaled:
            return self._phase_estimation_scale.scale_phases(phases)
        else:
            return phases


    def single_phase(self, scaled: bool = True) -> float:
        """Return the estimated phase as a number between 0.0 and 1.0, with 1.0.

        This method is similar to `PhaseEstimatorResult.single_phase`.

        Args:
            scaled (bool): If `False` return `phi` in `[0, 1)` rather than the most
                         frequent eigenvalue of the Hamiltonian.

        Returns:
            The estimated phase as a number between 0.0 and 1.0, with 1.0.
        """
        phase = super().single_phase()
        if scaled:
            return self._phase_estimation_scale.scale_phase(phase)
        else:
            return phase
