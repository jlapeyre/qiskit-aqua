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

class PhaseEstimationScale():
    """Set and use a bound on eigenvalues of a Hermitian operator in order
    to ensure phases are in the desired range and to convert measured phases into
    eigenvectors.

    The `bound` is set when constructing this class. Then the method `scale` is used
    to find the factor by which to scale the operator.
    """

    def __init__(self, bound):
        """
        Args:
            bound (float): an upper bound on the absolute value of the
        eigenvalues of a Hermitian operator. (The operator is not needed here.)
        """
        self._bound = bound

    @property
    def scale(self):
        """Return the scale factor by which a Hermitian operator must be multiplied
        so that the phase of the corresponding unitary is restricted to `[-pi, pi]`.
        This factor is computed from the bound on the absolute values of the eigenvalues
        of the operator. The methods `scale_phase` and `scale_phases` are used recover
        the eigenvalues corresponding the original (unscaled) Hermitian operator.
        """
        return  numpy.pi / self._bound


    def scale_phase(self, phi):
        """Convert a phase into an eigenvalue.

        The input phase `phi` corresponds to the eigenvalue of a unitary obtained by
        exponentiating a scaled Hermitian operator. Recall that the phase
        is obtained from `phi` as `2pi phi`. Furthermore the Hermitian operator
        was scaled so that `phi` is restricted to `[-1/2, 1/2]`, corresponding to
        phases in `[-pi, pi]`. But the values of `phi` read from the phase-readout
        register are in `[0, 1)`. Any value of `phi` greater than `1/2` corresponds
        to a raw phase of minus the complement with respect to `1`. After this possible
        shift, the phase is scaled by the inverse of the factor by which the
        Hermitian operator was scaled to recover the eigenvalue of the Hermitian
        operator.
        """
        w = 2 * self._bound
        if phi <= 0.5:
            return phi * w
        else:
            return (phi - 1) * w


    def scale_phases(self, phases):
        """Convert a list or dict of phases to eigenvalues.

        The values in the list, or keys in the dict, are values of `phi` and
        are converted as described in the description of `scale_phase`. In case
        `phases` is a dict, the values of the dict are passed unchanged.

        Args:
            phases: a list or dict of values of `phi`.
        """
        w = 2 * self._bound
        if isinstance(phases, list):
            phases = [x * w if x <= 0.5 else (x - 1) * w for x in phases]
        else:
            phases = {(x * w if x <= 0.5 else (x - 1) * w) : phases[x] for x in phases.keys()}

        return phases


def from_pauli_sum(pauli_sum):
    """Create a PhaseEstimationScale from a `SummedOp` representing a sum of Pauli Operators.

    It is assumed that the `SummedOp` `pauli_sum` is the sum of `PauliOp`s. The bound on
    the absolute value of the eigenvalues of the sum is obtained as the sum of the
    absolute values of the coefficients of the terms. This is the best bound available in
    the generic case. A `PhaseEstimationScale` object is instantiated using this bound.

    Args:
        pauli_sum:  A `SummedOp` whose terms are `PauliOp`s.

    Returns:
           A `PhaseEstimationScale` object
    """
    bound = sum([abs(pauli_sum.coeff * pauli.coeff) for pauli in pauli_sum])
    return PhaseEstimationScale(bound)
