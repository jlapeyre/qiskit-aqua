import numpy

class PhaseEstimationScale():

    def __init__(self, bound):
        self._bound = bound

    @property
    def scale(self):
        return  numpy.pi / self._bound

    def scale_phase(self, phase):
        w = 2 * self._bound
        if phase <= 0.5:
            return phase * w
        else:
            return (phase - 1) * w


    def scale_phases(self, phases):
        w = 2 * self._bound
        if isinstance(phases, list):
            phases = [x * w if x <= 0.5 else (x - 1) * w for x in phases]
        else:
            phases = {(x * w if x <= 0.5 else (x - 1) * w) : phases[x] for x in phases.keys()}

        return phases


def from_pauli_sum(pauli_sum):
    """Create a PhaseEstimationScale from a SummedOp representing a sum of Pauli Operators"""
    bound = sum([abs(pauli_sum.coeff * pauli.coeff) for pauli in pauli_sum])
    return PhaseEstimationScale(bound)
