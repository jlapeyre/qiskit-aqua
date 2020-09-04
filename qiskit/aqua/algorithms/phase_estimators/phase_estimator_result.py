import numpy
import qiskit

# Maybe we want to use this abstract class
# from qiskit.aqua.algorithms import AlgorithmResult

#class PhaseEstimatorResult(AlgorithmResult):
class PhaseEstimatorResult():
    """
    """

    def __init__(self, num_evaluation_qubits, phases):
        self._num_evaluation_qubits = num_evaluation_qubits
        self._phases = phases

    @property
    def phases(self):
        """
        Returns the all phases and their frequencies. This is either a dict whose
        keys are bit strings and values are counts, or an array whose values correspond
        to weights on bit strings.
        """
        return self._phases

    # If we reversed the bit order of evaluation register circuit (including the iqft) then we would avoid
    # the binary calculations in the case of the statevector simulator.
    def single_phase(self):
        """
        Return the estimated phase as a number between 0.0 and 1.0, with 1.0 corresponding to a
        phase of 2pi. It is assumed that the input vector is an eigenvecter of the unitary so that
        the peak of the probability density occurs at the bit string that most closely approximates
        the true phase.
        """
        if isinstance(self._phases, numpy.ndarray):
            idx = numpy.argmax(abs(self._phases)) # numpy.argmax ignores complex part of number. But, we take abs anyway
            binary_phase_string = numpy.binary_repr(idx, self._num_evaluation_qubits)[::-1]
        else:
            binary_phase_string = max(self._phases, key=self._phases.get)

        phase = _bit_string_to_phase(binary_phase_string)
        return phase

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
                phases = {_bit_string_to_phase(k) : counts[k] for k in counts.keys() if counts[k] > cutoff}
            else:
                phases = {k : counts[k] for k in counts.keys() if counts[k] > cutoff}

        else:
            phases = {}
            for idx, amplitude in enumerate(self._phases):
                if amplitude > cutoff:
                    binary_phase_string = numpy.binary_repr(idx, self._num_evaluation_qubits)[::-1]
                    if as_float:
                        _key = _bit_string_to_phase(binary_phase_string)
                    else:
                        _key = binary_phase_string
                    phases[_key] = amplitude

            phases = _sort_phases(phases)

        return phases

    def phases_as_floats(self):
        """
        Return a dictionary whose keys are phases as floats in `[0, 1)` and values are
        freqencies or counts.
        """
        return self.filter_phases(0, as_float=True)


class PhaseEstimationEigenvalues():

    def __init__(self, bound):
        self._bound = bound

    @property
    def scale(self):
        return  numpy.pi / self._bound

    def eigenvalues(self, phases):
        w = 2 * self._bound
        if isinstance(phases, list):
            phases = [x * w if x <= 0.5 else (x - 1) * w for x in phases]
        else:
            phases = {(x * w if x <= 0.5 else (x - 1) * w) : phases[x] for x in phases.keys()}

        return phases


def _bit_string_to_phase(binary_string):
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
