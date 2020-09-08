#from hamiltonian_pe import HamiltonianPE

from .phase_estimator_result import PhaseEstimatorResult


class HamiltonianPEResult(PhaseEstimatorResult):

    def __init__(self, num_evaluation_qubits, circuit_result, phase_estimation_scale,
                 phase_array=None, phase_dict=None):

        self._phase_estimation_scale = phase_estimation_scale

        super().__init__(num_evaluation_qubits, circuit_result, phase_array, phase_dict)


    def filter_phases(self, cutoff, scaled=True, as_float=True):

        phases = super().filter_phases(cutoff, as_float=as_float)
        if scaled:
            return self._phase_estimation_scale.scale_phases(phases)
        else:
            return phases


    def single_phase(self, scaled=True):
        phase = super().single_phase()
        if scaled:
            return self._phase_estimation_scale.scale_phase(phases)
        else:
            return phase
