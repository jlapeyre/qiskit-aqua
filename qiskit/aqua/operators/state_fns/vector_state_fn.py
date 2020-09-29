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

""" StateVector Class """


from typing import Union, Set, Optional, Dict, List
import numpy as np
from qiskit import QuantumCircuit

from qiskit import quantum_info
from qiskit.circuit import ParameterExpression
from qiskit.aqua import aqua_globals

from ..operator_base import OperatorBase
from .state_fn import StateFn
from ..list_ops.list_op import ListOp
from ...utils import arithmetic


class StateVector(StateFn):
    """ A class for state functions and measurements which are defined in vector
    representation, and stored using Terra's ``Statevector`` class.
    """

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[list, np.ndarray, quantum_info.Statevector] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
        Args:
            primitive: The ``quantum_info.Statevector``, NumPy array, or list, which defines the behavior of
                the underlying function.
            coeff: A coefficient multiplying the state function.
            is_measurement: Whether the StateFn is a measurement operator
        """
        # Lists and Numpy arrays representing statevectors are stored
        # in quantum_info.Statevector objects for easier handling.
        if isinstance(primitive, (np.ndarray, list)):
            primitive = quantum_info.Statevector(primitive)

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def primitive_strings(self) -> Set[str]:
        return {'Vector'}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive.dims())

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over statefns with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, StateVector) and self.is_measurement == other.is_measurement:
            # Covers MatrixOperator, quantum_info.Statevector and custom.
            return StateVector((self.coeff * self.primitive) + (other.primitive * other.coeff),
                                 is_measurement=self._is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import SummedOp
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return StateVector(self.primitive.conjugate(),
                             coeff=np.conj(self.coeff),
                             is_measurement=(not self.is_measurement))

    def permute(self, permutation: List[int]) -> 'StateVector':
        new_self = self
        new_num_qubits = max(permutation) + 1

        if self.num_qubits != len(permutation):
            # raise AquaError("New index must be defined for each qubit of the operator.")
            pass
        if self.num_qubits < new_num_qubits:
            # pad the operator with identities
            new_self = self._expand_dim(new_num_qubits - self.num_qubits)
        qc = QuantumCircuit(new_num_qubits)

        # extend the permutation indices to match the size of the new matrix
        permutation \
            = list(filter(lambda x: x not in permutation, range(new_num_qubits))) + permutation

        # decompose permutation into sequence of transpositions
        transpositions = arithmetic.transpositions(permutation)
        for trans in transpositions:
            qc.swap(trans[0], trans[1])
        from .. import CircuitOp
        matrix = CircuitOp(qc).to_matrix()
        vector = new_self.primitive.data
        return StateVector(primitive=matrix.dot(vector),
                             coeff=self.coeff,
                             is_measurement=self.is_measurement)

    def to_dict_fn(self) -> StateFn:
        """Creates the equivalent state function of type DictStateVector.

        Returns:
            A new DictStateVector equivalent to ``self``.
        """
        from .dict_state_fn import DictStateVector

        num_qubits = self.num_qubits
        new_dict = {format(i, 'b').zfill(num_qubits): v for i, v in enumerate(self.primitive.data)}
        return DictStateVector(new_dict, coeff=self.coeff, is_measurement=self.is_measurement)

    def _expand_dim(self, num_qubits: int) -> 'StateVector':
        primitive = np.zeros(2**num_qubits, dtype=complex)
        return StateVector(self.primitive.tensor(primitive),
                             coeff=self.coeff,
                             is_measurement=self.is_measurement)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, StateVector):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        return self.primitive.to_operator().data * self.coeff

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        vec = self.primitive.data * self.coeff

        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        return self

    def to_circuit_op(self) -> OperatorBase:
        """ Return ``StateFnCircuit`` corresponding to this StateFn."""
        from .circuit_state_fn import StateCircuit
        csfn = StateCircuit.from_vector(self.to_matrix(massive=True)) * self.coeff  # type: ignore
        return csfn.adjoint() if self.is_measurement else csfn

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('StateVector' if not self.is_measurement
                                   else 'MeasurementVector', prim_str)
        else:
            return "{}({}) * {}".format('StateVector' if not self.is_measurement
                                        else 'MeasurementVector',
                                        prim_str,
                                        self.coeff)

    # pylint: disable=too-many-return-statements
    def eval(self,
             front: Optional[Union[str, Dict[str, complex], np.ndarray, OperatorBase]] = None
             ) -> Union[OperatorBase, float, complex]:
        if front is None:  # this object is already a StateVector
            return self

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                   for front_elem in front.oplist])

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..operator_globals import EVAL_SIG_DIGITS
        from .operator_state_fn import DensityOperator
        from .circuit_state_fn import StateCircuit
        from .dict_state_fn import DictStateVector
        if isinstance(front, DictStateVector):
            return np.round(sum([v * self.primitive.data[int(b, 2)] * front.coeff  # type: ignore
                                 for (b, v) in front.primitive.items()]) * self.coeff,
                            decimals=EVAL_SIG_DIGITS)

        if isinstance(front, StateVector):
            # Need to extract the element or np.array([1]) is returned.
            return np.round(np.dot(self.to_matrix(), front.to_matrix())[0],
                            decimals=EVAL_SIG_DIGITS)

        if isinstance(front, StateCircuit):
            # Don't reimplement logic from StateCircuit
            return np.conj(
                front.adjoint().eval(self.adjoint().primitive)) * self.coeff  # type: ignore

        if isinstance(front, DensityOperator):
            return front.adjoint().eval(self.primitive) * self.coeff  # type: ignore

        return front.adjoint().eval(self.adjoint().primitive).adjoint() * self.coeff  # type: ignore

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        deterministic_counts = self.primitive.probabilities_dict()
        # Don't need to square because probabilities_dict already does.
        probs = np.array(list(deterministic_counts.values()))
        unique, counts = np.unique(aqua_globals.random.choice(list(deterministic_counts.keys()),
                                                              size=shots,
                                                              p=(probs / sum(probs))),
                                   return_counts=True)
        counts = dict(zip(unique, counts))
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))
