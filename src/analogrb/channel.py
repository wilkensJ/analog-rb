from __future__ import annotations
from typing import Optional, List
import numpy as np

import numpy as np


class Channel:
    def __init__(self) -> None:
        pass

    def apply(self, density_matrix: np.ndarray) -> None:
        pass

    def __call__(self, density_matrix: np.ndarray) -> np.ndarray:
        return self.apply(density_matrix)
    
    @property
    def params_to_save(self):
        return {'name': self.__class__.__name__}

class IdChannel(Channel):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        return density_matrix


class DepolarizingChannel(Channel):
    def __init__(self, depol_param: float) -> None:
        super().__init__()
        self.p = depol_param

    @property
    def depol_param(self):
        return self.p
    
    def apply(self, density_matrix: np.ndarray, **kwds) -> np.ndarray:
        dim = len(density_matrix)
        return (1 - self.depol_param) * density_matrix + self.depol_param * np.eye(
            dim
        ) / dim
        
    @property
    def params_to_save(self):
        d =  super().params_to_save
        add_d = {'depol_param': self.depol_param}
        return {**d, **add_d}

class RandUniformDepolarizingChannel(DepolarizingChannel):
    def __init__(self, low:float, high:float) -> None:
        self.low = low
        self.high = high
        super().__init__((high - low)/2.)
    
    @property
    def depol_param(self):
        return np.random.uniform(low=self.low, high=self.high)
    
    @property
    def params_to_save(self):
        d =  super().params_to_save
        add_d = {'low': self.low, 'high':self.high}
        return {**d, **add_d}

class EllensNoiseChannel(Channel):
    def __init__(self, some_parameter, another_parameter) -> None:
        super().__init__()
        self.some_parameter = some_parameter
        self.another_parameter = another_parameter
        
    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        # do something with the density matrix 
        # use self.some_parameters
        random_number = np.random.uniform(low=self.some_parameter, high=self.another_parameter)
        altered_density_matrix = random_number * density_matrix
        return altered_density_matrix
        

class UnitaryChannel(Channel):
    def __init__(
        self, gate: Optional[np.ndarray] = None, unitary: Optional[np.ndarray] = None
    ) -> None:
        super().__init__()
        self.gate = gate
        self._unitary = unitary

    @property
    def unitary(self):
        if self._unitary is not None:
            return self._unitary
        elif self.gate is not None:
            return np.kron(self.gate, self.gate.conj())
        else:
            raise AttributeError("Attributes gate and _unitary are both None.")

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            density_matrix (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        evolved_vectorized = self.unitary @ density_matrix.T.reshape(-1, 1)
        return evolved_vectorized.reshape(density_matrix.shape).T
    
    @property
    def params_to_save(self):
        d =  super().params_to_save
        add_d = {'gate_real': np.real(self.gate).tolist(), 'gate_imag':np.imag(self.gate).tolist()}
        return {**d, **add_d}

class StandardMeasurementChannel(Channel):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.build()

    def build(self):
        states = np.diag(np.ones(self.dim))
        self.measurement_matrices = np.einsum("ij,il->ijl", states, states)
        states = states.reshape(self.dim, 1, self.dim)
        self.measurement_vectors = np.einsum("ijk,ilm->ijklm", states, states).reshape(
            self.dim, -1
        )

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """The trace of the products with 'density_matrix' and each
        measurement matrix is calculated.

        :math:`{M_i}_{i=1}^{'d'}` the set of measurement matrices and :math:`D? the density matrix,
        then the outcome is calculated as:

        .. math::
            outcomes_i = Tr(A_i.dot(D)), for i=1,...,d

        Args:
            density_matrix[np.ndarray]: A density matrix.

        Returns:
            measurement_outcomes[np.ndarray]: The probabilities for each outcome in standard basis.
        """

        measurement_outcomes = np.trace(
            self.measurement_matrices @ density_matrix, axis1=1, axis2=2
        )
        return measurement_outcomes

