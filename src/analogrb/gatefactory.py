from __future__ import annotations
from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.linalg import expm

from analogrb.bosonic import NNHamiltonian


def haar_unitary(dim: int) -> np.ndarray:
    """A `dim`x`dim` matrix drawn from the Haar random distribution,
    normalized to be from SU(dim).

    Meaning that :math:`U \\sim \\text{Haar}(\\text{dim})` and :math:`U\\in\\text{SU}(\\text{dim})`.
    The Haar random unitary is drawn via the QR decomposition and then normalized with
    :math:`\\text{det}(U)^{1/\\text{dim}}`.

    Args:
        dim (int) : Dimension of the Haar random matrix.

    Returns:
        (np.ndarray) Haar random SU(`dim`) matrix.
    """

    U = np.random.normal(0, 1, [dim, dim]) + 1j * np.random.normal(0, 1, [dim, dim])
    U, R = np.linalg.qr(U)
    D = np.diag(R)
    U = U @ np.diag(np.divide(D, np.abs(D)))
    return U 


class GateFactory:
    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_gate()

    def generate_gate(self):
        pass

    @property
    def params_to_save(self):
        return {'name': self.__class__.__name__}


class NNHGateFactory(GateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, time: float = 1.0) -> None:
        super().__init__()
        self.Hamiltonian = Hamiltonian
        self.time = time  # Time used for time evolution of Hamiltonian

    def generate_gate(self, time: Optional[float] = None) -> np.ndarray:
        U = expm(
            -self.draw_hamiltonian() * 1j * (time if time is not None else self.time)
        )
        return U

    def draw_hamiltonian(self) -> np.ndarray:
        return self.Hamiltonian.evaluate(
            *self.draw_hopping_entries(),
            self.draw_onside_interaction_entries() if self.Hamiltonian.interacting else None
        )

    def draw_hopping_entries(self) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    def draw_onside_interaction_entries(self) -> np.ndarray:
        pass
    
    @property
    def params_to_save(self):
        return  {
            **super().params_to_save,
            'Hamiltonian': self.Hamiltonian.params_to_save,
            'time': self.time
        }

class ModelMismatchNNHGateFactory(NNHGateFactory):
    def __init__(self, mismatch, Hamiltonian: NNHamiltonian, time: float = 1.0) -> None:
        super().__init__(Hamiltonian, time)
        self.mismatch = mismatch
    
    def generate_gate(self, time: float | None = None) -> ndarray:
        H = self.draw_hamiltonian()
        U = expm(-H * 1j * (time if time is not None else self.time))
        Herr = H + self.mismatch * H
        Uerr = expm(-Herr * 1j * (time if time is not None else self.time))
        return U, Uerr
    
    @property
    def params_to_save(self):
        return  {
            **super().params_to_save,
            'mismatch' : self.mismatch
        }

class MixupNNHGateFactory(NNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, time: float = 1.0) -> None:
        super().__init__(Hamiltonian, time)
    
    def generate_gate(self, time: float | None = None) -> ndarray:
        H = self.draw_actual_Hamiltonian()
        Herr = H + self.draw_err_Hamiltonian()
        U = expm(-H * 1j * (time if time is not None else self.time))
        Uerr = expm(-Herr * 1j * (time if time is not None else self.time))
        return U, Uerr
    
    @property
    def params_to_save(self):
        return  {
            **super().params_to_save,
        }
    
    def draw_err_Hamiltonian(self):
        return self.Hamiltonian.evaluate(
            self.draw_hopping_entries()[0] * 0,
            self.draw_hopping_entries()[1] * 0,
            self.draw_onside_interaction_entries()
        )

    def draw_actual_Hamiltonian(self):
        return self.Hamiltonian.evaluate(
            *self.draw_hopping_entries(),
             self.draw_onside_interaction_entries() * 0
        )
    
class UniformNNHGateFactory(NNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, time: float = 1.0, low: int = -1, high: int = 1) -> None:
        super().__init__(Hamiltonian, time)
        self.low = low
        self.high = high

    def draw_hopping_entries(self) -> tuple[np.ndarray, np.ndarray]:
        diag_entries = np.random.uniform(size=self.Hamiltonian.d, low=self.low, high=self.high)
        band_entries = np.random.uniform(
            size=self.Hamiltonian.band_dim, low=self.low, high=self.high
        ) + 1j * np.random.uniform(size=self.Hamiltonian.band_dim, low=self.low, high=self.high)
        return diag_entries, band_entries
    
    def draw_onside_interaction_entries(self) -> np.ndarray:
        return np.array(np.random.uniform(size=self.Hamiltonian.d, low=self.low, high=self.high), dtype=np.complex128)
    
    @property
    def params_to_save(self):
        return {
            **super().params_to_save,
            'low': self.low,
            'high': self.high
        }

class ErrHUniformNNHGateFactory(NNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, errH:np.ndarray, time: float = 1, ) -> None:
        super().__init__(Hamiltonian, time)
        self.errH = errH
    
    def generate_gate(self, time: float | None = None) -> ndarray:
        H = self.draw_actual_Hamiltonian()
        Herr = H + self.errH
        U = expm(-H * 1j * (time if time is not None else self.time))
        Uerr = expm(-Herr * 1j * (time if time is not None else self.time))
        return U, Uerr
    
    @property
    def params_to_save(self):
        return {
            **super().params_to_save,
            'errH': self.errH,
        }
        

class UnwantedInteractionsUniformNHHGateFactory(UniformNNHGateFactory, MixupNNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, time: float = 1, low: int = -1, high: int = 1, interr = -0.01) -> None:
        UniformNNHGateFactory.__init__(self, Hamiltonian, time, low, high)
        MixupNNHGateFactory.__init__(self, Hamiltonian, time)
        self.interr = interr
    
    def draw_onside_interaction_entries(self) -> np.ndarray:
        return np.zeros(self.Hamiltonian.d) + self.interr 

    @property
    def params_to_save(self):
        return {
            **super().params_to_save,
            'interr': self.interr
        }

class TooLessInteractionsUniformNHHGateFactory(UniformNNHGateFactory, MixupNNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, time: float = 1, low: int = -1, high: int = 1, interr_percent = 0.1) -> None:
        UniformNNHGateFactory.__init__(self, Hamiltonian, time, low, high)
        MixupNNHGateFactory.__init__(self, Hamiltonian, time)
        self.interr_percent = interr_percent
    
    @property
    def params_to_save(self):
        return {
            **super().params_to_save,
            'interr': self.interr_percent
        }
    def generate_gate(self, time: float | None = None) -> ndarray:
        hopping = self.draw_hopping_entries()
        noerrint = self.draw_onside_interaction_entries()
        H = self.Hamiltonian.evaluate(hopping[0], hopping[1], noerrint)
        Herr = self.Hamiltonian.evaluate(hopping[0], hopping[1], noerrint * self.interr_percent)
        U = expm(-H * 1j * (time if time is not None else self.time))
        Uerr = expm(-Herr * 1j * (time if time is not None else self.time))
        return U, Uerr
    
    


class MMUniformNNHGateFactory(UniformNNHGateFactory, ModelMismatchNNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, mismatch, time: float = 1, low: int = -1, high: int = 1) -> None:
        UniformNNHGateFactory.__init__(self, Hamiltonian, time, low, high)
        ModelMismatchNNHGateFactory.__init__(self, mismatch, Hamiltonian, time)

class UniformHoppingConstantOnsiteIntNNHGateFactory(UniformNNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, time: float = 1, low: int = -1, high: int = 1, onsite_value:float = 1.) -> None:
        super().__init__(Hamiltonian, time, low, high)
        self.onsite_value = onsite_value

    def draw_onside_interaction_entries(self) -> np.ndarray:
        return np.ones(self.Hamiltonian.d) * self.onsite_value 
    
    @property
    def params_to_save(self):
        return {
            **super().params_to_save,
            'onsite_value': self.onsite_value
        }

class SycamoreGateFactory(NNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, time: float = 25 * 10 ** (-9)) -> None:
        super().__init__(Hamiltonian, time)
        MHz = 10**6
        self.low_diag = -20 * MHz
        self.high_diag = 20 * MHz
        self.band_entries = -20 * MHz
        self.onside_interaction_entry = -1 * MHz

    def draw_hopping_entries(self) -> tuple[np.ndarray, np.ndarray]:
        band_entries = np.repeat(self.band_entries, self.Hamiltonian.band_dim)
        diag_entries = np.random.uniform(size=self.Hamiltonian.d, low = self.low_diag, high=self.high_diag)
        return diag_entries, band_entries
    
    def draw_onside_interaction_entries(self) -> np.ndarray:
        return np.ones(self.Hamiltonian.d, dtype=np.complex128) * self.onside_interaction_entry

class MMSycamoreGateFactory(SycamoreGateFactory, ModelMismatchNNHGateFactory):
    def __init__(self, Hamiltonian: NNHamiltonian, mismatch) -> None:
        SycamoreGateFactory.__init__(self, Hamiltonian)
        ModelMismatchNNHGateFactory.__init__(self, mismatch, Hamiltonian)

class HaarGateFactory(GateFactory):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    def generate_gate(self):
        return haar_unitary(self.d)
    
    @property
    def params_to_save(self):
        return {**super().params_to_save, 'd':self.d}