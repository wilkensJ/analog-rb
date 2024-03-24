import sympy
import numpy as np
from analogrb.basis import fock
from scipy.special import binom
from scipy.linalg import expm


def symbolic_hopping(
    d: int, n: int, xdim: int = 1
) -> sympy.matrices.dense.MutableDenseMatrix:
    vs = np.array(fock(d, n))
    diff = -vs + vs[:, None, :]
    # Only ..., -1, ..., 1, ... contribute, exactly one particle hops.
    mask1 = np.sum(abs(diff), axis=2) == 2
    # Only nearest neighbour, ..., -1, 1, .... remain, for 2D the edges have to be set to zero.
    if xdim > 1:
        padded_dif = np.pad(
            diff, ((0, 0), (0, 0), (0, xdim - d % xdim if d % xdim else 0))
        ).reshape(*diff.shape[:-1], -1, xdim)
        mask2 = (
            np.sum(
                abs(
                    (
                        np.pad(padded_dif, ((0, 0), (0, 0), (0, 0), (1, 0)))
                        + np.pad(padded_dif, ((0, 0), (0, 0), (0, 0), (0, 1)))
                    )
                ),
                axis=(2, 3),
            )
            == 2
        )
        padded_dif = np.transpose(padded_dif, (0, 1, 3, 2))
        mask3 = (
            np.sum(
                abs(
                    (
                        np.pad(padded_dif, ((0, 0), (0, 0), (0, 0), (1, 0)))
                        + np.pad(padded_dif, ((0, 0), (0, 0), (0, 0), (0, 1)))
                    )
                ),
                axis=(2, 3),
            )
            == 2
        )
        mask2 = mask2 + mask3
    else:
        mask2 = (
            np.sum(
                abs(
                    (
                        np.pad(diff, ((0, 0), (0, 0), (1, 0)))
                        + np.pad(diff, ((0, 0), (0, 0), (0, 1)))
                    )
                ),
                axis=2,
            )
            == 2
        )
    all_masks = mask1 * mask2
    hopping = all_masks[:, :, None] * diff
    # Calculate the difference in particle number of each mode.
    diff_hs = hopping * vs
    # Calculate the prefactors of the bosonic creation and annihilation operators.
    take = np.sum(abs((diff_hs < 0) * diff_hs), axis=2)
    # Dont forget, that if before putting a particle there was 0, this has to contribute as 1!
    put = (np.sum(diff_hs != 0, axis=2) == 1) + np.sum(
        (diff_hs > 0) * (diff_hs + 1), axis=2
    )
    # Calculate the correct positioning of the hopping terms.
    first = np.argwhere(hopping == 1)[:, -1] + 1
    second = np.argwhere(hopping == -1)[:, -1] + 1
    hopping_terms = [f"h{a}{b}" for a, b in zip(first, second)]
    string_rep = np.full(hopping.shape[:-1], None, dtype="U16")
    string_rep[np.sum(hopping != 0, axis=2, dtype=bool)] = hopping_terms
    H = sympy.Matrix(
        *string_rep.shape,
        lambda i, j: sympy.sqrt((take * put)[i, j])
        * (
            sympy.var(string_rep[i, j])
            if i <= j
            else sympy.var(string_rep[j, i]).conjugate()
        ),
    )
    diagonal_hopping_str = [f"h{i}{i}" for i in range(1, len(vs) + 1)]
    diagonal = sympy.Matrix(
        *string_rep.shape,
        lambda i, j: sum(
            [
                sympy.Integer(vs[i][k]) * sympy.var(diagonal_hopping_str[k])
                for k in range(d)
            ]
        )
        if i == j
        else sympy.Integer(0),
    )
    return diagonal + H

def symbolic_OnSite_interaction(d:int, n:int) -> sympy.matrices.dense.MutableDenseMatrix:
    vs = np.array(fock(d, n))
    OnSite_2particle_str = [f"V{k}{k}{k}{k}" for k in range(1, d + 1)]
    OnSite_interaction_matrix = sympy.Matrix(
        len(vs), len(vs),
        lambda i,j: sum([
            sympy.Integer(vs[i][k] * (vs[i][k] - 1)) * sympy.var(OnSite_2particle_str[k])
            for k in range(d)
        ])
        if i == j
        else sympy.Integer(0),
    )
    return OnSite_interaction_matrix


def generate_hop_strings(d: int, xdim: int) -> tuple[list, list, list]:
    band_up = [f"h{i}{i + 1}" for i in range(1, d) if i % xdim] + [
        f"h{i}{i + xdim}" for i in range(1, d - xdim + 1)
    ]
    diagonal = [f"h{i}{i}" for i in range(1, d + 1)]
    return diagonal, band_up


def calculate_size_upperband(dim: int, xdim: int) -> int:
    ydim = dim // xdim
    horizontal = (xdim - 1) * ydim + dim % xdim - (1 if dim % xdim else 0)
    vertical = (ydim - 1) * xdim + dim % xdim
    return horizontal + vertical

def time_evolution(array:np.ndarray, time:float, dim_to_normalize:int):
    U = expm(-array * 1j * time)
    # Normalize the unitaries to be SU(d). (having determinant = 1).
    return U / (np.linalg.det(U) ** (1.0 / dim_to_normalize)).reshape(-1, 1)

class NNHamiltonian:
    def __init__(self, d, n, ydim = 1) -> None:
        self.d = d
        self.n = n
        self.ydim = ydim
        self.build()
        
    def build(self):
        self.fock_dim = int(binom(self.n + self.d - 1, self.n))
    
    def evaluate(self):
        pass
    
    @property
    def params_to_save(self):
        return {
            'name': self.__class__.__name__,
            'd': self.d, 
            'n': self.n,
            'ydim': self.ydim
        }

class NonintNNHamiltonian(NNHamiltonian):
    def __init__(self, d, n, ydim=1) -> None:
        super().__init__(d, n, ydim)
        self.interacting = False
    
        
    def build(self):
        super().build()
        self.band_dim = calculate_size_upperband(self.d, self.ydim)
        self.sym_H_hopping = symbolic_hopping(self.d, self.n, self.ydim)
        hstrings_diag, hstrings_band = generate_hop_strings(self.d, self.ydim)
        variables = [sympy.symbols(h) for h in hstrings_diag + hstrings_band]
        self.set_H_hopping = sympy.lambdify(variables, self.sym_H_hopping, modules="numpy")
       
    
    def evaluate(self, hopping_diag, hopping_band, *args):
        return self.set_H_hopping(*hopping_diag, *hopping_band)
    
    def show(self):
        return self.sym_H_hopping
        

class OnSiteIntNNHamiltonian(NonintNNHamiltonian):
    def __init__(self, d, n, ydim=1) -> None:
        super().__init__(d, n, ydim)
        self.interacting = True
    
    def build(self):
        super().build()
        self.sym_H_interacting = symbolic_OnSite_interaction(self.d, self.n)
        variables = [sympy.symbols(f"V{k}{k}{k}{k}") for k in range(1, self.d + 1)]
        self.set_H_interacting = sympy.lambdify(variables, self.sym_H_interacting, modules="numpy")

    def evaluate(self, hopping_diag:np.ndarray, hopping_band:np.ndarray, interacting_onsite:np.ndarray) -> np.ndarray:
        """Sets the hopping terms (diagonal and band) and the interaction terms to real values.

        Args:
            hopping_diag (np.ndarray): As many as there are sites.
            hopping_band (np.ndarray): As many as there are connections, depends on the structure of the lattice.
            interacting_onsite (np.ndarray): As many as there are sites

        Returns:
            np.ndarray: The Hamiltonian in matrix form.
        """
        
        return super().evaluate(hopping_diag, hopping_band) + self.set_H_interacting(*interacting_onsite)
    
    def show(self):
        return super().show() + self.sym_H_interacting