import subprocess
import numpy as np
import os
from scipy.special import binom
from copy import deepcopy
from typing import Union
from analogrb.save_load import MODULE_DIR

from analogrb import basis


CPP_EXECUTABLE = f"{MODULE_DIR()}/clebschgordan.out"
PROJECTORS_DIR = lambda d, n: f"{MODULE_DIR()}/projectors/{d}modes_{n}particles/"
PROJECTORS_FILE = lambda d, n, irrep3: f"{PROJECTORS_DIR(d, n)}projector_{'-'.join(map(str, irrep3))}.txt"
CONJ_SYM_IRREP = lambda d, n: [n] * (d-1) + [0]
IRREPS = lambda d, n: [[2 * l] + [l] * (d - 2) + [0] for l in range(1, n + 1)]
DIM_IRREPS = lambda d, n: [(d + 2 * l - 1) / (d -1) * int(binom(d + l - 2, l)) ** 2 for l in range(n, 0, -1)]

def dim_irrep(irrep: Union[list, np.ndarray]) -> int:
    irrep = np.array(irrep)
    kprime, k = np.triu_indices(len(irrep), 1)
    dim = np.prod(1 + (irrep[k] - irrep[kprime]) / (kprime - k))
    return int(np.round(dim,0))

def decrement(gpattern):
    gpattern = deepcopy(gpattern)
    k, l= 1, 1
    while l < gpattern.N and gpattern[k, l] == gpattern[k + 1, l + 1]:
        k -= 1
        if k == 0:
            l += 1
            k = l
    sign = -1
    gpattern[k, l] -= 1
    if l != N:
        while k != 1 or l != 1:
            k += 1
            if k > l:
                k = 1
                l -= 1
            sign *= (-1) ** (gpattern[k, l + 1] - gpattern[k, l])
            gpattern[k, l] = gpattern[k, l + 1]
    return gpattern, sign


class Pattern:
    def __init__(self, irrep):
        self.irrep = np.array(irrep)
        self.N = len(self.irrep)
        self.index = dim_irrep(self.irrep)
        self.sign = 1
        self.elem = np.zeros((self.N, self.N), dtype = int)
        self.build()
        
    def build(self):
        a = np.argmax(self.irrep == 0)
        self.elem[:, 0:a] = self.irrep[0]
        self.elem[::, ::-1][np.tril_indices(self.elem.shape[0], k=-1)] = 0
    
    def __getitem__(self, indices):
        a, b = indices
        return self.elem[self.N - b , a - 1]

    def __setitem__(self, indices, gvalue):
        a, b = indices
        self.elem[self.N - b , a - 1] = gvalue
    
    def decrement(self):
        k, l= 1, 1
        while l < self.N and self[k, l] == self[k + 1, l + 1]:
            k -= 1
            if k == 0:
                l += 1
                k = l
        self.sign *= -1
        self[k, l] -= 1
        if l != self.N:
            while k != 1 or l != 1:
                k += 1
                if k > l:
                    k = 1
                    l -= 1
                self.sign *= (-1) ** (self[k, l + 1] - self[k, l])
                self[k, l] = self[k, l + 1]
        
    def __sub__(self, gvalue):
        assert gvalue >= 0
        p = deepcopy(self)
        for k in range(gvalue):
            p.decrement()
        return p

def save_clebsch_gordan_coefficients(d, n):

    path_name = PROJECTORS_DIR(d, n)
    if not os.path.isdir(path_name):
        os.makedirs(path_name)

    symirrep = [n] + [0] * (d-1)
    conj_symirrep = CONJ_SYM_IRREP(d, n)
    irreps = IRREPS(d, n)

    symirrep_str = ' '.join(map(str, symirrep))
    conj_symirrep_str = ' '.join(map(str, conj_symirrep))
    irreps_str = [' '.join(map(str, i)) for i in irreps]
    inputs = [ 
        f'''5
{d}
{symirrep_str}
{conj_symirrep_str}
{irreps_str[k]}
1
{path_name}cgc_{'-'.join(map(str, irreps[k]))}.txt
0'''
    for k in range(n)
    ]

    for input_text in inputs:
        process = subprocess.Popen(CPP_EXECUTABLE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        a, b = process.communicate(input=input_text)


def compute_minus_positions(irrep: Union[list, np.ndarray]) -> list[int]:
    p = Pattern(np.array(irrep))
    signs_list = [p.sign]
    for k in range(p.index - 1):
        p -= 1
        signs_list.append(p.sign)
    return signs_list

def build_projectors_noninteracting(d, n):

    
    sym_basis = basis.dicke(d, n)
    sign_values = compute_minus_positions(CONJ_SYM_IRREP(d, n))
    conj_basis = np.array(basis.dicke(d,n))
    conj_basis *= np.array(sign_values)[..., None]
    conj_basis = conj_basis[::-1]
    
    

    transition_matrix_T = np.array(basis.dicke(d, n))
    left_transition = np.kron(transition_matrix_T.conj(), transition_matrix_T)
    right_transition = np.kron(transition_matrix_T.T, transition_matrix_T.T.conj())

    for irrep3 in IRREPS(d,n):
        
        with open(f"{PROJECTORS_DIR(d, n)}cgc_{'-'.join(map(str, irrep3))}.txt", "r") as file:
            cgc_table = np.loadtxt(file, skiprows=2)

        # The combinations of basis vectors are numbered in the .txt file, extract how many
        # there will be in the end.
        number_new_basis_vectors = int(cgc_table[:,2][-1])
        b3_basis = np.zeros((number_new_basis_vectors, len(sym_basis[0]) ** 2))
        
        for b1, b2, b3, cgc in cgc_table:
            b3_basis[int(b3)-1] +=  cgc * np.kron(sym_basis[int(b1)-1],conj_basis[int(b2)-1])

        p = np.zeros((len(b3_basis[0]), len(b3_basis[0])))
        for k in range(len(b3_basis)):
            p += np.outer(b3_basis[k], b3_basis[k])

        p = left_transition @ p @ right_transition
        with open(PROJECTORS_FILE(d, n, irrep3), "w+") as file:
            np.savetxt(file, p)

def is_projector(garray):
    return np.linalg.norm(garray @ garray - garray) < 10e-10

def dim_projector(projector):
    if not is_projector(projector):
        print('This is not a projector.')
    _, singvalues, _ = np.linalg.svd(projector)
    return np.sum(np.isclose(singvalues, 1.))


class Projector:
    def __init__(self, d, n) -> None:
        self.d = d
        self.n = n
        self.build()
    
    def build(self):
        self.projector = np.array([])
    
    @property
    def is_projector(self):
        return is_projector(self.projector)

    @property
    def dim_projector(self):
        return dim_projector(self.projector)

    def overlap(self, gmatrix:np.ndarray):
        return (gmatrix.reshape(1, -1) @ self.projector @ gmatrix.reshape(-1, 1))[0,0]


class NonintProjector(Projector):
    def __init__(self, d, n, irrep, build_pmatrix = True) -> None:
        self.irrep = irrep
        self.build_pmatrix = build_pmatrix
        super().__init__(d, n)
    
    def build(self):
        if self.build_pmatrix:
            if not os.path.isfile(PROJECTORS_FILE(self.d, self.n, self.irrep)):
                # TODO this has to be for each single projector not all of them.
                print(f'building and saving projectors in {PROJECTORS_DIR(self.d, self.n)}')
                save_clebsch_gordan_coefficients(self.d, self.n)
                build_projectors_noninteracting(self.d, self.n)
            
            with open(PROJECTORS_FILE(self.d, self.n, self.irrep), "r") as file:
                p = np.loadtxt(file)

            self.projector = p
        else:
            self.projector = None 
    
    @property
    def young_tableau(self):
        pass
    
    @property
    def gelfland_tseling_diagram(self):
        pass

    @property
    def name(self):
        return '-'.join(map(str, self.irrep))

    @property
    def dim(self):
        return dim_irrep(self.irrep)


class IntProjector(Projector):
    def __init__(self, d, n, build_pmatrix = True) -> None:
        self.build_pmatrix = build_pmatrix
        super().__init__(d, n)
    
    def build(self):
        if self.build_pmatrix:
            d_fock = int(binom(self.d + self.n - 1, self.n))
            # basis vectors of H.
            b = basis.standard(d_fock)
            # |0>\otimes|0>, |1>\otimes|1>, ... |d_fock>\otimes|d_fock>
            b_ii  = np.einsum('nk,nl->nkl', b, b).reshape(d_fock, -1)
            # Create the maximally mixed density matrix, 1 minus this gives the nontrivial projector.
            p_trivial = np.sum(np.einsum('ij, kl -> ikjl', b_ii, b_ii), axis=(0,1)) / d_fock
            self.projector = np.eye(d_fock ** 2) - p_trivial
        else:
            self.projector = None
    
    @property
    def irrep(self):
        return 'int'

    @property
    def name(self):
        return self.irrep
    
    @property
    def dim(self):
        return int(binom(self.d + self.n - 1, self.n)) - 1

class AllProjectors(list):
    def __init__(self, d, n, interacting, build_pmatrix = True) -> None:
        super().__init__()
        self.d = d
        self.n = n
        self.interacting = interacting
        self.build_pmatrix = build_pmatrix
        if not interacting:
            self.irreps = IRREPS(d, n)
        self.build()
    
    def build(self):
        
        if not self.interacting and self.n > 1:
            for irrep in self.irreps:
                self.append(NonintProjector(self.d, self.n, irrep, self.build_pmatrix))
        else:
            self.append(IntProjector(self.d, self.n, self.build_pmatrix))
            
            
        
    
    
    