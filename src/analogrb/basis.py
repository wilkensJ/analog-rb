import numpy as np
from itertools import product
from scipy.special import binom
from itertools import permutations

#######################################################
################### HELPER FUNCTIONS ##################
#######################################################


def gellmann_matrix(j, k, d):
    """Returns a generalized Gell-Mann matrix of dimension d.
    According to the convention in *Bloch Vectors for Qubits* by
    Bertlmann and Krammer (2008),
    https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices#Construction
    Taken from Jonathan Gross. Revision 449580a1
    Parameters
    ----------
    j : int
        First index for generalized Gell-Mann matrix
    k : int
        Second index for generalized Gell-Mann matrix
    d : int
        Dimension of the generalized Gell-Mann matrix

    Returns:
        A genereralized Gell-Mann matrix : np.ndarray
    """
    # Check the indices 'j' and 'k.
    if j > k:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = -1.0j
        gjkd[k - 1][j - 1] = 1.0j
    elif j == k and j < d:
        gjkd = np.sqrt(2 / (j * (j + 1))) * np.diag(
            [
                1 + 0.0j if n <= j else (-j + 0.0j if n == (j + 1) else 0 + 0.0j)
                for n in range(1, d + 1)
            ]
        )
    else:
        # Identity matrix
        gjkd = np.diag([1 + 0.0j for n in range(1, d + 1)])
        # normalize such that trace(gjkd*gjkd) = 2
        gjkd = gjkd * np.sqrt(2 / d)

    return gjkd


def adjoint_liouville_representation(gunitary):
    """This function take the adjoint action of a unitary and
    rewrites is into one matrix (so called liouville representation)
    Parameters
    ----------
    gunitary : np.ndarray
        a given unitary.
    """
    # Get the dimension of the matrix.
    dim = len(gunitary)
    # Get the basis matrices via the gellmann_matrix function.
    # This is a filled list.
    basis = np.array(gellmann(dim))
    # Initialize the final matrix representation which has the
    # dimension d**2 x d**2.
    final = np.zeros((dim**2, dim**2), dtype=np.complex64)
    # Go through the basis list and calculate the sum described above.
    for element in basis:
        left = gunitary.dot(element.dot(gunitary.T.conj())).reshape(-1, 1)
        right = element.reshape(1, -1)
        final = final + np.dot(left, right)
    # Normalize, and done!
    return final / 2.0


def rec(start, nodes, rep=1, depth=0):
    """A recursion function."""
    if depth == rep:
        return np.arange(start, nodes)
    else:
        return np.concatenate(
            [rec(n, nodes, rep=rep, depth=depth + 1) for n in np.arange(start, nodes)]
        )


def fock_in_particle_space(nodes, particles):
    """ """
    final = []
    for count in range(particles):
        binom_coeffs = [
            int(binom(particles + nodes - 2 - count - kk, particles - 1 - count))
            for kk in range(nodes)
        ]
        base = np.array(
            [np.zeros(binom_coeffs[n], dtype=int) + n for n in range(nodes)], dtype=list
        )
        indices = rec(0, nodes, rep=count)
        final.append(np.concatenate(base[indices]))
    final = np.array(final).reshape(particles, -1).T
    return final


#######################################################
######################## BASIS ########################
#######################################################


def dicke(nodes, particles):
    particlesbase = np.diag(np.ones(nodes, dtype=int))
    mybase = fock_in_particle_space(nodes, particles)
    list_basevectors = []
    for k in range(len(mybase)):
        perms = set(permutations(mybase[k]))
        sap = np.zeros(nodes**particles, dtype=int)
        for perm in perms:
            useforkron = particlesbase[list(perm)]
            blub = useforkron[0]
            for count in range(1, len(perm)):
                blub = np.kron(blub, useforkron[count])
            sap += blub
        list_basevectors.append(sap / np.sqrt(len(perms)))
    return list_basevectors


def standard(dim):
    """ """
    base_vectors = np.diag(np.ones(dim))
    return base_vectors


def gellmann(dim):
    """Return a basis of orthogonal Hermitian operators
    in a Hilbert space of dimension d, with the identity element
    in the last place.
    Taken from Jonathan Gross. Revision 449580a1
    Parameters
    ----------
    dim : int
        The amount of matrix basis elements and which
        dimension the matrices have.
    """
    return [gellmann_matrix(j, k, dim) for j, k in product(range(1, dim + 1), repeat=2)]


def fock(nodes, particles):
    """Generates the Fock Basis for 'number_particles' many
    particles in 'd' many nodes in the lattice or chain.
    For example a 3 node chain with 2 particles yields:
    [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
    Returns
    -------
    states:np.ndarray (2 dimensional vector)
        array filled with fock basis vectors.
    """
    # Initialize the amount of arrays needed and fill with zeros.
    states = np.zeros((int(binom(particles + nodes - 1, particles)), nodes), dtype=int)
    # First entry in first array is the amount of particles.
    states[0, 0] = particles
    # Initialize for loop variable.
    count = 0
    # Loop over all arrays in states.
    for i in range(1, states.shape[0]):
        states[i, : nodes - 1] = states[i - 1, : nodes - 1]
        states[i, count] -= 1
        states[i, count + 1] += 1 + states[i - 1, nodes - 1]
        if count >= nodes - 2:
            if np.any(states[i, : nodes - 1]):
                count = np.nonzero(states[i, : nodes - 1])[0][-1]
        else:
            count += 1
    return states
