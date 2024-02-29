from numpy.random import default_rng
from scipy import sparse
from scipy.sparse import linalg
from sklearn.linear_model import Ridge


def _rescale_matrix(A, spectral_radius):
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    eigenvalues, _ = linalg.eigs(A)
    max_eigenvalue = max(abs(eigenvalues))
    A_scaled = A / max_eigenvalue * spectral_radius
    return(A_scaled)


def make_reservoir(size, degree, radius):
    rng = default_rng()
    nonzero_mat = rng.random((size, size)) < degree / size
    random_weights_mat = -1 + 2 * rng.random((size, size)) # uniform distribution on [-1, 1] at each matrix element
    A = nonzero_mat * random_weights_mat
    A[A == -0.0] = 0.0
    sA = sparse.csr_matrix(A)

    try:
        sA_rescaled = _rescale_matrix(sA, radius)
        return sA_rescaled
    except linalg.eigen.arpack.ArpackNoConvergence:
        return make_reservoir(size, degree, radius)
        

def make_input_matrix(num_inputs, res_size, sigma):
    rng = default_rng()
    W_in = sigma * (-1 + 2 * rng.random((res_size, num_inputs)))
    return W_in