from __future__ import print_function, division, absolute_import

import atexit

import numpy as np
from scipy.sparse import coo_matrix, spmatrix

from .cffi_asarray import as_array
from ._sparseqr import ffi, lib

common = None


# Data format conversion
def scipy_to_cholmod_sparse(scipy_a):
    """
    Convert a SciPy sparse matrix to a CHOLMOD sparse matrix.
    The input is first internally converted to scipy.sparse.coo_matrix format.
    When no longer needed, the returned CHOLMOD sparse matrix must be deallocated using cholmod_free_sparse().
    """
    scipy_a = scipy_a.tocoo()

    num_values = scipy_a.nnz

    # There is a potential performance win if we know A is symmetric - can use Cholesky factorization
    chol_a = lib.cholmod_l_allocate_triplet(scipy_a.shape[0], scipy_a.shape[1], num_values, 0, lib.CHOLMOD_REAL,
                                            common)

    a_rows = ffi.cast("SuiteSparse_long*", chol_a.i)
    a_cols = ffi.cast("SuiteSparse_long*", chol_a.j)
    a_vals = ffi.cast("double*", chol_a.x)

    a_rows[0:num_values] = scipy_a.row
    a_cols[0:num_values] = scipy_a.col
    a_vals[0:num_values] = scipy_a.data

    chol_a.nnz = num_values

    assert lib.cholmod_l_check_triplet(chol_a, common) == 1

    # Convert to a cholmod_sparse matrix.
    result = lib.cholmod_l_triplet_to_sparse(chol_a, num_values, common)
    # Free the space used by the cholmod triplet matrix.
    _cholmod_free_triplet(chol_a)

    return result


def cholmod_sparse_to_scipy(chol_a):
    """Convert a CHOLMOD sparse matrix to a scipy.sparse.coo_matrix."""
    # Convert to a cholmod_triplet matrix.
    chol_a = lib.cholmod_l_sparse_to_triplet(chol_a, common)

    nnz = chol_a.nnz

    a_rows = ffi.cast("SuiteSparse_long*", chol_a.i)
    a_cols = ffi.cast("SuiteSparse_long*", chol_a.j)
    a_vals = ffi.cast("double*", chol_a.x)

    # Create copy of the array data, because the coo_matrix constructor doesn't and the cholmod memory fill get freed
    i = as_array(ffi, a_rows, nnz).copy()
    j = as_array(ffi, a_cols, nnz).copy()
    data = as_array(ffi, a_vals, nnz).copy()

    scipy_a = coo_matrix((data, (i, j)), shape=(chol_a.nrow, chol_a.ncol))

    # Free the space used by the cholmod triplet matrix.
    _cholmod_free_triplet(chol_a)

    return scipy_a


def numpy_to_cholmod_dense(numpy_a):
    """
    Convert a NumPy array (rank-1 or rank-2) to a CHOLMOD dense matrix.
    Rank-1 arrays are converted to column vectors.
    When no longer needed, the returned CHOLMOD dense matrix must be deallocated using cholmod_free_dense().
    """
    numpy_a = np.atleast_2d(numpy_a)
    if numpy_a.shape[0] == 1 and numpy_a.shape[1] > 1:  # prefer column vector
        numpy_a = numpy_a.T
    n_rows = numpy_a.shape[0]
    n_cols = numpy_a.shape[1]

    lda = n_rows  # cholmod_dense is column-oriented
    chol_a = lib.cholmod_l_allocate_dense(n_rows, n_cols, lda, lib.CHOLMOD_REAL, common)
    if chol_a == ffi.NULL:
        raise RuntimeError("Failed to allocate chol_A")
    data = ffi.cast("double*", chol_a.x)

    for j in range(n_cols):
        data[(j * lda):((j + 1) * lda)] = numpy_a[:, j]
    return chol_a


def cholmod_dense_to_numpy(chol_a):
    """
    Convert a CHOLMOD dense matrix to a NumPy array
    """
    a_data = ffi.cast("double*", chol_a.x)

    result = as_array(ffi, a_data, chol_a.nrow * chol_a.ncol).copy()
    result = result.reshape((chol_a.nrow, chol_a.ncol), order='F')
    return result


def permutation_vector_to_matrix(perm_vector):
    """
    Convert a permutation vector E (list or rank-1 array, length n) to a permutation matrix (n by n).
    The result is returned as a scipy.sparse.coo_matrix, where the entries at (E[k], k) are 1.
    """
    n = len(perm_vector)
    j = np.arange(n)
    return coo_matrix((np.ones(n), (perm_vector, j)), shape=(n, n))


def _cholmod_free_triplet(a):
    """
    Deallocate a CHOLMOD triplet format matrix
    """
    a_ptr = ffi.new("cholmod_triplet**")
    a_ptr[0] = a
    lib.cholmod_l_free_triplet(a_ptr, common)


def cholmod_free_sparse(a):
    """
    Deallocate a CHOLMOD sparse matrix
    """
    a_ptr = ffi.new("cholmod_sparse**")
    a_ptr[0] = a
    lib.cholmod_l_free_sparse(a_ptr, common)


def cholmod_free_dense(a):
    """
    Deallocate a CHOLMOD dense matrix
    """
    a_ptr = ffi.new("cholmod_dense**")
    a_ptr[0] = a
    lib.cholmod_l_free_dense(a_ptr, common)


def qr(a, tolerance=None, economy=None):
    """
    Given a sparse matrix A,
    returns Q, R, E, rank such that:
        Q*R = A*permutation_vector_to_matrix(E)
    rank is the estimated rank of A.

    If optional `tolerance` parameter is negative, it has the following meanings:
        #define SPQR_DEFAULT_TOL ...       /* if tol <= -2, the default tol is used */
        #define SPQR_NO_TOL ...            /* if -2 < tol < 0, then no tol is used */

    Be aware that this approach is slow and takes a lot of memory, because qr() explicitly constructs Q.
    Unless you have a large number of systems to solve with the same A, solve() is much faster.
    """

    chol_a = scipy_to_cholmod_sparse(a)

    chol_q = ffi.new("cholmod_sparse**")
    chol_r = ffi.new("cholmod_sparse**")
    chol_e = ffi.new("SuiteSparse_long**")

    if tolerance is None:
        tolerance = lib.SPQR_DEFAULT_TOL

    if economy is None:
        economy = False

    if isinstance(economy, bool):
        econ = a.shape[1] if economy else a.shape[0]
    else:
        # Treat as a number
        econ = int(economy)

    rank = lib.SuiteSparseQR_C_QR(
        # Input
        lib.SPQR_ORDERING_DEFAULT,
        tolerance,
        econ,
        chol_a,
        # Output
        chol_q,
        chol_r,
        chol_e,
        common
    )

    scipy_q = cholmod_sparse_to_scipy(chol_q[0])
    scipy_r = cholmod_sparse_to_scipy(chol_r[0])

    # If chol_E is null, there was no permutation.
    if chol_e == ffi.NULL:
        e = None
    else:
        e = as_array(ffi, chol_e[0], a.shape[1]).copy()

    cholmod_free_sparse(chol_q[0])
    cholmod_free_sparse(chol_r[0])
    cholmod_free_sparse(chol_a)

    return scipy_q, scipy_r, e, rank


def solve(a, b, tolerance=None):
    """
    Given a sparse m-by-n matrix A, and dense or sparse m-by-k matrix (storing k RHS vectors) b,
    solve A x = b in the least-squares sense.

    This is much faster than using qr() to solve the system, since Q is not explicitly constructed.

    Returns x on success, None on failure.

    The format of the returned x (on success) is either dense or sparse, corresponding to
    the format of the b that was supplied.

    The performance-optimal format for A is scipy.sparse.coo_matrix.

    If optional `tolerance` parameter is negative, it has the following meanings:
        #define SPQR_DEFAULT_TOL ...       /* if tol <= -2, the default tol is used */
        #define SPQR_NO_TOL ...            /* if -2 < tol < 0, then no tol is used */
    """
    if isinstance(b, spmatrix):
        return _solve_with_sparse_rhs(a, b, tolerance)
    else:
        return _solve_with_dense_rhs(a, b, tolerance)


def _solve_with_dense_rhs(a, b, tolerance=None):
    """
    Given a sparse m-by-n matrix A, and dense m-by-k matrix (storing k RHS vectors) b,
    solve A x = b in the least-squares sense.

    This is much faster than using qr() to solve the system, since Q is not explicitly constructed.

    Returns x (dense) on success, None on failure.

    The performance-optimal format for A is scipy.sparse.coo_matrix.

    If optional `tolerance` parameter is negative, it has the following meanings:
        #define SPQR_DEFAULT_TOL ...       /* if tol <= -2, the default tol is used */
        #define SPQR_NO_TOL ...            /* if -2 < tol < 0, then no tol is used */
    """

    chol_a = scipy_to_cholmod_sparse(a)
    chol_b = numpy_to_cholmod_dense(b)

    if tolerance is None:
        tolerance = lib.SPQR_DEFAULT_TOL

    chol_x = lib.SuiteSparseQR_C_backslash(
        # Input
        lib.SPQR_ORDERING_DEFAULT,
        tolerance,
        chol_a,
        chol_b,
        common)

    if chol_x == ffi.NULL:
        return None  # failed

    # Return x with the same shape as b.
    x_shape = list(b.shape)
    x_shape[0] = a.shape[1]
    numpy_x = cholmod_dense_to_numpy(chol_x).reshape(x_shape)

    # Free cholmod stuff
    cholmod_free_sparse(chol_a)
    cholmod_free_dense(chol_b)
    cholmod_free_dense(chol_x)
    return numpy_x


def _solve_with_sparse_rhs(a, b, tolerance=None):
    """
    Given a sparse m-by-n matrix A, and sparse m-by-k matrix (storing k RHS vectors) b,
    solve A x = b in the least-squares sense.

    This is much faster than using qr() to solve the system, since Q is not explicitly constructed.

    Returns x (sparse) on success, None on failure.

    The performance-optimal format for A and b is scipy.sparse.coo_matrix.

    If optional `tolerance` parameter is negative, it has the following meanings:
        #define SPQR_DEFAULT_TOL ...       /* if tol <= -2, the default tol is used */
        #define SPQR_NO_TOL ...            /* if -2 < tol < 0, then no tol is used */
    """

    chol_a = scipy_to_cholmod_sparse(a)
    chol_b = scipy_to_cholmod_sparse(b)

    if tolerance is None: tolerance = lib.SPQR_DEFAULT_TOL

    chol_x = lib.SuiteSparseQR_C_backslash_sparse(
        lib.SPQR_ORDERING_DEFAULT,
        tolerance,
        chol_a,
        chol_b,
        common)

    if chol_x == ffi.NULL:
        return None

    scipy_x = cholmod_sparse_to_scipy(chol_x)

    cholmod_free_sparse(chol_a)
    cholmod_free_sparse(chol_b)
    cholmod_free_sparse(chol_x)

    return scipy_x


# Set up cholmod de-initialize to run when Python exits
def _de_initialize():
    """De-initialize the CHOLMOD library."""
    lib.cholmod_l_finish(common)


def initialize(use_gpu):
    global common
    # Initialize cholmod
    common = ffi.new("cholmod_common*")
    lib.cholmod_l_start(common)

    # Tell CHOLMOD to use the GPU
    if use_gpu:
        total_mem = ffi.new("size_t*")
        available_mem = ffi.new("size_t*")
        common.useGPU = True
        cholmod_l_gpu_memorysize(total_mem, available_mem, common)
        common.gpuMemorySize = available_mem
        if common.gpuMemorySize <= 1:
            print("No GPU available")

    atexit.register(lambda: _de_initialize)
