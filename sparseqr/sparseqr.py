from __future__ import print_function, division, absolute_import
import _cffi_backend
from .cffi_asarray import as_array

import scipy.sparse
from scipy.sparse import coo_matrix
import numpy
import atexit

USE_GPU = True

# The compilation here works only if the files have been copied locally into the project,
# as the compilation requires write access into the directory the files reside in.
try:
    from ._sparseqr import ffi, lib
except ImportError:
    print("=== Wrapper module not compiled; compiling...")
    from .sparseqr_gen import main

    main()
    print("=== ...compiled.")

    from ._sparseqr import ffi, lib

# Initialize cholmod
common = ffi.new("cholmod_common*")
lib.cholmod_l_start(common)

# Tell CHOLMOD to use the GPU
if USE_GPU:
    total_mem = ffi.new("size_t*")
    available_mem = ffi.new("size_t*")
    common.useGPU = True
    # cholmod_l_gpu_memorysize(total_mem, available_mem, common)
    # common.gpuMemorySize = available_mem
    # if common.gpuMemorySize <= 1:
    #     print("No GPU available")


# Set up cholmod deinit to run when Python exits
def _deinit():
    """De-initialize the CHOLMOD library."""
    lib.cholmod_l_finish(common)


atexit.register(_deinit)


# Data format conversion

def scipy_to_cholmod_sparse(scipy_a):
    """
    Convert a SciPy sparse matrix to a CHOLMOD sparse matrix.
    The input is first internally converted to scipy.sparse.coo_matrix format.
    When no longer needed, the returned CHOLMOD sparse matrix must be deallocated using cholmod_free_sparse().
    """
    scipy_a = scipy_a.tocoo()

    num_values = scipy_a.nnz

    # There is a potential performance win if we know A is symmetric - Cholesky factorization
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

    # NOTE: Create a copy() of the array data, because the coo_matrix() constructor
    #       doesn't and the cholmod memory fill get freed.
    i = as_array(ffi, a_rows, nnz).copy()
    j = as_array(ffi, a_cols, nnz).copy()
    data = as_array(ffi, a_vals, nnz).copy()

    scipy_a = coo_matrix((data, (i, j)), shape=(chol_a.nrow, chol_a.ncol))

    # Free the space used by the cholmod triplet matrix.
    _cholmod_free_triplet(chol_a)

    return scipy_a


def numpy_to_cholmod_dense(numpy_A):
    """Convert a NumPy array (rank-1 or rank-2) to a CHOLMOD dense matrix.

Rank-1 arrays are converted to column vectors.

When no longer needed, the returned CHOLMOD dense matrix must be deallocated using cholmod_free_dense().
"""
    numpy_A = numpy.atleast_2d(numpy_A)
    if numpy_A.shape[0] == 1 and numpy_A.shape[1] > 1:  # prefer column vector
        numpy_A = numpy_A.T
    nrow = numpy_A.shape[0]
    ncol = numpy_A.shape[1]
    lda = nrow  # cholmod_dense is column-oriented
    chol_A = lib.cholmod_l_allocate_dense(nrow, ncol, lda, lib.CHOLMOD_REAL, common)
    if chol_A == ffi.NULL:
        raise RuntimeError("Failed to allocate chol_A")
    Adata = ffi.cast("double*", chol_A.x)
    for j in range(ncol):  # FIXME inefficient?
        Adata[(j * lda):((j + 1) * lda)] = numpy_A[:, j]
    return chol_A


def cholmod_dense_to_numpy(chol_A):
    '''Convert a CHOLMOD dense matrix to a NumPy array.'''
    Adata = ffi.cast("double*", chol_A.x)

    result = as_array(ffi, Adata, chol_A.nrow * chol_A.ncol).copy()
    result = result.reshape((chol_A.nrow, chol_A.ncol), order='F')
    return result


def permutation_vector_to_matrix(E):
    """Convert a permutation vector E (list or rank-1 array, length n) to a permutation matrix (n by n).
    The result is returned as a scipy.sparse.coo_matrix, where the entries at (E[k], k) are 1.
    """
    n = len(E)
    j = numpy.arange(n)
    return scipy.sparse.coo_matrix((numpy.ones(n), (E, j)), shape=(n, n))


# Memory management

# Used only internally by this module (the user sees only sparse and dense formats).
def _cholmod_free_triplet(A):
    """Deallocate a CHOLMOD triplet format matrix."""
    A_ptr = ffi.new("cholmod_triplet**")
    A_ptr[0] = A
    lib.cholmod_l_free_triplet(A_ptr, common)


def cholmod_free_sparse(A):
    """Deallocate a CHOLMOD sparse matrix."""
    A_ptr = ffi.new("cholmod_sparse**")
    A_ptr[0] = A
    lib.cholmod_l_free_sparse(A_ptr, common)


def cholmod_free_dense(A):
    '''Deallocate a CHOLMOD dense matrix.'''
    A_ptr = ffi.new("cholmod_dense**")
    A_ptr[0] = A
    lib.cholmod_l_free_dense(A_ptr, common)


# Solvers
def rz(A, B, tolerance=None):
    getCTX = int(0)
    chol_A = scipy_to_cholmod_sparse(A)
    chol_b = numpy_to_cholmod_dense(B)
    chol_Z = ffi.new("cholmod_dense**")
    chol_R = ffi.new("cholmod_sparse**")
    chol_E = ffi.new("SuiteSparse_long**")
    if tolerance is None:
        tolerance = 0.

    rank = lib.SuiteSparseQR_C(
        # Input
        lib.SPQR_ORDERING_DEFAULT,
        tolerance,
        A.shape[1],
        getCTX,
        chol_A,
        ffi.NULL,
        chol_b,
        # Output
        ffi.NULL,
        chol_Z,
        chol_R,
        chol_E,
        ffi.NULL,
        ffi.NULL,
        ffi.NULL,
        common
    )
    scipy_Z = cholmod_dense_to_numpy(chol_Z[0])
    scipy_R = cholmod_sparse_to_scipy(chol_R[0])

    # If chol_E is null, there was no permutation.
    if chol_E == ffi.NULL:
        E = None
    else:
        E = as_array(ffi, chol_E[0], A.shape[1]).copy()

    # Free cholmod stuff
    cholmod_free_dense(chol_Z[0])
    cholmod_free_sparse(chol_R[0])
    cholmod_free_sparse(chol_A)
    cholmod_free_dense(chol_b)

    return scipy_Z, scipy_R, E, rank


def qr(A, tolerance=None, economy=None):
    """
    Given a sparse matrix A,
    returns Q, R, E, rank such that:
        Q*R = A*permutation_vector_to_matrix(E)
    rank is the estimated rank of A.

    If optional `tolerance` parameter is negative, it has the following meanings:
        #define SPQR_DEFAULT_TOL ...       /* if tol <= -2, the default tol is used */
        #define SPQR_NO_TOL ...            /* if -2 < tol < 0, then no tol is used */

    For A, an m-by-n matrix, Q will be m-by-m and R will be m-by-n.

    If optional `economy` parameter is truthy, Q will be m-by-k and R will be
    k-by-n, where k = min(m, n).

    The performance-optimal format for A is scipy.sparse.coo_matrix.

    For solving systems of the form A x = b, see solve().

    qr() can also be used to solve systems, as follows:

        # inputs: scipy.sparse.coo_matrix A, rank-1 numpy.array b (RHS)
        import numpy
        import scipy

        Q, R, E, rank = sparseqr.qr( A )
        r = rank  # r could be min(A.shape) if A is full-rank

        # The system is only solvable if the lower part of Q.T @ B is all zero:
        print( "System is solvable if this is zero:", abs( (( Q.tocsc()[:,r:] ).T ).dot( B ) ).sum() )

        # Use CSC format for fast indexing of columns.
        R  = R.tocsc()[:r,:r]
        Q  = Q.tocsc()[:,:r]
        QB = (Q.T).dot(B).tocsc()  # for best performance, spsolve() wants the RHS to be in CSC format.
        result = scipy.sparse.linalg.spsolve(R, QB)

        # Recover a solution (as a dense array):
        x = numpy.zeros( ( A.shape[1], B.shape[1] ), dtype = result.dtype )
        x[:r] = result.todense()
        x[E] = x.copy()

        # Recover a solution (as a sparse matrix):
        x = scipy.sparse.vstack( ( result.tocoo(), scipy.sparse.coo_matrix( ( A.shape[1] - rank, B.shape[1] ), dtype = result.dtype ) ) )
        x.row = E[ x.row ]

    Be aware that this approach is slow and takes a lot of memory, because qr() explicitly constructs Q.
    Unless you have a large number of systems to solve with the same A, solve() is much faster.
    """

    chol_A = scipy_to_cholmod_sparse(A)

    chol_Q = ffi.new("cholmod_sparse**")
    chol_R = ffi.new("cholmod_sparse**")
    chol_E = ffi.new("SuiteSparse_long**")

    if tolerance is None: tolerance = lib.SPQR_DEFAULT_TOL

    if economy is None: economy = False

    if isinstance(economy, bool):
        econ = A.shape[1] if economy else A.shape[0]
    else:
        # Treat as a number
        econ = int(economy)

    rank = lib.SuiteSparseQR_C_QR(
        # Input
        lib.SPQR_ORDERING_DEFAULT,
        tolerance,
        econ,
        chol_A,
        # Output
        chol_Q,
        chol_R,
        chol_E,
        common
    )

    scipy_Q = cholmod_sparse_to_scipy(chol_Q[0])
    scipy_R = cholmod_sparse_to_scipy(chol_R[0])

    # If chol_E is null, there was no permutation.
    if chol_E == ffi.NULL:
        E = None
    else:
        # Have to pass through list().
        # https://bitbucket.org/cffi/cffi/issues/292/cant-copy-data-to-a-numpy-array
        # http://stackoverflow.com/questions/16276268/how-to-pass-a-numpy-array-into-a-cffi-function-and-how-to-get-one-back-out
        # E = numpy.zeros( A.shape[1], dtype = int )
        # E[0:A.shape[1]] = list( chol_E[0][0:A.shape[1]] )
        # UPDATE: I can do this without going through list() or making two extra copies.
        E = as_array(ffi, chol_E[0], A.shape[1]).copy()

    # Free cholmod stuff
    cholmod_free_sparse(chol_Q[0])
    cholmod_free_sparse(chol_R[0])
    cholmod_free_sparse(chol_A)
    # Apparently we don't need to do this. (I get a malloc error.)
    # lib.cholmod_l_free( A.shape[1], ffi.sizeof("SuiteSparse_long"), chol_E, cc )

    return scipy_Q, scipy_R, E, rank


def solve(A, b, tolerance=None):
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
    if isinstance(b, scipy.sparse.spmatrix):
        return _solve_with_sparse_rhs(A, b, tolerance)
    else:
        return _solve_with_dense_rhs(A, b, tolerance)


def _solve_with_dense_rhs(A, b, tolerance=None):
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

    chol_A = scipy_to_cholmod_sparse(A)
    chol_b = numpy_to_cholmod_dense(b)

    if tolerance is None: tolerance = lib.SPQR_DEFAULT_TOL

    chol_x = lib.SuiteSparseQR_C_backslash(
        # Input
        lib.SPQR_ORDERING_DEFAULT,
        tolerance,
        chol_A,
        chol_b,
        common)

    if chol_x == ffi.NULL:
        return None  # failed

    # Return x with the same shape as b.
    x_shape = list(b.shape)
    x_shape[0] = A.shape[1]
    numpy_x = cholmod_dense_to_numpy(chol_x).reshape(x_shape)

    # Free cholmod stuff
    cholmod_free_sparse(chol_A)
    cholmod_free_dense(chol_b)
    cholmod_free_dense(chol_x)
    return numpy_x


def _solve_with_sparse_rhs(A, b, tolerance=None):
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

    chol_A = scipy_to_cholmod_sparse(A)
    chol_b = scipy_to_cholmod_sparse(b)

    if tolerance is None: tolerance = lib.SPQR_DEFAULT_TOL

    chol_x = lib.SuiteSparseQR_C_backslash_sparse(
        # Input
        lib.SPQR_ORDERING_DEFAULT,
        tolerance,
        chol_A,
        chol_b,
        common)

    if chol_x == ffi.NULL:
        return None  # failed

    scipy_x = cholmod_sparse_to_scipy(chol_x)

    # Free cholmod stuff
    cholmod_free_sparse(chol_A)
    cholmod_free_sparse(chol_b)
    cholmod_free_sparse(chol_x)

    return scipy_x
