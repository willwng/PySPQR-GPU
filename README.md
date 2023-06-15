# Python wrapper for SuiteSparseQR

This module wraps the [SuiteSparseQR](http://faculty.cse.tamu.edu/davis/suitesparse.html)
decomposition function for use with SciPy sparse matrices.

Also wrapped are the SuiteSparseQR solvers for ``Ax = b`` for the cases with sparse `A` and dense or sparse `b`.
This is especially useful for solving sparse overdetermined linear systems in the least-squares sense.
Here `A` is of size m-by-n and `b` is m-by-k (storing `k` different right-hand side vectors, each considered separately).

# Usage

```python
import numpy
import scipy.sparse.linalg
import sparseqr

A = scipy.sparse.rand(n, n, density=0.1)
x_truth = np.random.rand(n, 1)

# Solve many linear systems "M x = b for b in columns(B)"
b = A * x_truth
x_solve = sparseqr.solve(A, b, tolerance=0)
```

