import time

import numpy as np
import scipy
import sparseqr

if __name__ == "__main__":
    start_time = time.time()
    n = 500
    # QR decompose a sparse matrix M such that  Q R = M E
    A = scipy.sparse.rand(n, n, density=0.1)
    x_truth = np.random.rand(n, 1)

    # Solve many linear systems "M x = b for b in columns(B)"
    b = A * x_truth
    x_solve = sparseqr.solve(A, b, tolerance=0)

    print("Relative error in solution: ", np.linalg.norm(x_truth - x_solve) / np.linalg.norm(x_solve))
    print("Relative error in residual: ", np.linalg.norm(A.dot(x_solve) - b) / np.linalg.norm(b))
    print(f"Time used: {time.time() - start_time:.2f} seconds")
