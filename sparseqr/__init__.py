from __future__ import absolute_import

USE_GPU = False

# If not already, compile the wrapper
try:
    from ._sparseqr import ffi, lib
except ImportError:
    print("--- Compiling SparseQR wrapper ---")
    from .sparseqr_gen import main

    main(use_gpu=USE_GPU)

finally:
    from .sparseqr import qr, solve, permutation_vector_to_matrix, initialize
    initialize(use_gpu=USE_GPU)
