#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import scipy.sparse.linalg
import sparseqr

if __name__ == "__main__":
    # QR decompose a sparse matrix M such that  Q R = M E
    #
    M = scipy.sparse.rand(10, 10, density=0.1)

    # Solve many linear systems "M x = b for b in columns(B)"
    #
    B = scipy.sparse.rand(10, 1, density=0.1)  # many RHS, sparse (could also have just one RHS with shape (10,))
    x = sparseqr.solve(M, B, tolerance=0)
    print(x)
