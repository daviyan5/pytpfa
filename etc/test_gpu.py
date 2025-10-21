#!/usr/bin/env python3

import sys
import numpy as np

try:
    import petsc4py
    from petsc4py import PETSc
except ImportError:
    print("Error: petsc4py not installed")
    sys.exit(1)

print("petsc4py version:", petsc4py.__version__)

petsc4py.init(sys.argv)

print("PETSc version:", PETSc.Sys.getVersion())
print("Scalar type:", PETSc.ScalarType)

try:
    vec = PETSc.Vec().create()
    vec.setType(PETSc.Vec.Type.CUDA)
    vec.setSizes(1000)
    vec.setUp()
    print("CUDA support: Available")
    vec.destroy()
except:
    print("CUDA support: Not available")

n = 100
A = PETSc.Mat().create()
A.setSizes([n, n])
A.setType("aij")
A.setUp()

for i in range(n):
    A[i, i] = 2.0
    if i > 0:
        A[i, i - 1] = -1.0
    if i < n - 1:
        A[i, i + 1] = -1.0

A.assemblyBegin()
A.assemblyEnd()

print("Test completed successfully")
