import petsc4py
import sys

petsc4py.init(sys.argv)

from petsc4py import PETSc
import os
import yaml
import time
import numpy as np

options = PETSc.Options()
device = options.getString("device", "cpu")
output_path = options.getString("output", ".")
N = options.getInt("number", 100000)

if device == "gpu":
    vec_type = PETSc.Vec.Type.CUDA
    mat_type = PETSc.Mat.Type.AIJCUSPARSE
else:
    vec_type = PETSc.Vec.Type.MPI
    mat_type = PETSc.Mat.Type.AIJ

setup_stage = PETSc.Log.Stage("Setup")
solve_stage = PETSc.Log.Stage("Solve")

setup_stage.push()
x = PETSc.Vec().create()
x.setType(vec_type)
x.setSizes(N)
x.setFromOptions()
x.set(0.0)

A = PETSc.Mat().create()
A.setType(mat_type)
A.setSizes([N, N])
A.setFromOptions()
A.setUp()

rstart, rend = A.getOwnershipRange()
local_N = rend - rstart

i_indices = np.arange(rstart, rend, dtype=PETSc.IntType)

i_main = i_indices
j_main = i_indices
v_main = np.full(local_N, 2.0, dtype=PETSc.ScalarType)

mask_lower = i_indices > 0
i_lower = i_indices[mask_lower]
j_lower = i_indices[mask_lower] - 1
v_lower = np.full(i_lower.shape[0], -1.0, dtype=PETSc.ScalarType)

mask_upper = i_indices < N - 1
i_upper = i_indices[mask_upper]
j_upper = i_indices[mask_upper] + 1
v_upper = np.full(i_upper.shape[0], -1.0, dtype=PETSc.ScalarType)

np_i = np.concatenate((i_main, i_lower, i_upper))
np_j = np.concatenate((j_main, j_lower, j_upper))
np_v = np.concatenate((v_main, v_lower, v_upper))

A.setPreallocationCOO(np_i, np_j)
A.setValuesCOO(np_v, PETSc.InsertMode.INSERT_VALUES)

dt = 0.01
M = A.copy()
M.scale(dt)
M.shift(1.0)

b = x.duplicate()
b.set(1.0)
rhs = x.duplicate()

ksp = PETSc.KSP().create()
ksp.setOperators(M)

ksp.setType(PETSc.KSP.Type.FGMRES)
pc = ksp.getPC()

if device == "gpu":
    pc.setType(PETSc.PC.Type.GAMG)
else:
    pc.setType(PETSc.PC.Type.GAMG)

ksp.setTolerances(rtol=1e-12, max_it=1000)
ksp.setFromOptions()

setup_stage.pop()


solve_stage.push()
start_time = time.time()

num_steps = 12
for i in range(num_steps):
    b.copy(rhs)
    rhs.scale(dt)
    rhs.axpy(1.0, x)

    ksp.solve(rhs, x)

end_time = time.time()
solve_stage.pop()


exec_time = end_time - start_time
residual = ksp.getResidualNorm()

if PETSc.COMM_WORLD.getRank() == 0:
    results = {"execution_time": exec_time, "residual_norm": residual}

    print(f"Device: {device}")
    print(f"N: {N}")
    print(f"Execution Time: {exec_time:.6f} seconds")
    print(f"Residual Norm: {residual:.6e}")

    print(f"x type: {x.getType()}")
    print(f"M type: {M.getType()}")

    os.makedirs(output_path, exist_ok=True)

    filename = f"results_{device}_{N}_{PETSc.COMM_WORLD.getSize()}.yaml"
    filepath = os.path.join(output_path, filename)

    with open(filepath, "w") as f:
        yaml.dump(results, f)
