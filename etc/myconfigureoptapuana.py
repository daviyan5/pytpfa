#!/usr/bin/env python3
import os

petsc_hash_pkgs = os.path.join(os.getenv("HOME"), "petsc-hash-pkgs")

cuda_dir = "/usr/local/cuda"
if os.path.exists("/usr/local/cuda-12.1"):
    cuda_dir = "/usr/local/cuda-12.1"
elif os.path.exists("/usr/local/cuda-11.8"):
    cuda_dir = "/usr/local/cuda-11.8"

configure_options_perf = [
    "--package-prefix-hash=" + petsc_hash_pkgs,
    "--with-clanguage=c",
    "--with-shared-libraries=yes",
    "--with-debugging=no",
    "--with-precision=double",
    "--with-make-test-np=2",
    "--with-openmp",
    "--with-openmp-kernels",
    "--with-64-bit-indices=1",
    "CFLAGS=-O3 -march=native -mtune=native",
    "FFLAGS=-O3 -march=native -mtune=native -ffree-line-length-512",
    "CXXFLAGS=-O3 -march=native -mtune=native",
    "--with-ssl=1",
    "--with-strict-petscerrorcode",
    "--with-cuda=1",
    "--with-cuda-dir=" + cuda_dir,
    "--with-cuda-arch=80",
    "--with-cudac=nvcc",
    "--download-hdf5",
    "--download-hypre",
    "--download-kokkos",
    "--download-kokkos-kernels",
    "--download-superlu_dist",
    "--download-umpire",
    "--with-petsc4py",
    "--with-mpi=1",
    "--with-blaslapack-lib=-lblas -llapack",
]

if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.abspath("config"))
    import configure

    configure.petsc_configure(configure_options_perf)
