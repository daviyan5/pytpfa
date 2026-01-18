#!/usr/bin/env python3

import os

petsc_hash_pkgs = os.path.join(os.getenv("HOME"), "petsc-hash-pkgs")

configure_options_jarvis = [
    "--package-prefix-hash=" + petsc_hash_pkgs,
    "--with-clanguage=c",
    "--with-shared-libraries=yes",
    "--with-debugging=no",
    "--with-precision=double",
    "--with-make-test-np=4",
    "--with-openmp",
    "--with-openmp-kernels",
    "--with-64-bit-indices=1",
    "CFLAGS+=-O3 -pedantic -Wno-long-long -Wno-overlength-strings",
    "FFLAGS+=-O3 -ffree-line-length-512",
    "CXXFLAGS+=-O3 -pedantic -Wno-long-long -Wno-overlength-strings",
    "--with-ssl=0",
    "--with-tau-perfstubs=0",
    "--with-strict-petscerrorcode",
    "--with-cuda-dir=/usr/local/cuda-12.6",
    "--with-hdf5",
    "--with-petsc4py",
    "--download-cmake",
    "--download-mpich",
    "--download-hdf5",
    "--download-hypre",
    "--download-kokkos",
    "--download-kokkos-kernels",
    "--download-superlu_dist",
    "--download-f2cblaslapack=1",
    "--download-umpire",
]


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.abspath("config"))
    import configure

    configure.petsc_configure(configure_options_jarvis)
