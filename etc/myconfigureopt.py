#!/usr/bin/env python3

import os

petsc_hash_pkgs = os.path.join(os.getenv("HOME"), "petsc-hash-pkgs")

configure_options_perf = [
    "--package-prefix-hash=" + petsc_hash_pkgs,
    "--with-clanguage=c",
    "--with-shared-libraries=yes",
    "--with-debugging=no",
    "--with-precision=double",
    "--with-make-test-np=2",
    "--with-openmp",
    "--with-openmp-kernels",
    "CFLAGS+=-O3 -pedantic -Wno-long-long -Wno-overlength-strings",
    "FFLAGS+=-O3 -ffree-line-length-512",
    "CXXFLAGS+=-O3 -pedantic -Wno-long-long -Wno-overlength-strings",
    "--with-ssl=1",
    "--with-tau-perfstubs=0",
    "--with-strict-petscerrorcode",
    "--with-cuda-dir=/usr/local/cuda-12.9",
    "--with-hdf5",
    "--with-petsc4py",
    "--download-hdf5",
    "--download-hypre",
    "--download-kokkos",
    "--download-kokkos-kernels",
    "--download-superlu_dist",
    "--download-umpire",
]


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.abspath("config"))
    import configure

    configure.petsc_configure(configure_options_perf)
