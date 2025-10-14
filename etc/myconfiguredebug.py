#!/usr/bin/env python3

import os

petsc_hash_pkgs = os.path.join(os.getenv("HOME"), "petsc-hash-pkgs")

configure_options_debug = [
    "--package-prefix-hash=" + petsc_hash_pkgs,
    "--with-clanguage=c",
    "--with-debugging=yes",
    "--with-precision=double",
    "--with-make-test-np=2",
    "COPTFLAGS=-g -Og -fPIC",
    "FOPTFLAGS=-g -Og",
    "CXXOPTFLAGS=-g -Og -fPIC",
    "--with-ssl=1",
    "--with-tau-perfstubs=0",
    "--with-strict-petscerrorcode",
    "--with-coverage",
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

    configure.petsc_configure(configure_options_debug)
