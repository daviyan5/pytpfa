# Makefile for running TPFA examples with optional MPI and debug flags.
.DEFAULT_GOAL := help

MPI ?= 1
DEBUG ?= no
OPT ?= yes
KERNPROF ?= no

SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RUN_DIR = $(SCRIPT_DIR)tools/run.py
EXAMPLE_DIR := $(SCRIPT_DIR)tools/examples

ifeq ($(DEBUG),yes)
    DEBUG_FLAG = -malloc_debug -ksp_converged_reason -log_view
else
    DEBUG_FLAG =
endif

ifeq ($(OPT),yes)
    OPT_FLAG = -opt -ksp_reuse_preconditioner true
else
    OPT_FLAG =
endif

PYTHON_CMD = python3 $(RUN_DIR)

ifeq ($(KERNPROF),yes)
    KERN_PROF = kernprof -l -v
    PYTHON_CMD = $(RUN_DIR)
else
    KERN_PROF =
endif

MPIRUN = mpirun --bind-to core -n $(MPI) 
BASE_CMD = $(MPIRUN) $(KERN_PROF) $(PYTHON_CMD)

EXAMPLES = example1 example2 example3 example4

.PHONY: all $(EXAMPLES) help clean

all: $(EXAMPLES)

example1:
	$(BASE_CMD) -name TPFA_Example1 -reservoir $(EXAMPLE_DIR)/example_1/reservoir.ini $(DEBUG_FLAG) $(OPT_FLAG)

example2:
	$(BASE_CMD) -name TPFA_Example2 -reservoir $(EXAMPLE_DIR)/example_2/reservoir.ini $(DEBUG_FLAG) $(OPT_FLAG)

example3:
	$(BASE_CMD) -name TPFA_Example3 -reservoir $(EXAMPLE_DIR)/example_3/reservoir.ini $(DEBUG_FLAG) $(OPT_FLAG)

example4:
	$(BASE_CMD) -name TPFA_Example4 -reservoir $(EXAMPLE_DIR)/example_4/reservoir.ini $(DEBUG_FLAG) $(OPT_FLAG)

clean:
	@echo "Cleaning generated files from $(EXAMPLE_DIR)..."
	@find $(EXAMPLE_DIR) -name "*.log" -type f -delete
	@find $(EXAMPLE_DIR) -name "*.lprof" -type f -delete
	@find $(EXAMPLE_DIR) -name "*.vts" -type f -delete
	@find $(EXAMPLE_DIR) -name "*.vtk" -type f -delete
	@find $(EXAMPLE_DIR) -name "*.npy" -type f -delete
	@echo "Clean complete."

help:
	@echo "Makefile for running TPFA examples"
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all          Run all examples."
	@echo "  example1     Run the first example."
	@echo "  example2     Run the second example."
	@echo "  example3     Run the third example."
	@echo "  example4     Run the fourth example."
	@echo "  clean        Remove all generated log, vtk, and profile files from example folders."
	@echo "  help         Show this help message."
	@echo ""
	@echo "Options (can be set from the command line):"
	@echo "  MPI=<n>      Set the number of MPI processes (default: 1)."
	@echo "  DEBUG=<yes|no> Enable PETSc debug flags (default: no)."
	@echo "  OPT=<yes|no>   Enable optimization flags (default: yes)."
	@echo "  KERNPROF=<yes|no> Enable kernprof for profiling (default: no)."
	@echo ""
	@echo "Example:"
	@echo "  make example1 MPI=4 DEBUG=yes"
	@echo ""