.DEFAULT_GOAL := help

MPI ?= 1
DEBUG ?= no
OPT ?= yes
KERNPROF ?= no

SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RUN_DIR = $(SCRIPT_DIR)tools/run.py
EXAMPLE_DIR := $(SCRIPT_DIR)tools/examples

TOTAL_MEM_KB := $(shell grep MemTotal /proc/meminfo | awk '{print $$2}')
LIMIT_KB := $(shell echo "scale=0; 0.8 * $(TOTAL_MEM_KB) / $(MPI)" | bc)

ifeq ($(DEBUG),yes)
    DEBUG_FLAG = -malloc_debug -ksp_converged_reason -log_view
else
    DEBUG_FLAG =
endif

ifeq ($(OPT),yes)
    OPT_FLAG = -ksp_reuse_preconditioner true
else
    OPT_FLAG =
endif

PYTHON_CMD = python3 $(RUN_DIR)

ifeq ($(KERNPROF),yes)
	PYTHON_EXEC = kernprof -l -v $(PYTHON_CMD)
else
	PYTHON_EXEC = $(PYTHON_CMD)
endif

MPIRUN_CMD = mpirun --bind-to core -n $(MPI)

EXAMPLE1_FLAGS = -name TPFA_Example1 -reservoir $(EXAMPLE_DIR)/example_1/reservoir.ini
EXAMPLE2_FLAGS = -name TPFA_Example2 -reservoir $(EXAMPLE_DIR)/example_2/reservoir.ini
EXAMPLE3_FLAGS = -name TPFA_Example3 -reservoir $(EXAMPLE_DIR)/example_3/reservoir.ini
EXAMPLE4_FLAGS = -name TPFA_Example4 -reservoir $(EXAMPLE_DIR)/example_4/reservoir.ini

EXAMPLES = example1 example2 example3 example4

.PHONY: all $(EXAMPLES) help clean

all: $(EXAMPLES)

example1:
	@echo "Running $@ with $(MPI) processes. Memory limit per process: $(LIMIT_KB) KB"
	$(MPIRUN_CMD) bash -c 'ulimit -v $(LIMIT_KB); $(PYTHON_EXEC) $(EXAMPLE1_FLAGS) $(DEBUG_FLAG) $(OPT_FLAG)'

example2:
	@echo "Running $@ with $(MPI) processes. Memory limit per process: $(LIMIT_KB) KB"
	$(MPIRUN_CMD) bash -c 'ulimit -v $(LIMIT_KB); $(PYTHON_EXEC) $(EXAMPLE2_FLAGS) $(DEBUG_FLAG) $(OPT_FLAG)'

example3:
	@echo "Running $@ with $(MPI) processes. Memory limit per process: $(LIMIT_KB) KB"
	$(MPIRUN_CMD) bash -c 'ulimit -v $(LIMIT_KB); $(PYTHON_EXEC) $(EXAMPLE3_FLAGS) $(DEBUG_FLAG) $(OPT_FLAG)'

example4:
	@echo "Running $@ with $(MPI) processes. Memory limit per process: $(LIMIT_KB) KB"
	$(MPIRUN_CMD) bash -c 'ulimit -v $(LIMIT_KB); $(PYTHON_EXEC) $(EXAMPLE4_FLAGS) $(DEBUG_FLAG) $(OPT_FLAG)'

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