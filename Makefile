.DEFAULT_GOAL := help

# --- Configurable Options ---
MPI ?= 1
DEBUG ?= no
OPT ?= yes      # Use optimized flags (skip checks)
POST ?= no      # Enable post-processing (VTK output)
PROFILE ?= no   # Enable cProfile/line_profiler
LIMIT ?= no     # Limit memory usage per process
GPU ?= yes      # Enable GPU support (requires CUDA and PETSc with GPU support)
TYPE ?= ksp

# --- Project Paths ---
SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RUN_DIR = $(SCRIPT_DIR)tools/run.py
COMPARE_DIR = $(SCRIPT_DIR)tools/scripts/compare.py
EXAMPLE_DIR := $(SCRIPT_DIR)tools/examples
TEST_GPU_SCRIPT = $(SCRIPT_DIR)etc/test_gpu.py
RESULTS_DIR = $(SCRIPT_DIR)etc/results

# --- Memory Limiter (conditional) ---
ifeq ($(LIMIT),yes)
    TOTAL_MEM_KB := $(shell grep MemTotal /proc/meminfo | awk '{print $$2}')
    LIMIT_KB := $(shell echo "scale=0; 0.95 * $(TOTAL_MEM_KB) / $(MPI)" | bc)
    RUN_PREFIX = ulimit -v $(LIMIT_KB);
    LIMIT_MSG = Memory limit per process: $(LIMIT_KB) KB.
else
    RUN_PREFIX =
    LIMIT_MSG = Memory limit disabled.
endif

# --- Flag Definitions ---
ifeq ($(DEBUG),yes)
    DEBUG_FLAG = -malloc_debug -ksp_converged_reason
else
    DEBUG_FLAG =
endif

ifeq ($(OPT),yes)
    OPT_FLAG = -opt
else
    OPT_FLAG =
endif

ifeq ($(GPU),yes)
	GPU_FLAG = -gpu -mat_type aijcusparse
else
	GPU_FLAG =
endif

ifeq ($(POST),yes)
    POST_FLAG = -post
else
    POST_FLAG =
endif

ifeq ($(PROFILE),yes)
    PROFILE_FLAG = -profile -log_view
else
    PROFILE_FLAG =
endif

TYPE_FLAG = -type $(TYPE)

# --- Command Definitions ---
PYTHON_EXEC = python3 $(RUN_DIR)

EXPORT_CMD = export OMP_NUM_THREADS=1 && export OMP_PROC_BIND=close && export OMP_PLACES=cores && export PETSC_ARCH=myconfigureopt
GPU_EXPORT_CMD = export CUDA_VISIBLE_DEVICES=0
ifeq ($(GPU),yes)
	EXPORT_CMD := $(EXPORT_CMD) && $(GPU_EXPORT_CMD)
endif
MPIRUN_CMD = $(EXPORT_CMD) && mpirun --bind-to core -n $(MPI)
KSP_FLAGS = -ksp_type fgmres -pc_type gamg -ksp_reuse_preconditioner true -snes_lag_jacobian -2 -snes_lag_preconditioner -2
HYPRE_FLAGS = -pc_hypre euclid -pc_hypre_boomeramg_trunc_factor 0.1 -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_max_levels 10
ALL_FLAGS = $(TYPE_FLAG) $(DEBUG_FLAG) $(OPT_FLAG) $(POST_FLAG) $(PROFILE_FLAG) $(KSP_FLAGS) $(GPU_FLAG)

# --- Example Definitions ---
EXAMPLE1_FLAGS = -name TPFA_Example1 -reservoir $(EXAMPLE_DIR)/example_1/reservoir.ini
EXAMPLE2_FLAGS = -name TPFA_Example2 -reservoir $(EXAMPLE_DIR)/example_2/reservoir.ini
EXAMPLE3_FLAGS = -name TPFA_Example3 -reservoir $(EXAMPLE_DIR)/example_3/reservoir.ini
EXAMPLE4_FLAGS = -name TPFA_Example4 -reservoir $(EXAMPLE_DIR)/example_4/reservoir.ini
EXAMPLELI_RESERVOIR := $(EXAMPLE_DIR)/example_Li/reservoir.ini
EXAMPLELI_FLAGS = -name TPFA_ExampleLI -reservoir $(EXAMPLELI_RESERVOIR)

EXAMPLES = example1 example2 example3 example4

.PHONY: all $(EXAMPLES) help clean compare_li test test_gpu

all: $(EXAMPLES)
# --- Prepare GPU ---
prepare_gpu:
	@echo "Preparing GPU for exclusive process mode and starting MPS daemon..."
	@sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS && sudo nvidia-cuda-mps-control -d

# --- Main Targets ---
example1:
	@echo "Running $@ with $(MPI) processes. $(LIMIT_MSG)"
	$(MPIRUN_CMD) bash -c '$(RUN_PREFIX) $(PYTHON_EXEC) $(EXAMPLE1_FLAGS) $(ALL_FLAGS)'

example2:
	@echo "Running $@ with $(MPI) processes. $(LIMIT_MSG)"
	$(MPIRUN_CMD) bash -c '$(RUN_PREFIX) $(PYTHON_EXEC) $(EXAMPLE2_FLAGS) $(ALL_FLAGS)'

example3:
	@echo "Running $@ with $(MPI) processes. $(LIMIT_MSG)"
	$(MPIRUN_CMD) bash -c '$(RUN_PREFIX) $(PYTHON_EXEC) $(EXAMPLE3_FLAGS) $(ALL_FLAGS)'

example4:
	@echo "Running $@ with $(MPI) processes. $(LIMIT_MSG)"
	$(MPIRUN_CMD) bash -c '$(RUN_PREFIX) $(PYTHON_EXEC) $(EXAMPLE4_FLAGS) $(ALL_FLAGS)'

example_li:
	@echo "Running $@ with $(MPI) processes. $(LIMIT_MSG)"
	$(MPIRUN_CMD) bash -c '$(RUN_PREFIX) $(PYTHON_EXEC) $(EXAMPLELI_FLAGS) $(ALL_FLAGS)'

test_gpu:
	@echo "Running GPU benchmark for 1 to 6 MPI processes..."
	@mkdir -p $(RESULTS_DIR)
	@for i in $$(seq 6 -1 1); do \
		echo "--- Running with $$i process(es) ---"; \
		$(EXPORT_CMD) && mpirun --bind-to core -n $$i \
			python3 $(TEST_GPU_SCRIPT) \
			--output-dir $(RESULTS_DIR) \
			--mpi-procs $$i \
			-log_view \
			-dm_mat_type hypre -dm_vec_type mpicuda; \

	done
	@echo "GPU benchmark finished. Results are in $(RESULTS_DIR)"


# --- Utility Targets ---
test:
	@echo "Running pytest with $(MPI) processes..."
	$(MPIRUN_CMD) pytest -q

compare_li:
	@echo "Running comparison for example_li..."
	@python3 $(COMPARE_DIR) -reservoir $(EXAMPLELI_RESERVOIR)

clean:
	@echo "Cleaning generated files from $(EXAMPLE_DIR)..."
	@find $(EXAMPLE_DIR) -path '*/truth' -prune -o \
		\( -name "*.log" -o -name "*.lprof" -o -name "*.vts" -o -name "*.vtk" -o -name "*.npy" -o -name "*.info" -o -name "*.bin" -o -name "*.json" \) \
		-type f -exec rm -f {} +
	@echo "Cleaning results files from $(SCRIPT_DIR)tools/scripts/results..."
	@find $(SCRIPT_DIR)tools/scripts/results -type f \
		\( -name "*.yml" -o -name "*.yaml" -o -name "*.json" -o -name "*.log" -o -name "*.png" \) \
		-exec rm -f {} +
	@echo "Cleaning etc/results..."
	@rm -f $(RESULTS_DIR)/*.png
	@echo "Clean complete."

help:
	@echo "Makefile for running TPFA examples"
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  all          Run all examples."
	@echo "  example1     Run the first example."
	@echo "  example2     Run the second example."
	@echo "  example3     Run the third example."
	@echo "  example4     Run the fourth example."
	@echo "  example_li   Run the Li example."
	@echo "  test_gpu     Run the GPU benchmark script for 1-6 processes."
	@echo "  test         Run pytest with MPI."
	@echo "  compare_li   Run the comparison for the Li example's results."
	@echo "  clean        Remove all generated log, vtk, and profile files."
	@echo "  help         Show this help message."
	@echo ""
	@echo "Options (can be set from the command line):"
	@echo "  MPI=<n>      Set the number of MPI processes (default: 1)."
	@echo "  DEBUG=<yes|no> Enable PETSc debug flags (default: no)."
	@echo "  OPT=<yes|no>   Enable optimization (skip checks) (default: yes)."
	@echo "  POST=<yes|no>  Enable post-processing VTK output (default: no)."
	@echo "  PROFILE=<yes|no> Enable performance profiling (default: no)."
	@echo "  LIMIT=<yes|no> Limit memory usage per process (default: no)."
	@echo ""
	@echo "Example:"
	@echo "  make example_li MPI=4 POST=yes LIMIT=yes"
	@echo ""
