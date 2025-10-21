#!/bin/bash
set -e
set -u
set -o pipefail

usage() {
    echo "Error: Invalid arguments."
    echo "Usage: $0 --petsc_dir=<PATH> --mode=<OPT|DEBUG> --target=<PERSONAL|APUANA> [--stage=<NUMBER>]"
    echo "Example: $0 --petsc_dir=/home/user/petsc --mode=OPT --target=APUANA --stage=3"
    exit 1
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PETSC_DIR_ARG=""
MODE_ARG=""
TARGET_ARG=""
STAGE_ARG=1

for i in "$@"; do
    case $i in
        --petsc_dir=*)
            PETSC_DIR_ARG="${i#*=}"
            shift
            ;;
        --mode=*)
            MODE_ARG="${i#*=}"
            shift
            ;;
        --target=*)
            TARGET_ARG="${i#*=}"
            shift
            ;;
        --stage=*)
            STAGE_ARG="${i#*=}"
            shift
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$PETSC_DIR_ARG" || -z "$MODE_ARG" || -z "$TARGET_ARG" ]]; then
    usage
fi

if [[ ! -d "$PETSC_DIR_ARG" ]]; then
    echo "Error: PETSc directory not found at '${PETSC_DIR_ARG}'"
    exit 1
fi

MODE=${MODE_ARG^^}
TARGET=${TARGET_ARG^^}

if [[ "$MODE" != "OPT" && "$MODE" != "DEBUG" ]]; then
    echo "Error: Mode must be 'OPT' or 'DEBUG'."
    usage
fi

if [[ "$TARGET" != "PERSONAL" && "$TARGET" != "APUANA" ]]; then
    echo "Error: Target must be 'PERSONAL' or 'APUANA'."
    usage
fi

if ! [[ "$STAGE_ARG" =~ ^[0-9]+$ ]]; then
    echo "Error: --stage must be a non-negative integer."
    usage
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export PETSC_DIR=$(realpath "${PETSC_DIR_ARG}")

if [[ "$TARGET" == "APUANA" ]]; then
    LOG_DIR="$HOME/petsc_compile_logs"
    MAKE_NP="-j16"
else
    LOG_DIR="/tmp/compile_petsc"
    MAKE_NP=""
fi

INSTALL_SUBDIR="build-install"

if [[ "$MODE" == "DEBUG" ]]; then
    if [[ "$TARGET" == "APUANA" ]]; then
        export PETSC_ARCH="myconfiguredebugapuana"
        CONFIGURE_SCRIPT="${SCRIPT_DIR}/myconfiguredebugapuana.py"
    else
        export PETSC_ARCH="myconfiguredebug"
        CONFIGURE_SCRIPT="${SCRIPT_DIR}/myconfiguredebug.py"
    fi
else
    if [[ "$TARGET" == "APUANA" ]]; then
        export PETSC_ARCH="myconfigureoptapuana"
        CONFIGURE_SCRIPT="${SCRIPT_DIR}/myconfigureoptapuana.py"
    else
        export PETSC_ARCH="myconfigureopt"
        CONFIGURE_SCRIPT="${SCRIPT_DIR}/myconfigureopt.py"
    fi
fi

if [[ ! -f "${CONFIGURE_SCRIPT}" ]]; then
    echo "Error: Configuration script not found at '${CONFIGURE_SCRIPT}'"
    exit 1
fi

run_step() {
    local step_num=$1
    local step_name=$2
    local log_file=$3
    local command_to_run=("${@:4}")
    
    if [[ $step_num -lt $STAGE_ARG ]]; then
        echo "--> [${step_num}/7] ${step_name}... Skipped."
        return
    fi
    
    echo -n "--> [${step_num}/7] ${step_name}..."
    rm -f "${log_file}"
    "${command_to_run[@]}" &> "${log_file}"
    local ret=$?
    if [ $ret -ne 0 ]; then
        echo " FAILED! Check log: ${log_file}"
        tail -20 "${log_file}"
        exit $ret
    fi
    echo " Done."
}

if [[ $STAGE_ARG -le 1 ]]; then
    rm -rf "${LOG_DIR}"
fi
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "Starting PETSc build in ${MODE} mode for ${TARGET}."
echo " - Starting from stage: ${STAGE_ARG}"
echo " - Detailed logs will be saved in: ${LOG_DIR}"
echo "--------------------------------------------------------"

cd "${PETSC_DIR}"

run_step 1 "Cleaning previous build" "${LOG_DIR}/01-clean.log" bash -c "make distclean || true"
run_step 2 "Configuring PETSc" "${LOG_DIR}/02-configure.log" "${CONFIGURE_SCRIPT}"
run_step 3 "Building PETSc libraries" "${LOG_DIR}/03-build_all.log" make ${MAKE_NP} all
run_step 4 "Installing PETSc to staging dir" "${LOG_DIR}/04-install_petsc.log" make DESTDIR="${PETSC_DIR}/${INSTALL_SUBDIR}" install
run_step 5 "Building petsc4py" "${LOG_DIR}/05-pip_build.log" bash -c "pip uninstall -y petsc4py || true && pip install ."
run_step 6 "Installing petsc4py (manual)" "${LOG_DIR}/06-petsc4py_install.log" bash -c "cd src/binding/petsc4py && make install && pip install ."
run_step 7 "Checking PETSc build" "${LOG_DIR}/07-check.log" make check

echo "--------------------------------------------------------"
echo "PETSc and petsc4py build process completed successfully."
echo "========================================================"