#!/bin/bash
#SBATCH --job-name=petsc_compile_gpu
#SBATCH --output=logs/petsc_compile_%j.out
#SBATCH --error=logs/petsc_compile_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=long-complex
#SBATCH --time=06:00:00

set -eo pipefail

echo "Starting PETSc GPU compilation"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "----------------------------------------"

mkdir -p logs

nvidia-smi

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
module load OpenMPI/4.1.5-GCC-12.2.0

ENV_NAME="petsc_env"
if [ ! -d "$HOME/envs/$ENV_NAME" ]; then
    python3 -m venv $HOME/envs/$ENV_NAME
fi

source $HOME/envs/$ENV_NAME/bin/activate
pip install --upgrade pip
pip install numpy mpi4py

export PETSC_DIR=$HOME/petsc
export PETSC_ARCH="myconfigureoptapuana"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

if [ ! -d "$PETSC_DIR" ]; then
    cd $HOME
    git clone https://gitlab.com/petsc/petsc.git
fi

cd $PETSC_DIR

$HOME/compile_petsc.sh --petsc_dir=$PETSC_DIR --mode=OPT --target=APUANA --stage=1

echo "Compilation completed"
date