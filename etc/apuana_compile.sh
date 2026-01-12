#!/bin/bash
#SBATCH --job-name=petsc_compile
#SBATCH --output=logs/petsc_compile_%j.out
#SBATCH --error=logs/petsc_compile_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=short-simple
#SBATCH --time=04:00:00

set -eo pipefail

echo "=== PETSc Compilation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================="

mkdir -p $HOME/logs

# Modules
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
module load OpenMPI/4.1.5-GCC-12.2.0

nvidia-smi

# Environment
source $HOME/envs/petsc_env/bin/activate

export PETSC_DIR=$HOME/petsc
export PETSC_ARCH="myconfigureoptapuana"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Clone if needed
if [ ! -d "$PETSC_DIR" ]; then
    cd $HOME
    git clone -b release https://gitlab.com/petsc/petsc.git
fi

cd $PETSC_DIR

# Run YOUR compile script
$HOME/compile_petsc.sh --petsc_dir=$PETSC_DIR --mode=OPT --target=APUANA --stage=1

echo "=== Done ==="
date