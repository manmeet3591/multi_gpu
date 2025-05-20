#!/bin/bash
#SBATCH -J apptainer_multi_gpu
#SBATCH -o apptainer.%j.out
#SBATCH -e apptainer.%j.err
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 00:10:00
#SBATCH -p gh
#SBATCH -A EAR24019

module load tacc-apptainer

# Path to container image
CONTAINER=/scratch/08105/ms86336/wind_1km_1hr/apptainer_multi_gpu.sif

# Python code to test GPU visibility
PY_CMD="import torch; print(f'GPU available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run Python code inside container using Apptainer + --nv
ibrun -np 2 apptainer exec --nv $CONTAINER python3 -c "$PY_CMD"
