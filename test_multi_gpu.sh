#!/bin/bash
#SBATCH -J vista_gh_test        # Job name
#SBATCH -o test_output.%j.out   # Standard output
#SBATCH -e test_error.%j.err    # Standard error
#SBATCH -N 2                    # Request 2 nodes
#SBATCH -n 2                    # 1 task per node
#SBATCH -t 00:10:00             # 10 minutes wall time
#SBATCH -p gh                   # GH partition
#SBATCH -A EAR24019    # Replace with your TACC allocation/project

# Load necessary modules
module load gcc cuda
module load python3

# Simple test: Print node info and run a basic GPU command
echo "Running on nodes:"
scontrol show hostname $SLURM_JOB_NODELIST

echo "CUDA Devices:"
ibrun -np 2 python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Optional: Test inter-node MPI communication
ibrun -np 2 hostname
