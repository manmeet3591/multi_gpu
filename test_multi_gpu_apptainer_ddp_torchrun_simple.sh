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

export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$SLURM_NTASKS

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"

CONTAINER=/scratch/08105/ms86336/wind_1km_1hr/apptainer_multi_gpu.sif

# Launch 2 tasks with correct env vars
for ((i=0; i<$SLURM_NTASKS; i++)); do
  export RANK=$i
  export LOCAL_RANK=0  # If each node has only 1 GPU
  export CUDA_VISIBLE_DEVICES=0
  ibrun -n 1 -o $i apptainer exec --nv $CONTAINER python ddp_torchrun_simple.py &
done

wait
