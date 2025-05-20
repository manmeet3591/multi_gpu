!/bin/bash
#SBATCH -J apptainer_multi_gpu
#SBATCH -o apptainer.%j.out
#SBATCH -e apptainer.%j.err
#SBATCH -N 2                      # Number of nodes
#SBATCH -n 2                      # Total number of tasks/processes
#SBATCH -t 00:10:00               # Wall time
#SBATCH -p gh                     # GPU queue
#SBATCH -A EAR24019              # Allocation/project

module load tacc-apptainer

export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$SLURM_NTASKS
export CONTAINER=/scratch/08105/ms86336/wind_1km_1hr/apptainer_multi_gpu.sif

# Get hostname of the first node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"

# Launch each task manually
for ((i=0; i<$SLURM_NTASKS; i++)); do
  export RANK=$i
  export LOCAL_RANK=0             # One GPU per node (adjust if more GPUs used per node)
  export CUDA_VISIBLE_DEVICES=0   # Use GPU 0 on each node

  ibrun -n 1 -o $i apptainer exec --nv $CONTAINER \
    python ddp_torchrun.py \
    --local_rank=$LOCAL_RANK \
    --local_world_size=1 &
done

wait
