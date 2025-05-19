# multi_gpu

1. idev -p gh -N 1 -n 1 -t 48:00:00 # N - number of nodes, n - number of mpi jobs requested
2. scontrol show hostnames $SLURM_JOB_NODELIST
3. ssh c502-002 - for one node
4. srun -N4 --ntasks-per-node=1 nvidia-smi
5. srun --ntasks=4 -N4 hostname
