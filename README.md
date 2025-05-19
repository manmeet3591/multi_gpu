# multi_gpu

idev -p gh -N 1 -n 1 -t 48:00:00 # N - number of nodes, n - number of mpi jobs requested
scontrol show hostnames $SLURM_JOB_NODELIST
ssh c502-002 - for one node
srun -N4 --ntasks-per-node=1 nvidia-smi
srun --ntasks=4 -N4 hostname
