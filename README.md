# multi_gpu

1. idev -p gh -N 16 -n 16 -t 48:00:00 # N - number of nodes, n - number of mpi jobs requested
This logs interactively into the grace hopper nodes
2. c609-101[gh](999)$ scontrol show hostnames $SLURM_JOB_NODELIST

c609-101

c609-102

c609-111

c609-112

c609-121

c609-122

c609-131

c609-132

c609-141

c609-142

c609-151

c609-152

c610-001

c610-002

c610-011

4. ssh c502-002 - for one node

5. srun -N4 --ntasks-per-node=1 nvidia-smi

Displays nvidia-smi output for all the 4 nodes, if we put N16, then it will show for 16 nodes

6. srun --ntasks=4 -N4 hostname

c609-101.vista.tacc.utexas.edu

c609-102.vista.tacc.utexas.edu

c609-112.vista.tacc.utexas.edu

c609-111.vista.tacc.utexas.edu

7. sbatch submit_script.sh

8. squeue -u $USER

9. scancel job_id
