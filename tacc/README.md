apptainer build  apptainer_multi_gpu.sif apptainer_multi_gpu.def

login1.vista(1059)$ idev -p gh -N 2 -n 2 -t 48:00:00

c640-042[gh](1045)$ ibrun -np 2 ./run_multigpu.sh c640-042 2

# Be verty careful about the master node number
