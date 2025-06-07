#!/bin/bash

MASTER_ADDR=$1
NNODES=$2
NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}

echo "MASTER_ADDR=$MASTER_ADDR"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"

apptainer exec \
  --bind /scratch/08105/ms86336:/opt/notebooks \
  --nv apptainer_multi_gpu.sif \
  torchrun \
    --nproc_per_node=1 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    /scratch/08105/ms86336/wind_1km_1hr/multinode.py 10 2 --batch_size 32
