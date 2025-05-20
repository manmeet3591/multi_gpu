import os
import torch

gpu_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if gpu_available else "None"

local_rank = int(os.environ.get("LOCAL_RANK", -1))
global_rank = int(os.environ.get("RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))
master_addr = os.environ.get("MASTER_ADDR", "Not set")
master_port = os.environ.get("MASTER_PORT", "Not set")

print(f"GPU available: {gpu_available} | Device: {device_name} | "
      f"Local rank: {local_rank} | Global rank: {global_rank} | "
      f"World size: {world_size} | Master: {master_addr}:{master_port}")
