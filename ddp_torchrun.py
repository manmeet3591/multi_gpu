import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyDataset(data.Dataset):
    def __init__(self, num_samples=1000):
        self.x = torch.randn(num_samples, 10)
        self.y = torch.randn(num_samples, 5)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )

    def forward(self, x):
        return self.net(x)


def train_ddp(local_rank, world_size, batch_size=32, epochs=5):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize process group
    dist.init_process_group(backend="nccl")

    # Dataset and distributed sampler
    dataset = ToyDataset()
    sampler = data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=dist.get_rank())
    dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Model setup
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)  # ensure proper shuffling
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Rank {dist.get_rank()}] Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item()}", flush=True)

        avg_loss = total_loss / len(dataloader)
        print(f"[Rank {dist.get_rank()}] Epoch {epoch+1} Average Loss: {avg_loss:.4f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()

    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Initializing DDP training: local_rank={local_rank}, world_size={world_size}", flush=True)
    print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)

    train_ddp(local_rank, world_size)


################################
#### Working code below
#################################

# import argparse
# import os
# import sys
# import tempfile
# from urllib.parse import urlparse

# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP


# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def demo_basic(local_world_size, local_rank):
#     # Get number of devices assigned to this process
#     device_count = torch.cuda.device_count()
#     device_ids = list(range(device_count))

#     print(
#         f"[{os.getpid()}] rank = {dist.get_rank()}, "
#         f"world_size = {dist.get_world_size()}, "
#         f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}, "
#         f"device_count: {device_count}, device_ids = {device_ids}",
#         flush=True
#     )

#     model = ToyModel().cuda(device_ids[0])
#     ddp_model = DDP(model, device_ids=device_ids)

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     # Run a few training steps to confirm everything works
#     for epoch in range(3):
#         optimizer.zero_grad()
#         inputs = torch.randn(20, 10).to(device_ids[0])
#         outputs = ddp_model(inputs)
#         labels = torch.randn(20, 5).to(device_ids[0])
#         loss = loss_fn(outputs, labels)
#         print(f"[{os.getpid()}] rank = {dist.get_rank()} | Epoch {epoch+1} | Loss: {loss.item()}", flush=True)
#         loss.backward()
#         optimizer.step()
#         print(f"[{os.getpid()}] rank = {dist.get_rank()} | Epoch {epoch+1} training step completed", flush=True)


# def spmd_main(local_world_size, local_rank):
#     env_dict = {
#         key: os.environ[key]
#         for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
#     }

#     print(f"[{os.getpid()}] Initializing process group with: {env_dict}", flush=True)
#     print("torch.cuda.is_available():", torch.cuda.is_available(), flush=True)
#     print("torch.cuda.device_count():", torch.cuda.device_count(), flush=True)

#     dist.init_process_group(backend="nccl")

#     print(
#         f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
#         f"rank = {dist.get_rank()}, backend={dist.get_backend()}",
#         flush=True
#     )

#     demo_basic(local_world_size, local_rank)
#     dist.destroy_process_group()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument("--local_world_size", type=int, default=1)
#     args = parser.parse_args()

#     spmd_main(args.local_world_size, args.local_rank)


#########################
##### working code below
#########################

# import argparse
# import os
# import sys
# import tempfile
# from urllib.parse import urlparse

# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP


# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def demo_basic(local_world_size, local_rank):
#     # Get number of devices assigned to this process
#     device_count = torch.cuda.device_count()
#     device_ids = list(range(device_count))

#     print(
#         f"[{os.getpid()}] rank = {dist.get_rank()}, "
#         f"world_size = {dist.get_world_size()}, "
#         f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}, "
#         f"device_count: {device_count}, device_ids = {device_ids} \n",
#         end=''
#     )

#     model = ToyModel().cuda(device_ids[0])
#     ddp_model = DDP(model, device_ids=device_ids)

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10).to(device_ids[0]))
#     labels = torch.randn(20, 5).to(device_ids[0])
#     loss = loss_fn(outputs, labels)
#     loss.backward()
#     optimizer.step()


# def spmd_main(local_world_size, local_rank):
#     env_dict = {
#         key: os.environ[key]
#         for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
#     }

#     print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
#     print("torch.cuda.is_available():", torch.cuda.is_available())
#     print("torch.cuda.device_count():", torch.cuda.device_count())

#     dist.init_process_group(backend="nccl")

#     print(
#         f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
#         f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n",
#         end=''
#     )

#     demo_basic(local_world_size, local_rank)
#     dist.destroy_process_group()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument("--local_world_size", type=int, default=1)
#     args = parser.parse_args()

#     spmd_main(args.local_world_size, args.local_rank)
