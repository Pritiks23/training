"""
ddp_training.py

Production-style Distributed Data Parallel (DDP) training example.

Key Features:
- torchrun launcher support
- NCCL backend
- DistributedSampler
- Automatic gradient synchronization
- Mixed Precision Training (AMP)
- Throughput benchmarking
- Iteration timing
- Multi-GPU scaling visibility

Launch:

Single Node / 2 GPUs:
torchrun --nproc_per_node=2 ddp_training.py

Requirements:
- 2+ NVIDIA GPUs
- CUDA
- PyTorch with distributed support
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms, models


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 4

# Optional:
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL


# ============================================================
# SETUP DISTRIBUTED
# ============================================================

def setup():

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup():
    dist.destroy_process_group()


# ============================================================
# DATA
# ============================================================


# Create dataloaders for train, val, and test splits
def create_dataloaders(rank, world_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # ============================================================
    # CRITICAL FIX: avoid DDP download race condition
    # ============================================================

    if rank == 0:
        datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

    # force ALL processes to wait until download is complete
    dist.barrier()

    # Now safe: all ranks just load existing files
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    # Split train into train/val
    val_size = 5000
    train_size = len(train_set) - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_sampler = DistributedSampler(
        train_subset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_subset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    test_sampler = DistributedSampler(
        test_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


# ============================================================
# MODEL
# ============================================================

def create_model(device):

    model = models.resnet18(num_classes=10)

    model = model.to(device)


    # Use local_rank for device_ids, as recommended by PyTorch DDP docs
    local_rank = device.index if hasattr(device, 'index') and device.index is not None else int(str(device).split(":")[-1])
    ddp_model = DDP(
        model,
        device_ids=[local_rank]
    )

    return ddp_model


# ============================================================
# TRAINING
# ============================================================

def train():

    rank, world_size, local_rank, device = setup()

    if rank == 0:
        print("=" * 60)
        print("DDP TRAINING STARTED")
        print(f"World Size: {world_size}")
        print("=" * 60)


    train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler = create_dataloaders(rank, world_size)
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    scaler = torch.cuda.amp.GradScaler()
    total_training_start = time.time()
    model.train()

    def evaluate(loader, sampler, desc):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            sampler.set_epoch(0)
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += images.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        model.train()
        return avg_loss, accuracy

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        total_samples = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            iteration_start = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            backward_start = time.time()
            scaler.scale(loss).backward()
            torch.cuda.synchronize(device)
            backward_time = time.time() - backward_start
            scaler.step(optimizer)
            scaler.update()
            batch_size_actual = images.size(0)
            running_loss += loss.item()
            total_samples += batch_size_actual
            iteration_time = time.time() - iteration_start
            throughput = batch_size_actual / iteration_time
            if batch_idx % 50 == 0:
                allocated_memory = (
                    torch.cuda.memory_allocated(device) / 1024**3
                )
                print(
                    f"[Rank {rank}] "
                    f"Epoch {epoch+1} "
                    f"Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Iter Time: {iteration_time:.4f}s | "
                    f"Backward Time: {backward_time:.4f}s | "
                    f"Throughput: {throughput:.2f} samples/s | "
                    f"GPU Memory: {allocated_memory:.2f} GB"
                )
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        samples_per_second = total_samples / epoch_time
        val_loss, val_acc = evaluate(val_loader, val_sampler, desc="Validation")
        if rank == 0:
            print("\n" + "=" * 60)
            print(f"EPOCH {epoch+1} COMPLETE")
            print(f"Average Train Loss: {avg_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc*100:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Samples/sec per GPU: {samples_per_second:.2f}")
            print("=" * 60 + "\n")

    total_training_time = time.time() - total_training_start
    test_loss, test_acc = evaluate(test_loader, test_sampler, desc="Test")
    if rank == 0:
        print("=" * 60)
        print("TRAINING FINISHED")
        print(f"Total Training Time: {total_training_time:.2f}s")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
        print("=" * 60)
    cleanup()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is required."

    gpu_count = torch.cuda.device_count()

    assert gpu_count >= 2, (
        f"Requires at least 2 GPUs. Found {gpu_count}"
    )

    train()
"""
ddp_training.py

Production-style Distributed Data Parallel (DDP) training example.

Key Features:
- torchrun launcher support
- NCCL backend
- DistributedSampler
- Automatic gradient synchronization
- Mixed Precision Training (AMP)
- Throughput benchmarking
- Iteration timing
- Multi-GPU scaling visibility

Launch:

Single Node / 2 GPUs:
torchrun --nproc_per_node=2 ddp_training.py

Requirements:
- 2+ NVIDIA GPUs
- CUDA
- PyTorch with distributed support
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Only check for GPUs if running distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            assert gpu_count >= 2, (
                f"Requires at least 2 GPUs. Found {gpu_count}"
            )
    train()
