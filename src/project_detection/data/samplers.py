from __future__ import annotations

import torch
from torch.utils.data import Sampler


class EpochRandomSampler(Sampler):
    """Deterministic single-rank shuffle, allowing mid-epoch recovery."""

    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(
            torch.randperm(len(self.data_source), generator=generator).tolist()
        )

    def __len__(self):
        return len(self.data_source)


class DistributedEvalSampler(Sampler):
    """Shard evaluation data without padding or duplicated samples."""

    def __init__(self, data_source, num_replicas, rank):
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.data_source = data_source
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.data_source), self.num_replicas))

    def __len__(self):
        size = len(self.data_source)
        if self.rank >= size:
            return 0
        return (size - 1 - self.rank) // self.num_replicas + 1
