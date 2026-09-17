from __future__ import annotations

import datetime as dt
import os

import torch
import torch.distributed as dist


def initialize(backend="nccl", timeout_seconds=600):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with world_size > 1 requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=dt.timedelta(seconds=int(timeout_seconds)),
        )
    return local_rank, world_size


def is_initialized():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_initialized() else 0


def world_size():
    return dist.get_world_size() if is_initialized() else 1


def is_main_process():
    return rank() == 0


def barrier():
    if is_initialized():
        dist.barrier()


def all_ranks_true(value, device):
    if not is_initialized():
        return bool(value)
    flag = torch.tensor(int(bool(value)), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def gather_object_to_main(value):
    if not is_initialized():
        return [value]
    destination = [None for _ in range(world_size())] if is_main_process() else None
    dist.gather_object(value, destination, dst=0)
    return destination


def broadcast_object_from_main(value):
    if not is_initialized():
        return value
    values = [value if is_main_process() else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def mean_tensor_dict(values):
    if not is_initialized():
        return values
    names = list(values)
    stacked = torch.stack([values[name].detach() for name in names])
    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
    stacked /= world_size()
    return {name: value for name, value in zip(names, stacked)}


def destroy():
    if is_initialized():
        dist.destroy_process_group()
