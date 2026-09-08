from __future__ import annotations

import torch


def normalize_image(tensor, mean, std):
    """Normalize a CHW or NCHW tensor with channel-order-aware values."""
    if tensor.ndim not in (3, 4):
        raise ValueError("Expected a CHW or NCHW image tensor")
    shape = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)
    mean_tensor = torch.as_tensor(
        mean, dtype=tensor.dtype, device=tensor.device
    ).view(shape)
    std_tensor = torch.as_tensor(
        std, dtype=tensor.dtype, device=tensor.device
    ).view(shape)
    if mean_tensor.shape[-3] != tensor.shape[-3]:
        raise ValueError("Image normalization channels do not match the tensor")
    if torch.any(std_tensor == 0):
        raise ValueError("Image normalization std must be non-zero")
    return (tensor - mean_tensor) / std_tensor
