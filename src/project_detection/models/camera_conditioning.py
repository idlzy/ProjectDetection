"""Camera calibration inputs for optional intrinsic-conditioned models."""

import torch


def forward_with_targets(model, images, targets):
    """Run a detector with per-image intrinsics when conditioning is enabled."""
    if not getattr(model, "camera_conditioning", False):
        return model(images)
    if len(targets) != images.shape[0]:
        raise ValueError("One camera matrix is required per image")
    try:
        camera_matrices = torch.stack(
            [target["camera_matrix"] for target in targets]
        ).to(device=images.device, dtype=images.dtype)
    except KeyError as error:
        raise ValueError("camera_matrix is required for camera conditioning") from error
    return model(images, camera_matrices)
