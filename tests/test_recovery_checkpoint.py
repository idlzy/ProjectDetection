import random

import numpy as np
import torch

from project_detection.engine import (
    EpochRandomSampler,
    load_checkpoint,
    save_checkpoint,
)


def test_epoch_random_sampler_is_repeatable_per_epoch():
    dataset = list(range(32))
    sampler = EpochRandomSampler(dataset, seed=3407)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)

    assert list(sampler) == first
    assert second != first
    assert sorted(first) == list(range(len(dataset)))


def test_recovery_checkpoint_preserves_step_and_training_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint_path = tmp_path / "recovery.pth"

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=3,
        best_metric=0.25,
        config={"name": "test"},
        step_in_epoch=41,
    )
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    checkpoint, _ = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        restore_random_state=True,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["step_in_epoch"] == 41
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)
