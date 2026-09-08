"""Dataset package with lazy training dependencies.

Keeping the PyTorch/OpenCV dataset lazy allows manifest utilities such as
``tools/split_dataset.py`` to run in a lightweight local Python environment.
"""

__all__ = ["Mw3dReadyDataset", "collate_detection_batch"]


def __getattr__(name):
    if name in __all__:
        from .mw3d import Mw3dReadyDataset, collate_detection_batch

        return {
            "Mw3dReadyDataset": Mw3dReadyDataset,
            "collate_detection_batch": collate_detection_batch,
        }[name]
    raise AttributeError(name)
