import os
from pathlib import Path
from typing import Any

import torch


def save_atomic(state: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint is not a mapping")
    return loaded
