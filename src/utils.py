from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_torch_flags(tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    # TF32 is a nice speed boost on A100 for matmul/conv while keeping decent accuracy.
    if tf32:
        # Newer PyTorch prefers the fp32_precision knobs (avoids deprecation warnings).
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
        except Exception:
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            # cudnn conv precision is controlled separately
            torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
        except Exception:
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_channels_last(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return model.to(memory_format=torch.channels_last)
    except Exception:
        return model


def tensor_to_channels_last(x: torch.Tensor) -> torch.Tensor:
    try:
        return x.contiguous(memory_format=torch.channels_last)
    except Exception:
        return x


@dataclass
class AverageMeter:
    name: str
    fmt: str = ".4f"
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} (avg: {self.avg:{self.fmt}})"


def pretty_dict(d: Dict[str, Any], indent: int = 0) -> str:
    pad = " " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(pretty_dict(v, indent=indent + 2))
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def parse_args_common():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--images_dir", type=str, default="", help="Override data.images_dir")
    p.add_argument("--checkpoint", type=str, default="", help="Checkpoint path (for test.py)")
    p.add_argument("--resume", type=str, default="", help="Checkpoint path to resume training")
    p.add_argument("--rebuild_index", action="store_true", help="Force rebuild filename->path index")
    return p
