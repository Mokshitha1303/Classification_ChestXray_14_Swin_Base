from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class SplitItem:
    filename: str
    labels: np.ndarray  # shape (C,)


def load_disease_map(path: str | Path) -> List[str]:
    """
    disease_map.txt format (comma-separated):
      Atelectasis, Cardiomegaly, ..., Hernia
    """
    path = Path(path)
    text = path.read_text().strip()
    # Allow either comma-separated or line-separated
    if "," in text:
        names = [x.strip() for x in text.split(",") if x.strip()]
    else:
        names = [x.strip() for x in text.splitlines() if x.strip()]
    if len(names) == 0:
        raise ValueError(f"Empty disease_map file: {path}")
    return names


def load_split_file(path: str | Path, num_classes: int) -> List[SplitItem]:
    path = Path(path)
    items: List[SplitItem] = []
    for ln, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != (1 + num_classes):
            raise ValueError(
                f"Bad line in {path} at line {ln}: expected {1+num_classes} columns, got {len(parts)}"
            )
        fname = parts[0]
        labels = np.asarray([int(x) for x in parts[1:]], dtype=np.float32)
        items.append(SplitItem(filename=fname, labels=labels))
    if len(items) == 0:
        raise ValueError(f"No samples found in split file: {path}")
    return items


def compute_pos_weight(train_items: Sequence[SplitItem]) -> np.ndarray:
    """
    pos_weight for BCEWithLogitsLoss:
      pos_weight[c] = num_negative[c] / num_positive[c]
    """
    y = np.stack([it.labels for it in train_items], axis=0)  # (N,C)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    # Avoid divide by zero
    pos = np.clip(pos, 1.0, None)
    return (neg / pos).astype(np.float32)


def build_image_index(images_dir: str | Path) -> Dict[str, str]:
    """
    Recursively scan images_dir and return {filename: full_path}.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

    index: Dict[str, str] = {}
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(".png"):
                full = str(Path(root) / fn)
                # If duplicates exist, keep the first and ignore the rest
                if fn not in index:
                    index[fn] = full
    if len(index) == 0:
        raise RuntimeError(f"No PNG images found under: {images_dir}")
    return index


def load_or_build_image_index(
    images_dir: str | Path,
    cache_path: str | Path,
    rebuild: bool = False,
) -> Dict[str, str]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        with cache_path.open("r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid index cache format: {cache_path}")
        return {str(k): str(v) for k, v in obj.items()}

    index = build_image_index(images_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(index, f)
    return index


def verify_split_files_exist(items: Sequence[SplitItem], index: Dict[str, str]) -> List[str]:
    missing = [it.filename for it in items if it.filename not in index]
    return missing
