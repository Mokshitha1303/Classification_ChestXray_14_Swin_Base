from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data_utils import SplitItem


class ChestXray14Dataset(Dataset):
    def __init__(
        self,
        items: List[SplitItem],
        image_index: Dict[str, str],
        transform: Optional[Callable] = None,
        strict: bool = True,
        return_filename: bool = False,
    ) -> None:
        self.items = items
        self.image_index = image_index
        self.transform = transform
        self.strict = strict
        self.return_filename = return_filename

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        path = self.image_index.get(item.filename)
        if path is None:
            if self.strict:
                raise FileNotFoundError(f"Image not found in index: {item.filename}")
            # fallback: try raw filename (rare)
            path = item.filename

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = torch.from_numpy(item.labels).float()

        if self.return_filename:
            return img, y, item.filename
        return img, y
