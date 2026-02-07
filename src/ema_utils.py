from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch


class ModelEMA:
    """
    Simple EMA for model weights (works for most PyTorch modules).
    Keeps an internal copy of the model and updates it after optimizer steps.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device: Optional[str] = None) -> None:
        self.decay = float(decay)
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        if device is not None and device != "":
            self.ema.to(device=device)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k, v in ema_state.items():
            if k not in model_state:
                continue
            model_v = model_state[k].detach()
            if not torch.is_floating_point(v):
                ema_state[k].copy_(model_v)
            else:
                v.mul_(self.decay).add_(model_v, alpha=(1.0 - self.decay))

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)

    def to(self, device: torch.device):
        self.ema.to(device=device)
        return self
