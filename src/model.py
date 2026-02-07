from __future__ import annotations

from typing import Any, Dict

import timm
import torch


def create_swin_model(
    model_name: str,
    num_classes: int,
    img_size: int | None = None,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    grad_checkpointing: bool = False,
) -> torch.nn.Module:
    """
    Create a Swin model via timm and adapt head for multi-label classification.
    """
    # NOTE:
    #   timm Swin variants like "*_224" default to img_size=224.
    #   For 512×512 training we must override img_size at model creation,
    #   otherwise timm's PatchEmbed will assert the input resolution.
    kwargs: Dict[str, Any] = dict(
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    if img_size is not None:
        kwargs["img_size"] = int(img_size)

    try:
        m = timm.create_model(model_name, **kwargs)
    except TypeError as e:
        # Older timm versions may not accept img_size for some models.
        # If that happens, upgrade timm (recommended). We fall back to
        # creating the model without img_size so at least the error is clear.
        if "img_size" in str(e):
            raise TypeError(
                "Your timm version does not accept img_size for this model. "
                "Please upgrade timm (pip install -U timm)."
            ) from e
        raise

    # Enable gradient checkpointing if supported
    if grad_checkpointing and hasattr(m, "set_grad_checkpointing"):
        try:
            m.set_grad_checkpointing(enable=True)
        except Exception:
            pass
    elif grad_checkpointing and hasattr(m, "grad_checkpointing"):
        try:
            m.grad_checkpointing = True
        except Exception:
            pass

    return m
