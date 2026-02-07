from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# AMP API compatibility (torch.amp vs torch.cuda.amp)
try:  # PyTorch >= 2.0
    from torch.amp import autocast  # type: ignore

    _TORCH_AMP_API = True
except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast  # type: ignore

    _TORCH_AMP_API = False
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .data_utils import (
    load_disease_map,
    load_or_build_image_index,
    load_split_file,
    verify_split_files_exist,
)
from .dataset import ChestXray14Dataset
from .metrics import compute_auc_per_class, compute_roc_curves
from .model import create_swin_model
from .utils import get_device, load_yaml, parse_args_common, set_torch_flags


def build_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if _TORCH_AMP_API:
        return autocast(device_type=device.type, enabled=True)
    return autocast(enabled=True)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_targets = []
    for batch in tqdm(loader, desc="Test", leave=False):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with _autocast_ctx(device, enabled=amp):
            logits = model(x)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_targets.append(y.detach().float().cpu().numpy())
    return np.concatenate(all_targets, axis=0), np.concatenate(all_logits, axis=0)


def main() -> None:
    args = parse_args_common().parse_args()
    cfg = load_yaml(args.config)

    if args.images_dir:
        cfg.setdefault("data", {})["images_dir"] = args.images_dir
    if args.rebuild_index:
        cfg.setdefault("data", {})["rebuild_index"] = True

    images_dir = cfg["data"].get("images_dir", "")
    if not images_dir:
        raise ValueError("Config requires data.images_dir (absolute path to images).")

    set_torch_flags(
        tf32=bool(cfg.get("train", {}).get("tf32", True)),
        cudnn_benchmark=bool(cfg.get("train", {}).get("cudnn_benchmark", True)),
    )
    device = get_device()
    print("Using device:", device)

    repo_root = Path(__file__).resolve().parents[1]
    splits_dir = repo_root / cfg["data"].get("splits_dir", "data")
    disease_map_path = splits_dir / cfg["data"]["disease_map"]
    class_names = load_disease_map(disease_map_path)
    num_classes = int(cfg["model"].get("num_classes", len(class_names)))

    test_split = splits_dir / cfg["data"]["test_split"]
    test_items = load_split_file(test_split, num_classes)

    index_cache = repo_root / cfg["data"].get("index_cache_path", "data/image_index.json")
    image_index = load_or_build_image_index(
        images_dir=images_dir,
        cache_path=index_cache,
        rebuild=bool(cfg["data"].get("rebuild_index", False)),
    )

    if bool(cfg["data"].get("strict", True)):
        missing = verify_split_files_exist(test_items, image_index)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} filenames from test split are missing under images_dir. "
                f"Example: {missing[:5]}"
            )

    img_size = int(cfg["train"]["img_size"])
    tfm = build_eval_transform(img_size)

    test_ds = ChestXray14Dataset(test_items, image_index, transform=tfm, strict=True, return_filename=False)

    eval_bs = int(cfg.get("eval", {}).get("batch_size", 32))
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=int(cfg.get("eval", {}).get("num_workers", 8)),
        pin_memory=bool(cfg["train"].get("pin_memory", True)),
        persistent_workers=bool(cfg["train"].get("persistent_workers", True)),
        drop_last=False,
    )

    # Model
    model = create_swin_model(
        model_name=str(cfg["model"]["name"]),
        num_classes=num_classes,
        img_size=img_size,
        pretrained=False,  # we will load checkpoint
        drop_rate=float(cfg["model"].get("drop_rate", 0.0)),
        drop_path_rate=float(cfg["model"].get("drop_path_rate", 0.1)),
        grad_checkpointing=False,
    )
    model.to(device)

    # AMP is only meaningful on CUDA.
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    # Load checkpoint
    ckpt_path = args.checkpoint or str((Path(cfg["train"].get("save_dir", "checkpoints")) / "best.pth"))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    use_ema = bool(cfg.get("eval", {}).get("use_ema", True)) and (ckpt.get("ema") is not None)
    if use_ema:
        model.load_state_dict(ckpt["ema"], strict=True)
        print(f"Loaded EMA weights from: {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded model weights from: {ckpt_path}")

    # Predict
    y_true, logits = predict_logits(model, test_loader, device=device, amp=amp)
    probs = 1.0 / (1.0 + np.exp(-logits))

    auc_res = compute_auc_per_class(y_true, probs)
    print("Test mean AUC:", auc_res.mean_auc)

    # Save per-class AUC
    save_dir = repo_root / Path(cfg["train"].get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    out_txt = save_dir / "test_per_class_auc.txt"
    with out_txt.open("w") as f:
        for name, val in zip(class_names, auc_res.per_class):
            f.write(f"{name}\t{val:.6f}\n")
        f.write(f"MEAN\t{auc_res.mean_auc:.6f}\n")
    print("Saved:", out_txt)

    # Plot ROC curves
    curves = compute_roc_curves(y_true, probs)
    plot_dir = repo_root / Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_dir / "test_roc_all_classes.png"

    plt.figure(figsize=(10, 8))
    for name, curve in zip(class_names, curves):
        if curve is None:
            continue
        fpr, tpr, c_auc = curve
        plt.plot(fpr, tpr, label=f"{name} (AUC={c_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ChestX-ray14 — Test ROC Curves (All Classes)")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
