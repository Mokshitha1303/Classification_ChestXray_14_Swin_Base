from __future__ import annotations

import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# AMP API has moved from torch.cuda.amp -> torch.amp in newer PyTorch.
# We support both to keep the repo usable across versions.
try:  # PyTorch >= 2.0
    from torch.amp import GradScaler, autocast  # type: ignore

    _TORCH_AMP_API = True
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

    _TORCH_AMP_API = False
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .data_utils import (
    compute_pos_weight,
    load_disease_map,
    load_or_build_image_index,
    load_split_file,
    verify_split_files_exist,
)
from .dataset import ChestXray14Dataset
from .ema_utils import ModelEMA
from .metrics import compute_auc_per_class
from .model import create_swin_model
from .utils import (
    ensure_dir,
    get_device,
    load_yaml,
    parse_args_common,
    seed_everything,
    set_torch_flags,
    tensor_to_channels_last,
    to_channels_last,
    pretty_dict,
)


def _autocast_ctx(device: torch.device, enabled: bool):
    """Compatibility helper for torch.amp.autocast vs torch.cuda.amp.autocast."""
    if not enabled:
        return nullcontext()
    if _TORCH_AMP_API:
        # torch.amp.autocast requires a device_type argument.
        return autocast(device_type=device.type, enabled=True)
    return autocast(enabled=True)


def build_transforms(img_size: int, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    losses = []
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with _autocast_ctx(device, enabled=amp):
            logits = model(x)
            loss = bce(logits, y)

        losses.append(loss.item())
        all_logits.append(logits.detach().float().cpu().numpy())
        all_targets.append(y.detach().float().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))

    auc_res = compute_auc_per_class(targets_np, probs_np)
    return float(np.mean(losses)), auc_res.mean_auc, targets_np, probs_np


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _infer_mode(metric_name: str, override: str | None = None) -> str:
    """Infer optimization direction for a metric.

    - losses -> "min"
    - AUCs/accuracies -> "max"
    You can override with cfg.train.best_mode or cfg.train.early_stopping.mode.
    """

    if override is not None:
        ov = str(override).strip().lower()
        if ov in {"min", "max"}:
            return ov

    name = str(metric_name).strip().lower()
    if "loss" in name or name.endswith("_loss"):
        return "min"
    return "max"


def _get_metric(metric_name: str, train_loss: float, val_loss: float, val_mean_auc: float) -> float:
    """Resolve a metric name to a scalar value computed in this script."""
    key = str(metric_name).strip().lower()
    if key in {"val_loss", "valid_loss", "validation_loss", "loss"}:
        return float(val_loss)
    if key in {"train_loss"}:
        return float(train_loss)
    if key in {"val_mean_auc", "mean_auc", "auc"}:
        return float(val_mean_auc)
    raise ValueError(
        f"Unknown metric name: {metric_name}. Supported: val_mean_auc, val_loss, train_loss"
    )


def _is_improvement(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return current < (best - float(min_delta))
    return current > (best + float(min_delta))


def _sort_topk(records: list[dict], mode: str) -> list[dict]:
    return sorted(records, key=lambda r: float(r["metric"]), reverse=(mode == "max"))


def main() -> None:
    args = parse_args_common().parse_args()
    cfg = load_yaml(args.config)

    if args.images_dir:
        cfg.setdefault("data", {})["images_dir"] = args.images_dir
    if args.resume:
        cfg.setdefault("train", {})["resume"] = args.resume
    if args.rebuild_index:
        cfg.setdefault("data", {})["rebuild_index"] = True

    # Basic config sanity
    images_dir = cfg["data"].get("images_dir", "")
    if not images_dir:
        raise ValueError("Config requires data.images_dir (absolute path to images).")

    seed = int(cfg.get("train", {}).get("seed", 42))
    seed_everything(seed)

    set_torch_flags(
        tf32=bool(cfg.get("train", {}).get("tf32", True)),
        cudnn_benchmark=bool(cfg.get("train", {}).get("cudnn_benchmark", True)),
    )

    device = get_device()
    print("Using device:", device)
    print("Config:\n" + pretty_dict(cfg))

    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    splits_dir = repo_root / cfg["data"].get("splits_dir", "data")

    train_split = splits_dir / cfg["data"]["train_split"]
    val_split = splits_dir / cfg["data"]["val_split"]
    test_split = splits_dir / cfg["data"]["test_split"]
    disease_map_path = splits_dir / cfg["data"]["disease_map"]

    class_names = load_disease_map(disease_map_path)
    num_classes = int(cfg["model"].get("num_classes", len(class_names)))
    assert num_classes == len(class_names), "model.num_classes must equal len(disease_map)"

    # Load splits (STRICT)
    train_items = load_split_file(train_split, num_classes)
    val_items = load_split_file(val_split, num_classes)
    test_items = load_split_file(test_split, num_classes)

    # Build or load image index
    index_cache = repo_root / cfg["data"].get("index_cache_path", "data/image_index.json")
    image_index = load_or_build_image_index(
        images_dir=images_dir,
        cache_path=index_cache,
        rebuild=bool(cfg["data"].get("rebuild_index", False)),
    )

    if bool(cfg["data"].get("strict", True)):
        missing = (
            verify_split_files_exist(train_items, image_index)
            + verify_split_files_exist(val_items, image_index)
            + verify_split_files_exist(test_items, image_index)
        )
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} filenames from split files are missing under images_dir. "
                f"Example: {missing[:5]}"
            )

    # pos_weight from training labels
    pos_weight_np = compute_pos_weight(train_items)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)

    # Data
    img_size = int(cfg["train"]["img_size"])
    train_tf = build_transforms(img_size, train=True)
    eval_tf = build_transforms(img_size, train=False)

    train_ds = ChestXray14Dataset(train_items, image_index, transform=train_tf, strict=True)
    val_ds = ChestXray14Dataset(val_items, image_index, transform=eval_tf, strict=True)
    test_ds = ChestXray14Dataset(test_items, image_index, transform=eval_tf, strict=True)

    batch_size = int(cfg["train"]["batch_size"])
    eval_bs = int(cfg.get("eval", {}).get("batch_size", batch_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 8)),
        pin_memory=bool(cfg["train"].get("pin_memory", True)),
        persistent_workers=bool(cfg["train"].get("persistent_workers", True)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=int(cfg.get("eval", {}).get("num_workers", cfg["train"].get("num_workers", 8))),
        pin_memory=bool(cfg["train"].get("pin_memory", True)),
        persistent_workers=bool(cfg["train"].get("persistent_workers", True)),
        drop_last=False,
    )

    # Model
    model = create_swin_model(
        model_name=str(cfg["model"]["name"]),
        num_classes=num_classes,
        img_size=img_size,
        pretrained=bool(cfg["model"].get("pretrained", True)),
        drop_rate=float(cfg["model"].get("drop_rate", 0.0)),
        drop_path_rate=float(cfg["model"].get("drop_path_rate", 0.1)),
        grad_checkpointing=bool(cfg["model"].get("grad_checkpointing", False)),
    )
    model.to(device)

    if bool(cfg["train"].get("channels_last", True)):
        model = to_channels_last(model)

    if bool(cfg["train"].get("compile", False)):
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as e:
            print("torch.compile failed, continuing without it:", e)

    # EMA
    ema_cfg = cfg["train"].get("ema", {}) or {}
    use_ema = bool(ema_cfg.get("enabled", True))
    ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.9999))) if use_ema else None

    # Loss / Optim / Sched
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=tuple(cfg["train"].get("betas", [0.9, 0.999])),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 1))
    min_lr = float(cfg["train"].get("min_lr", 1e-6))

    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # cosine decay
        progress = (current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # scale from 1.0 to min_lr/lr
        return cosine * (1.0 - min_lr / float(cfg["train"]["lr"])) + (min_lr / float(cfg["train"]["lr"]))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # AMP is only meaningful on CUDA.
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    if _TORCH_AMP_API:
        # torch.amp.GradScaler signature varies by PyTorch version.
        # Newer versions use `device=` (not `device_type=`). We also want CPU runs to work.
        try:
            scaler = GradScaler(device=device.type, enabled=amp)  # torch>=2.0
        except TypeError:
            # Fallback: older torch.amp.GradScaler may not accept `device=`; use default device.
            scaler = GradScaler(enabled=amp)
    else:
        scaler = GradScaler(enabled=amp)

    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    clip_grad = float(cfg["train"].get("clip_grad_norm", 0.0))

    save_dir = ensure_dir(repo_root / Path(cfg["train"].get("save_dir", "checkpoints")))
    history_path = save_dir / "history.csv"
    best_path = save_dir / "best.pth"
    last_path = save_dir / "last.pth"

    # Checkpoint selection (best + top-k)
    best_metric_name = str(cfg["train"].get("best_metric", "val_mean_auc"))
    best_mode = _infer_mode(best_metric_name, cfg["train"].get("best_mode", None))
    best_min_delta = float(cfg["train"].get("best_min_delta", 0.0))

    # Track best so far for resume-safe selection
    best_metric = float("inf") if best_mode == "min" else -float("inf")

    # Save the best K checkpoints (in addition to best.pth / last.pth)
    save_top_k = int(cfg["train"].get("save_top_k", 1))
    topk_dir = ensure_dir(save_dir / "topk") if save_top_k and save_top_k > 1 else None
    topk_state: dict = {"k": save_top_k, "metric": best_metric_name, "mode": best_mode, "records": []}

    # Early stopping (optional)
    es_cfg = cfg["train"].get("early_stopping", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    es_monitor = str(es_cfg.get("monitor", "val_loss"))
    es_mode = _infer_mode(es_monitor, es_cfg.get("mode", None))
    es_patience = int(es_cfg.get("patience", 5))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_best = float("inf") if es_mode == "min" else -float("inf")
    es_bad_epochs = 0

    # Resume (optional)
    resume_path = cfg["train"].get("resume", "")
    start_epoch = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
            ema.load_state_dict(ckpt["ema"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = int(ckpt.get("epoch", 0) + 1)
        best_metric = float(ckpt.get("best_metric", best_metric))

        # Restore early stopping state (if available)
        es_state = ckpt.get("early_stopping_state", {}) or {}
        if isinstance(es_state, dict):
            es_best = float(es_state.get("best", es_best))
            es_bad_epochs = int(es_state.get("bad_epochs", es_bad_epochs))

        # Restore top-k state (if available)
        ck_topk = ckpt.get("topk_state", {}) or {}
        if isinstance(ck_topk, dict) and ck_topk.get("records") is not None:
            topk_state["records"] = ck_topk.get("records", [])

        # Clean & prune top-k records to current K
        if topk_dir is not None:
            records = [
                r
                for r in (topk_state.get("records") or [])
                if isinstance(r, dict) and r.get("path") and os.path.exists(r["path"])
            ]
            records = _sort_topk(records, best_mode)[:save_top_k]
            topk_state["records"] = records
            try:
                with (topk_dir / "topk.json").open("w") as f:
                    json.dump(topk_state, f, indent=2)
            except Exception:
                pass
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    # CSV history header
    if not history_path.exists():
        with history_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_mean_auc", "time_sec"])

    # Train loop
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if bool(cfg["train"].get("channels_last", True)):
                x = tensor_to_channels_last(x)

            with _autocast_ctx(device, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y) / float(grad_accum)

            scaler.scale(loss).backward()

            if (step % grad_accum) == 0:
                if clip_grad and clip_grad > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

            running_loss += loss.item() * float(grad_accum) * x.size(0)
            n_seen += x.size(0)

            if step % int(cfg["train"].get("log_interval", 50)) == 0:
                pbar.set_postfix(loss=running_loss / max(1, n_seen), lr=optim.param_groups[0]["lr"])

        scheduler.step()

        train_loss = running_loss / max(1, n_seen)

        # Validation (use EMA weights by default)
        eval_model = ema.ema if (ema is not None and bool(cfg.get("eval", {}).get("use_ema", True))) else model
        val_loss, val_mean_auc, _, _ = evaluate(eval_model, val_loader, device=device, amp=amp)

        epoch_time = time.time() - epoch_start
        lr_now = optim.param_groups[0]["lr"]

        # Metric selection for checkpoints
        metric_value = _get_metric(best_metric_name, train_loss, val_loss, val_mean_auc)

        # Early stopping update (monitors *validation loss* by default)
        if es_enabled:
            monitor_value = _get_metric(es_monitor, train_loss, val_loss, val_mean_auc)
            if _is_improvement(monitor_value, es_best, es_mode, es_min_delta):
                es_best = float(monitor_value)
                es_bad_epochs = 0
            else:
                es_bad_epochs += 1

        # Update best metric (for best.pth)
        improved_best = _is_improvement(metric_value, best_metric, best_mode, best_min_delta)
        if improved_best:
            best_metric = float(metric_value)

        # Print epoch summary
        extra = f" {best_metric_name}={metric_value:.6f}"
        if es_enabled:
            extra += f" | early_stop({es_monitor}) bad_epochs={es_bad_epochs}/{es_patience} best={es_best:.6f}"
        print(
            f"[Epoch {epoch+1}] lr={lr_now:.3e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mean_auc={val_mean_auc:.4f}" + extra
        )

        # Prepare shared state
        model_state = model.state_dict()
        ema_state = ema.state_dict() if ema is not None else None

        early_state = {
            "enabled": es_enabled,
            "monitor": es_monitor,
            "mode": es_mode,
            "patience": es_patience,
            "min_delta": es_min_delta,
            "best": es_best,
            "bad_epochs": es_bad_epochs,
        }

        # Save top-K checkpoints (lightweight: model + ema + metrics)
        if topk_dir is not None:
            records = [
                r
                for r in (topk_state.get("records") or [])
                if isinstance(r, dict) and r.get("path") and os.path.exists(r["path"])
            ]

            qualifies = len(records) < save_top_k
            if not qualifies and records:
                worst = min(float(r["metric"]) for r in records) if best_mode == "max" else max(float(r["metric"]) for r in records)
                qualifies = _is_improvement(metric_value, worst, best_mode, min_delta=0.0)

            if qualifies:
                mtag = f"{metric_value:.6f}".replace(".", "p").replace("-", "m")
                ck_name = f"epoch{epoch+1:03d}_{best_metric_name}_{mtag}.pth"
                ck_path = topk_dir / ck_name
                save_checkpoint(
                    ck_path,
                    {
                        "epoch": epoch,
                        "model": model_state,
                        "ema": ema_state,
                        "best_metric_name": best_metric_name,
                        "metric_value": float(metric_value),
                        "val_loss": float(val_loss),
                        "val_mean_auc": float(val_mean_auc),
                        "class_names": class_names,
                        "config": cfg,
                    },
                )
                records.append(
                    {
                        "metric": float(metric_value),
                        "epoch": int(epoch),
                        "path": str(ck_path),
                        "val_loss": float(val_loss),
                        "val_mean_auc": float(val_mean_auc),
                    }
                )
                records = _sort_topk(records, best_mode)
                while len(records) > save_top_k:
                    dropped = records.pop(-1)
                    try:
                        os.remove(dropped["path"])
                    except FileNotFoundError:
                        pass

                topk_state["records"] = records
                try:
                    with (topk_dir / "topk.json").open("w") as f:
                        json.dump(topk_state, f, indent=2)
                except Exception:
                    pass

        # Full checkpoint payload (resume-safe)
        payload = {
            "epoch": epoch,
            "model": model_state,
            "ema": ema_state,
            "optim": optim.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "best_mode": best_mode,
            "class_names": class_names,
            "config": cfg,
            "metrics": {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mean_auc": float(val_mean_auc),
                best_metric_name: float(metric_value),
            },
            "early_stopping_state": early_state,
            "topk_state": topk_state,
        }

        # Save best (resume-safe)
        if bool(cfg["train"].get("save_best", True)) and improved_best:
            save_checkpoint(best_path, payload)
            print(f"Saved new best checkpoint: {best_path} ({best_metric_name}={best_metric:.6f})")

        # Save last (always)
        save_checkpoint(last_path, payload)

        # Append CSV
        with history_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, lr_now, train_loss, val_loss, val_mean_auc, epoch_time])

        # Early stopping decision
        if es_enabled and es_bad_epochs >= es_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"No improvement in {es_monitor} for {es_patience} epochs (best={es_best:.6f})."
            )
            break

    print(f"Training complete. Best {best_metric_name}: {best_metric}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
