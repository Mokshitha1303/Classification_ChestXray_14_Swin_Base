# ChestX-ray14 Swin-Base

Details of the Hyperparameters are as follows:

- **512×512** inputs
- **Weighted BCEWithLogitsLoss** (`pos_weight` computed from training split)
- **EMA (Exponential Moving Average)** weights (optional, enabled by default)
- **Strict / official split files** (no random splitting)
- **Resume-safe checkpoints** (`checkpoints/last.pth`)
- **Top-K checkpoint saving** (default: keep top-3 under `checkpoints/topk/`)
- **Early stopping** on validation loss (patience = 5 by default)

## 1) Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the dataset
You need the **PNG images** for NIH ChestX-ray14 on disk (Kaggle “nih-chest-xrays” PNG version is commonly used).

The images can be nested under folders like:
`images_001/images`, `images_002/images`, …

The code will **recursively scan** your `images_dir` and build a cache mapping:
`filename -> full_path`

### 3. Set the image root directory
Edit `configs/train_512_ema.yaml`:

```yaml
data:
  images_dir: /ABSOLUTE/PATH/TO/ALL_PNG_IMAGES
```

### 4. Train
```bash
bash scripts/train.sh
# or
python -m src.train --config configs/train_512_ema.yaml
```

### Resume training (after a crash / timeout)
```bash
python -m src.train --config configs/train_512_ema.yaml --resume checkpoints/last.pth
```

Checkpoints written during training:
- `checkpoints/last.pth` — always the latest (resume-safe)
- `checkpoints/best.pth` — best by `train.best_metric` (resume-safe)
- `checkpoints/topk/` — top-K model weights (default: top-3). `checkpoints/topk/topk.json` tracks ranks.

### 5. Test / Evaluate
```bash
python -m src.test --config configs/train_512_ema.yaml --checkpoint checkpoints/best.pth
```

Outputs:
- `checkpoints/history.csv` (epoch metrics)
- `checkpoints/test_per_class_auc.txt` (per-class + mean AUC)
- `plots/test_roc_all_classes.png` (ROC curves)

Checkpoint files:
- `checkpoints/last.pth` — resume-safe (model + optimizer + scheduler + scaler + EMA)
- `checkpoints/best.pth` — best by `train.best_metric` (resume-safe)
- `checkpoints/topk/*.pth` — top-K lightweight checkpoints (model + EMA + metrics)
- `checkpoints/topk/topk.json` — metadata for the kept top-K

---

## 2) Official splits (STRICT)

This repo uses the provided split files under `data/`:

- `data/train_official.txt`
- `data/val_official.txt`
- `data/test_official.txt`

Each line is space-separated:
```
00000001_000.png 0 1 0 0 0 0 0 0 0 0 0 0 0 0
```

The 14 classes are read from:
- `data/disease_map.txt`

---

## 3) Notes for A100 GPU

Defaults are tuned for a single modern GPU (AMP enabled).  
If you hit OOM at 512×512, reduce `batch_size` or increase `grad_accum_steps`.

We also enable:
- TF32 matmul (good for A100)
- optional `torch.compile` (off by default)

### About the Swin model name ending in `_224`
Even if the timm model name is `swin_base_patch4_window7_224`, this repo **overrides the model's `img_size` at creation**
to match `train.img_size` (default **512**). This avoids the common timm error:
`AssertionError: Input height (512) doesn't match model (224)`.

---

## 4) Repo structure

```
cxr14_swinbase_repo/
  configs/
    train_512_ema.yaml
  data/
    train_official.txt
    val_official.txt
    test_official.txt
    disease_map.txt
  src/
    data_utils.py
    dataset.py
    ema_utils.py
    metrics.py
    model.py
    train.py
    test.py
    utils.py
  scripts/
    train.sh
  checkpoints/
  plots/
  requirements.txt
  README.md
```

---

## 5) Repro tips

- The dataset scan index is cached at `data/image_index.json`.
- If you move your images directory, delete that cache or set `data.rebuild_index: true`.

---

## Disclaimer

ChestX-ray14 labels are text-mined and can be noisy. This repo is for research/benchmarking.
