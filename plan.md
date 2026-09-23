# Crowd Counting with Density Maps (MCNN): Development Plan

## How to use this plan (instructions for Claude Code)

- Implement the project **one module at a time, in order**. Do not start a module until the previous one meets its acceptance criteria.
- After finishing each module: run its tests, report what was built, and **stop for review** before continuing.
- Keep every path, hyperparameter, and dataset location in `configs/default.yaml`. No hardcoded paths in code.
- Code must run in three environments: local CPU (for tests and small smoke runs), Google Colab (GPU), and Kaggle (GPU). Detect the device automatically.
- Prefer small, readable functions with type hints and docstrings. No framework magic (no PyTorch Lightning); plain PyTorch.

---

## 1. Project summary

Estimate the number of people in an image by predicting a **density map** whose pixel values sum to the crowd count. The model is **MCNN** (Multi-Column CNN, Zhang et al., CVPR 2016), trained on the **ShanghaiTech** dataset (Part A: dense crowds, Part B: sparser street scenes). Training happens on Colab/Kaggle GPUs; the final deliverable includes a **Gradio web demo** for image (and optionally video) inference with heatmap visualisation.

## 2. Tech stack

| Purpose | Choice |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch, torchvision |
| Data / math | numpy, scipy (`KDTree`), h5py |
| Images / video | opencv-python, Pillow, matplotlib |
| Config | PyYAML |
| Progress / logging | tqdm, TensorBoard (+ CSV log) |
| Web demo | Gradio |
| Testing | pytest |

## 3. Repository structure

```
crowd-counting-mcnn/
├── configs/
│   └── default.yaml
├── data/                      # gitignored; raw + processed data
├── checkpoints/               # gitignored
├── outputs/                   # gitignored; predictions, videos, plots
├── notebooks/
│   ├── train_colab.ipynb
│   └── train_kaggle.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py              # load/merge YAML + CLI overrides
│   ├── utils.py               # seeding, device, logging, checkpoint I/O
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parse_annotations.py
│   │   ├── density.py         # density map generation
│   │   ├── preprocess.py      # CLI: raw dataset -> processed .h5 files
│   │   ├── dataset.py         # torch Dataset + transforms
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── mcnn.py
│   ├── train.py               # CLI entry point
│   ├── evaluate.py            # CLI entry point
│   ├── inference.py           # image + video inference (CLI + importable API)
│   └── visualize.py           # heatmap overlays, plots
├── app/
│   └── gradio_app.py
├── tests/
│   ├── test_density.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_train_smoke.py
│   └── test_inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Modules

### Module 0: Project scaffolding and config

**Goal:** A runnable skeleton with configuration, utilities, and tooling.

**Tasks**
- Create the directory structure above, `requirements.txt`, `.gitignore`, and a stub `README.md`.
- `configs/default.yaml` with sections: `paths` (raw_data_dir, processed_dir, checkpoint_dir, output_dir), `data` (part: `A` or `B`, crop_size, val_fraction, downsample=4), `density` (mode: `adaptive` or `fixed`, fixed_sigma, beta, k_neighbors), `train` (epochs, batch_size, lr, weight_decay, optimizer, seed, num_workers, log_every, density_scale), `inference` (max_side, overlay_alpha, video_frame_stride).
- `src/config.py`: load YAML, allow dotted CLI overrides (e.g. `--set train.lr=1e-4`), return a simple nested object or dict.
- `src/utils.py`: `set_seed`, `get_device` (cuda > mps > cpu), `save_checkpoint` / `load_checkpoint` (model, optimizer, epoch, best_mae, config), a logger that writes to console and file.

**Acceptance criteria**
- `pip install -r requirements.txt` succeeds.
- `python -c "from src.config import load_config; print(load_config('configs/default.yaml'))"` works.
- `pytest` runs (even with zero tests collected yet) without import errors.

---

### Module 1: Data preparation and density map generation

**Goal:** Convert raw ShanghaiTech data (images + `.mat` head annotations) into processed image/density-map pairs.

**Dataset layout (as downloaded, e.g. from Kaggle):**
```
ShanghaiTech/part_A/train_data/images/IMG_1.jpg
ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_1.mat
ShanghaiTech/part_A/test_data/...
ShanghaiTech/part_B/...
```
Head coordinates live at `mat["image_info"][0, 0][0, 0][0]` as an `(N, 2)` array of `(x, y)`. Make the root path configurable, since the folder name differs between Kaggle mirrors; search for `part_A` / `part_B` case-insensitively.

**Tasks**
- `parse_annotations.py`: `load_points(mat_path) -> np.ndarray (N, 2)`.
- `density.py`:
  - `generate_density_map(shape, points, mode, fixed_sigma, beta, k)`.
  - **Adaptive (Part A default):** for each head, sigma = `beta * mean distance to k nearest neighbours` (beta=0.3, k=3) using `scipy.spatial.KDTree`. If only one point exists, fall back to `fixed_sigma`.
  - **Fixed (Part B default):** constant sigma (default 4).
  - Implement efficiently: place a **locally computed, truncated Gaussian kernel** (radius ≈ 3·sigma) around each point, not a full-image `gaussian_filter` per point.
  - **Renormalise each kernel after cropping at image borders** so every head contributes exactly 1.0 to the sum.
  - Clip points that fall outside the image bounds.
- `preprocess.py` (CLI): for a given part, iterate over train and test splits, generate density maps, save each as `processed/part_X/{train,test}/IMG_n.h5` containing the density map (float32), plus a copy/symlink of the image. Show a tqdm progress bar. Skip files that already exist unless `--overwrite`.
- Write a `processed/part_X/splits.json` holding a reproducible train/val split (val_fraction of train, seeded). ShanghaiTech has no official val set.
- Add a `--visualize N` flag that saves N side-by-side image/density-map PNGs to `outputs/` for sanity checking.

**Tests (`tests/test_density.py`)**
- Synthetic image with 5 points: `abs(density.sum() - 5) < 1e-3` in both modes.
- A point on the image corner still sums to 1.0.
- Empty points array returns an all-zero map of the right shape.
- Output dtype is float32 and shape equals input `(H, W)`.

**Acceptance criteria**
- Preprocessing Part B completes and, for every file, the density sum matches the annotation count within 1e-2.
- Visualised samples look like blobs sitting on heads.

---

### Module 2: Dataset, transforms, and DataLoader

**Goal:** A PyTorch `Dataset` that returns training-ready tensors.

**Tasks**
- `CrowdDataset(split, cfg, train: bool)` reading from the processed directory and `splits.json`.
- Returns `(image_tensor [3,H,W], density_tensor [1,H/4,W/4], count float)`.
- **Training transforms** (`transforms.py`), applied identically to image and density:
  - Random crop of `crop_size` (default 256). Pad with zeros if the image is smaller than the crop.
  - Random horizontal flip.
  - Optional mild colour jitter / grayscale on the image only.
  - ImageNet mean/std normalisation on the image.
- **Eval transforms:** full image, padded on the bottom/right to a multiple of 4. Batch size 1.
- **Downsampling the target:** MCNN outputs at 1/4 resolution, so downsample the density map by **4×4 sum-pooling** (preserves the count exactly). Do not use bilinear resize without count correction.
- Optionally multiply the target by `train.density_scale` (default 1; try 100 if the model collapses to predicting zeros). Everything downstream must divide by the same factor.
- `build_dataloaders(cfg)` returning train/val/test loaders.

**Tests (`tests/test_dataset.py`)** (use a tiny synthetic processed dataset created in a pytest fixture)
- Shapes are correct and the density map is exactly 1/4 of the image size.
- The density sum after cropping ≤ full-image count; with no crop, it equals the full count within 1e-3.
- Horizontal flip keeps the count unchanged.

**Acceptance criteria**
- Iterating one epoch of the Part B train loader on CPU works without errors.

---

### Module 3: MCNN model

**Goal:** Implement MCNN faithful to the original paper.

**Architecture**
Three parallel columns, each: conv → ReLU → maxpool(2) → conv → ReLU → maxpool(2) → conv → ReLU → conv → ReLU. Padding keeps spatial size ("same"), so the two pools produce a 1/4-resolution output.

| Column | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
|---|---|---|---|---|
| Large | 9×9, 16 | 7×7, 32 | 7×7, 16 | 7×7, 8 |
| Medium | 7×7, 20 | 5×5, 40 | 5×5, 20 | 5×5, 10 |
| Small | 5×5, 24 | 3×3, 48 | 3×3, 24 | 3×3, 12 |

Concatenate the column outputs (8+10+12 = 30 channels) and fuse with a 1×1 conv to 1 channel. Add a final ReLU so the density is non-negative.

**Tasks**
- `src/models/mcnn.py` with class `MCNN(in_channels=3)`.
- Weight init: normal(0, 0.01) for conv weights, zero bias.
- `count_parameters()` helper; print the parameter count on model creation.

**Tests (`tests/test_model.py`)**
- Input `[2, 3, 256, 256]` → output `[2, 1, 64, 64]`.
- Input with non-square dims divisible by 4 (e.g. 768×1024) → correct 1/4 output.
- Output is non-negative.
- Forward + backward pass runs on CPU.

---

### Module 4: Training pipeline

**Goal:** A robust training script that survives Colab/Kaggle disconnects.

**Tasks**
- `src/train.py` CLI: `python -m src.train --config configs/default.yaml --set data.part=B`.
- Loss: pixel-wise MSE between predicted and target density maps (sum reduction divided by batch size, or mean with `density_scale`; make it configurable and document which is default).
- Optimiser: Adam, lr 1e-5 default (configurable), optional weight decay. Optional `ReduceLROnPlateau` on val MAE.
- Each epoch: train, then validate on the val split computing **MAE and RMSE of counts** (full images, batch size 1).
- Checkpointing:
  - `last.pth` every epoch (for resuming).
  - `best.pth` when val MAE improves.
  - `--resume path/to/last.pth` restores model, optimiser, scheduler, epoch, best MAE.
  - `paths.checkpoint_dir` should be pointable at Google Drive / `/kaggle/working`.
- Logging: TensorBoard scalars (train loss, val MAE, val RMSE, lr) plus a `metrics.csv`. Every K epochs, save a PNG comparing a val image, its GT density, and the predicted density.
- Mixed precision (`torch.cuda.amp`) when running on CUDA, toggleable.
- Seeded and deterministic where practical.

**Tests (`tests/test_train_smoke.py`)**
- On a synthetic mini dataset (e.g. 4 images), 2 epochs on CPU complete, write `last.pth` and `metrics.csv`, and resume works.
- Overfit check: training on a single image drives loss down by at least 50%.

**Acceptance criteria**
- A short Colab run on Part B shows val MAE decreasing over the first ~20 epochs.

---

### Module 5: Evaluation

**Goal:** Report standard metrics on the official test split.

**Tasks**
- `src/evaluate.py` CLI: `--checkpoint`, `--part`, `--save-predictions`.
- Compute MAE and RMSE over the test set; print a summary table and save `outputs/eval_part_X.json`.
- Optional: save per-image predictions (GT count, predicted count, abs error) to CSV, and density visualisations for the 10 worst-error images.
- Plot predicted vs GT counts (scatter with y=x line) to `outputs/`.

**Reference numbers (original MCNN paper)**

| Split | MAE | RMSE |
|---|---|---|
| Part A | 110.2 | 173.2 |
| Part B | 26.4 | 41.3 |

**Acceptance criteria for this reimplementation:** Part B MAE < 35 and Part A MAE < 130 is a reasonable target. Being within ~20% of the paper is fine.

---

### Module 6: Inference (image and video)

**Goal:** A reusable inference API plus a CLI, used by the web demo.

**Tasks**
- `src/inference.py`:
  - `class CrowdCounter` that loads a checkpoint once and exposes:
    - `predict_image(img: np.ndarray | PIL.Image) -> (count: float, density: np.ndarray)`.
    - `predict_video(path, frame_stride) -> list[(frame_idx, count)]` and optionally writes an annotated output video.
  - Preprocessing: convert to RGB, resize so the longest side ≤ `inference.max_side` (default 1024) keeping aspect ratio, normalise, pad to a multiple of 4. Crop padding off the output; divide by `density_scale`.
  - Resizing changes the image area but the density sum is still the count; no correction is needed as long as the density is summed, not resized.
- `src/visualize.py`:
  - `density_to_heatmap(density, target_size)`: upsample for display only, normalise, apply a colormap (e.g. `cv2.COLORMAP_JET`).
  - `overlay(image, heatmap, alpha)`.
  - `draw_count(image, count)`: count text box in a corner.
- CLI: `python -m src.inference --checkpoint best.pth --input path/to/img_or_video --output outputs/`.
- Video output: annotated MP4 with heatmap overlay and a running count, plus a CSV of counts per processed frame.

**Tests (`tests/test_inference.py`)**
- With a randomly initialised model, `predict_image` returns a float and a non-negative 2D array for images of odd sizes (e.g. 333×517).
- Heatmap overlay returns an image the same size as the input.
- Video inference on a tiny generated 10-frame video writes an output file and a CSV.

---

### Module 7: Gradio web demo

**Goal:** A simple, clean demo anyone can use from a browser.

**Tasks**
- `app/gradio_app.py` using `gr.Blocks`:
  - **Image tab:** upload image → displays estimated count (large number), heatmap overlay, and raw density map. Includes an overlay-opacity slider.
  - **Model selector:** "Dense crowds (Part A)" / "Sparse crowds (Part B)", mapping to two checkpoints defined in config. Load models lazily and cache them.
  - **Video tab (optional, behind a flag):** upload short video → annotated video + count-over-time line plot. Cap duration/frames to keep it responsive.
  - Example images in `app/examples/` (a few ShanghaiTech test images).
  - Clear error message if no checkpoint is found, telling the user where to put it.
- CLI flags: `--share` (for Colab public link), `--port`, `--checkpoint-a`, `--checkpoint-b`.
- Must run on CPU for inference.

**Acceptance criteria**
- `python app/gradio_app.py` launches locally on CPU with trained weights and returns a result for an uploaded image in a few seconds.
- Launching from Colab with `--share` produces a working public URL.

---

### Module 8: Notebooks and documentation

**Goal:** One-click training on Colab/Kaggle and a clear README.

**Tasks**
- `notebooks/train_colab.ipynb`:
  1. Check GPU (`nvidia-smi`).
  2. Mount Google Drive.
  3. Clone the repo and `pip install -r requirements.txt`.
  4. Download ShanghaiTech via the Kaggle API (user uploads `kaggle.json`) into `/content/data`.
  5. Run preprocessing for Parts A and B.
  6. Train with `paths.checkpoint_dir` pointing into Drive; include a resume cell.
  7. Evaluate on the test set and display plots.
  8. Launch the Gradio demo with `--share`.
- `notebooks/train_kaggle.ipynb`: same flow, but the dataset is attached as a Kaggle input (read-only under `/kaggle/input/...`), processed data and checkpoints go to `/kaggle/working`.
- `README.md`: project overview, how density maps work (short), setup, preprocessing, training, evaluation, inference, demo, results table (fill in after training), and screenshots of the demo.

---

## 5. Common pitfalls to guard against

- **Model outputs all zeros:** densities are tiny per pixel, so MSE can be minimised by predicting zero. Remedies: `density_scale` (e.g. 100), a lower/higher lr, checking that targets are not accidentally zeroed by the downsampling step.
- **Count drift from resizing density maps:** always sum-pool or rescale by the area factor; add assertions in the dataset.
- **Colab disconnects:** always write checkpoints to Drive and support `--resume`.
- **Mismatched image/density augmentation:** apply crops/flips with shared random parameters.
- **Grayscale images in Part A:** some images are single-channel; convert everything to RGB at load time.
- **Odd image sizes at inference:** pad to a multiple of 4 and crop the output back.

## 6. Out of scope (possible future extensions)

- Stronger backbones (CSRNet, DM-Count loss, transformer-based counters).
- Training on UCF-QNRF or NWPU-Crowd.
- Real-time webcam/RTSP stream counting.
- Export to ONNX / TorchScript for faster CPU inference.
- Region-of-interest counting and crowd-density alerts.
