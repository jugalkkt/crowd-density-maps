# Crowd Counting with Density Maps (MCNN)

Estimate how many people are in an image by predicting a **density map** whose
pixels sum to the crowd count. The model is **MCNN** (Multi-Column CNN,
[Zhang et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)),
trained on **ShanghaiTech** Part A (dense crowds) and Part B (street scenes),
with a Gradio demo for image and video inference.

---

## How density-map counting works

Detecting individual people fails in dense crowds — heads are a handful of
pixels and occlude each other. Instead, each annotated head becomes a
normalised 2-D Gaussian in a **density map**, so the map integrates to the
number of people. The network regresses that map, and the count is just its
sum. As a bonus you get a spatial picture of *where* the crowd is.

Two ways to choose the Gaussian width:

| Mode | sigma | Used for |
|---|---|---|
| `fixed` | a constant (default 4 px) | Part B, where heads are roughly one size |
| `adaptive` | `beta * mean distance to the k nearest heads` (beta 0.3, k 3) | Part A, where perspective makes head size vary hugely |

Heads near an image border get a kernel that is clipped by the edge, so each
kernel is **renormalised after cropping** — every head contributes exactly 1.0
and the map's sum matches the annotation count.

MCNN has three parallel columns with 9x9, 7x7 and 5x5 first-layer kernels, so
different columns specialise in different head sizes. Two 2x max-pools make the
output **1/4 of the input resolution**, which is why targets are reduced by
4x4 **sum**-pooling (a bilinear resize would not preserve the count).

---

## Setup

```bash
git clone https://github.com/jugalkkt/crowd-density-maps.git
cd crowd-density-maps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                 # ~60 tests, all CPU, under a minute
```

Runs on local CPU, Colab and Kaggle; the device (`cuda` > `mps` > `cpu`) is
detected automatically. Every path and hyperparameter lives in
[configs/default.yaml](configs/default.yaml) — nothing is hardcoded. Override
any entry from the CLI:

```bash
python -m src.train --set train.lr=1e-4 --set data.part=A
```

### Getting the data

Download ShanghaiTech (e.g. the `tthien/shanghaitech` Kaggle dataset) and point
`paths.raw_data_dir` at whatever folder contains `part_A` / `part_B`:

```
<raw_data_dir>/
└── ShanghaiTech/
    ├── part_A/
    │   ├── train_data/{images,ground-truth}/
    │   └── test_data/{images,ground-truth}/
    └── part_B/...
```

The part folder is found case-insensitively and recursively, so mirrors with a
different top-level name work without edits.

---

## Pipeline

### 1. Preprocess

```bash
python -m src.data.preprocess --part B --verify --visualize 5
python -m src.data.preprocess --part A --set density.mode=adaptive --verify
```

Writes `data/processed/part_X/{train,test}/IMG_n.h5` (float32 density map plus
the head count), links the images alongside them, and records a seeded
train/val split in `splits.json` — ShanghaiTech ships no official val set.
`--verify` checks every density sum against its annotation count;
`--visualize N` saves side-by-side PNGs to `outputs/density_samples/` so you can
confirm the blobs sit on heads. Existing files are skipped unless `--overwrite`.

### 2. Train

```bash
python -m src.train --set data.part=B
python -m src.train --set data.part=B --resume checkpoints/part_B/last.pth
```

- Loss: pixel-wise MSE, summed over pixels and divided by the batch size
  (`train.loss_reduction=sum_over_batch`, the paper's objective). `mean` is
  available but pairs badly with tiny per-pixel densities — see *Pitfalls*.
- Adam at lr 1e-5, optional `ReduceLROnPlateau` on validation MAE.
- Random 256x256 crops, horizontal flips, ImageNet normalisation. Optional
  colour jitter / grayscale via `train.augment`.
- Every epoch: validate on full images (batch size 1), then write `last.pth`;
  write `best.pth` whenever val MAE improves. Point
  `paths.checkpoint_dir` at Google Drive or `/kaggle/working` so a disconnect
  costs at most one epoch.
- Logs: TensorBoard scalars under `<checkpoint_dir>/part_X/tb`, a
  `metrics.csv`, a `train.log`, and a val image/GT/prediction triptych every
  `train.viz_every` epochs.
- Mixed precision is used automatically on CUDA (`train.amp`).

### 3. Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/part_B/best.pth --part B --save-predictions
```

Prints MAE/RMSE next to the paper's numbers and writes
`outputs/eval_part_B.json`, a predicted-vs-GT scatter plot, a per-image CSV and
density comparisons for the 10 worst-error images.

### 4. Inference

```bash
python -m src.inference --checkpoint checkpoints/part_B/best.pth \
    --input path/to/image_or_video_or_dir --output outputs/
```

Or from Python:

```python
from src.inference import CrowdCounter

counter = CrowdCounter("checkpoints/part_B/best.pth")
count, density = counter.predict_image(image)          # density.sum() == count
count, heatmap, overlay = counter.render(image, alpha=0.5)
results = counter.predict_video("clip.mp4", frame_stride=5,
                                output_video="out.mp4", output_csv="counts.csv")
```

Images are resized so the longest side is at most `inference.max_side`, padded
to a multiple of 4, and the padding is cropped off the output. Resizing changes
the image area but not the number of people, and the density map is **summed**
rather than resized, so no area correction is needed.

### 5. Demo

```bash
python app/gradio_app.py                      # local, CPU
python app/gradio_app.py --share              # public link, e.g. from Colab
python app/gradio_app.py --enable-video --port 7860
```

Upload an image, get the estimated count, a heatmap overlay with an opacity
slider, and the raw density map. The model selector switches between the Part A
(dense) and Part B (sparse) checkpoints, loading each lazily and caching it. Put
a few ShanghaiTech test images in `app/examples/` and they appear as one-click
examples. If no checkpoint is found, the page says exactly where it looked.

### Notebooks

- [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) — GPU check, Drive
  mount, Kaggle-API download, preprocessing, training with resume, evaluation,
  and the demo with `--share`.
- [notebooks/train_kaggle.ipynb](notebooks/train_kaggle.ipynb) — same flow with
  the dataset attached read-only under `/kaggle/input` and everything written to
  `/kaggle/working`.

---

## Results

Reference numbers from the MCNN paper:

| Split | MAE | RMSE |
|---|---|---|
| Part A | 110.2 | 173.2 |
| Part B | 26.4 | 41.3 |

This reimplementation (fill in after training — `outputs/eval_part_X.json`):

| Split | MAE | RMSE | Epochs | Notes |
|---|---|---|---|---|
| Part A | _TBD_ | _TBD_ | | `density.mode=adaptive` |
| Part B | _TBD_ | _TBD_ | | `density.mode=fixed`, sigma 4 |

Target: Part B MAE < 35 and Part A MAE < 130 — within roughly 20% of the paper.

### Demo screenshots

Add screenshots to `docs/` and link them here once you have trained weights.

---

## Repository layout

```
configs/default.yaml     all paths and hyperparameters
src/config.py            YAML loading + dotted --set overrides
src/utils.py             seeding, device, logging, checkpoint I/O
src/data/
  parse_annotations.py   read head points out of the .mat files
  density.py             truncated, renormalised Gaussian density maps
  preprocess.py          CLI: raw dataset -> .h5 + splits.json
  transforms.py          paired image/density crops, flips, sum-pooling
  dataset.py             CrowdDataset + build_dataloaders
src/models/mcnn.py       the three-column network (133,705 parameters)
src/train.py             training loop, checkpointing, TensorBoard
src/evaluate.py          MAE/RMSE on the test split + reports
src/inference.py         CrowdCounter API and CLI (images + video)
src/visualize.py         heatmaps, overlays, comparison and scatter plots
app/gradio_app.py        the web demo
tests/                   pytest suite, runs on CPU without the real dataset
```

`data/`, `checkpoints/` and `outputs/` are gitignored.

---

## Pitfalls this code guards against

- **Predicting all zeros.** Per-pixel densities are tiny, so a `mean`-reduced
  MSE is nearly minimised by outputting zero everywhere. The default
  `sum_over_batch` reduction keeps the gradient scale usable; if the model still
  collapses, set `train.density_scale=100` (predictions are divided back out
  everywhere downstream, so counts stay in real units).
- **Count drift from resizing.** Targets are reduced by 4x4 sum-pooling, never
  bilinear resizing, and the dataset raises if an image and its density map
  disagree in size.
- **Clipped Gaussians at borders.** Each kernel is renormalised after cropping,
  so a head in the corner still contributes exactly 1.0.
- **Mismatched augmentation.** Crops and flips are applied to the image and the
  density map with the same parameters.
- **Grayscale images in Part A.** Everything is converted to RGB at load time.
- **Odd image sizes at inference.** Inputs are padded to a multiple of 4 and the
  padding is cropped off the prediction before summing.
- **Colab/Kaggle disconnects.** `last.pth` is written every epoch and `--resume`
  restores model, optimiser, scheduler, epoch and best MAE.

## Out of scope

Stronger backbones (CSRNet, DM-Count), other datasets (UCF-QNRF, NWPU-Crowd),
real-time webcam/RTSP counting, ONNX/TorchScript export, ROI counting and
density alerts.

## Citation

```bibtex
@inproceedings{zhang2016single,
  title={Single-Image Crowd Counting via Multi-Column Convolutional Neural Network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={CVPR},
  year={2016}
}
```
