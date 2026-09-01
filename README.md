# AIGC Image Detector

Prototype for detecting AI-generated images with robustness to common
post-processing transformations (JPEG compression, blur, resize, noise,
color jitter, center crop).

## Project Overview

This project trains a lightweight CNN (`SmallCNN`) to classify images as
real or AI-generated, with a focus on robustness under realistic
post-processing rather than just clean-image accuracy. The core approach:
train-time augmentation using the same transform family the model is
evaluated against, so the network learns features that survive
compression, blurring, downsampling, and cropping rather than relying on
brittle high-frequency artifacts that vanish under light processing.

The model is trained on a combination of CIFAKE (32x32 real/synthetic
image pairs), a streamed subset of SID_Set (higher-resolution, more
diverse generators), and a streamed subset of WildFake (many generator
families: ADM, DDIM, DDPM, Imagen, VQDM, Midjourney, personalised SD,
GANs, plus GAN/diffusion real-image sources), and evaluated for both raw
accuracy and AUC across every transform in the problem statement's table.
The WildFake subset deliberately **excludes** the two archives that make
up the organisers' demo benchmark (COCO val2017 and "DALL-E Advanced") so
there is no train/validation overlap. A second backbone
option (configurable, higher-capacity) was also explored in collaboration
with a teammate and cross-evaluated against this model using a shared
evaluation harness — see `outputs/` for the resulting comparison tables.

## Development Tools

- Google Colab (GPU training)
- VS Code (local development, notebook editing)
- GitHub (version control, branch-per-person workflow)

## Models / APIs

- `SmallCNN` — custom lightweight CNN (6 conv layers, ~128 channel max,
  BatchNorm + Dropout), trained from scratch, well under the 2B parameter
  cap. Operates on 32x32 inputs.
- `efficientnet_b0` — optional ImageNet-pretrained backbone (via `timm`),
  fine-tuned as a binary classifier at 224x224.
- `freq_cnn` — the `SmallCNN` architecture applied to a 3-channel
  log-magnitude Fourier spectrum of the image (catches up-sampling
  artifacts an RGB model misses).
- Backbone is selected with `--backbone {small_cnn,efficientnet_b0,freq_cnn}`;
  `src/ensemble.py` averages several checkpoints (e.g. a spatial model +
  `freq_cnn`).

## Libraries & Frameworks

- PyTorch / torchvision — model, training loop, data loading
- `datasets` (Hugging Face) — streaming partial downloads of SID_Set
- `requests` + `stream-unzip` — streaming a bounded subset of WildFake
  from ModelScope without downloading the full ~1.3 TB
- `kagglehub` — CIFAKE download
- scikit-learn — AUC / precision / recall / F1 metrics
- Pillow, NumPy — image transforms and augmentation
- matplotlib — robustness summary charts

## Datasets & Assets

- **CIFAKE** (Kaggle) — 100,000 train / 20,000 test images, 32x32,
  balanced REAL/FAKE.
- **SID_Set** (Hugging Face, `saberzl/SID_Set`) — streamed subset (~10,000
  training images, ~2,000 held-out test images), full resolution,
  3-way labeled (real / full-synthetic / tampered — tampered images are
  treated as FAKE for this task). Downloaded via `datasets.load_dataset`
  in streaming mode to avoid pulling the full 140GB dataset.
- **WildFake** (ModelScope, `hy2628982280/WildFake`) — streamed subset
  (~9k training images, ~2k held-out test images) drawn from many
  generator families and real-image sources. WildFake ships as a handful
  of 6–50 GB zip archives; `src/download_wildfake.py` opens an HTTP
  stream per archive, decodes it on the fly with `stream-unzip`, and
  stops after the first N images (a few hundred MB per archive instead of
  tens of GB). The `Images/Real/coco.zip` and
  `Images/Diffusion_based/DALLE.zip` archives are hard-blocked in that
  script because they contain the organisers' demo benchmark
  (**COCO val2017** + **DALL·E Advanced**) — that data is never
  downloaded and never seen in training.
- **WildFake demo/validation subset** (COCO val2017 + DALL·E Advanced,
  4998 + 8843 images, per the problem statement) — reference benchmark
  only, obtained separately from the organisers, **not** used in
  training.

## Setup & Installation

```bash
git clone <repo-url>
cd aigc-detector
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Datasets are not committed to this repo (`data/` is gitignored — each
teammate downloads their own local copy).

**CIFAKE:**
```bash
# via kagglehub, or download manually from:
# https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
```
Expected layout: `data/cifake/train/{REAL,FAKE}`, `data/cifake/test/{REAL,FAKE}`

**SID_Set subset** (streamed, no full download):
```bash
python -m src.download_sid
# tune volume:  --n_train 4000 --n_test 1000
```
Streams `saberzl/SID_Set` and writes the first `--n_train` examples as the
training subset and the next `--n_test` as a non-overlapping holdout
(SID_Set labels: 0 = real → REAL; 1 = fully synthetic, 2 = tampered →
FAKE). Note: SID_Set's streaming loader fetches a full shard before it
yields the first example, so the first output line can take a few minutes.
Expected layout: `data/sid_subset/{REAL,FAKE}` (training),
`data/sid_test_holdout/{REAL,FAKE}` (held-out, non-overlapping test set)

**WildFake subset** (streamed from ModelScope, no full download):
```bash
python -m src.download_wildfake
# tune volume:  --fake_per_source 1500 --real_per_source 1500
# single sources: --only afhq ddim gan
```
Streams each selected WildFake archive over HTTP, unzips it on the fly,
and stops after the first N images per source. `Images/Real/coco.zip` and
`Images/Diffusion_based/DALLE.zip` are hard-blocked (reserved demo
benchmark) and never fetched. Expected layout:
`data/wildfake_subset/{REAL,FAKE}` (training),
`data/wildfake_test_holdout/{REAL,FAKE}` (held-out, non-overlapping —
drawn from the same stream, strictly after the training slice).

## Reproducing Results

Dataset dirs after downloading (see **Data** above): `data/CIFAKE/train`,
`data/CIFAKE/test`, `data/sid_subset`, `data/sid_test_holdout`,
`data/wildfake_subset`, `data/wildfake_test_holdout`. The demo/validation
benchmark (`data/demo_benchmark`) is downloaded separately with
`python -m src.download_demo_benchmark` and is used for evaluation only.

**1. Train.** One or more `--data_dir` are merged. `--backbone` picks the
architecture; `--cap_per_source` / `--balance_classes` keep one dataset
from dominating; `--no_augment` gives the non-augmented A/B baseline.
```bash
python -m src.train --data_dir data/CIFAKE/train data/sid_subset data/wildfake_subset \
  --backbone freq_cnn --cap_per_source 15000 --balance_classes --epochs 10 --out checkpoints/freq_balanced.pt
```
Saves the checkpoint from the best validation-accuracy epoch. Train-time
augmentation (stacked JPEG / blur / resize / noise / colour-jitter / crop
/ rotation / flip) is applied to the training split only; validation stays
clean.

**2. Run inference** (image directory → JSON of `{image_path, pred}`):
```bash
python -m src.infer --checkpoint checkpoints/freq_balanced.pt \
  --input_dir data/demo_benchmark --output_json outputs/demo_preds.json
```

**3. Robustness evaluation** — accuracy and AUC across the full transform grid:
```bash
python -m src.eval_robustness --checkpoint checkpoints/freq_balanced.pt --test_dir data/demo_benchmark --out_csv outputs/demo_robustness.csv
python -m src.eval_auc        --checkpoint checkpoints/freq_balanced.pt --test_dir data/demo_benchmark --out_csv outputs/demo_auc.csv
```

**4. Summary table + chart** (clean vs mean/worst transformed):
```bash
python -m src.robustness_summary \
  --csv outputs/demo_robustness.csv:Accuracy outputs/demo_auc.csv:AUC \
  --out_md outputs/demo_robustness_summary.md --out_png outputs/demo_robustness_chart.png
```

**5. Calibration** (temperature + fixed-FPR operating points):
```bash
python -m src.calibrate --checkpoint checkpoints/freq_balanced.pt --val_dir data/sid_test_holdout data/wildfake_test_holdout
```

**6. Cross-dataset generalisation probe** (per-dataset AUC, shortcut spread):
```bash
python -m src.eval_crossdataset --checkpoint checkpoints/freq_balanced.pt \
  --dataset SID=data/sid_test_holdout WildFake=data/wildfake_test_holdout DemoBenchmark=data/demo_benchmark
```

**7. Error analysis** (worst FP/FN saved as images + written note):
```bash
python -m src.error_analysis --checkpoint checkpoints/freq_balanced.pt --test_dir data/demo_benchmark
```

## Results

See `outputs/`:
- `small_demo_robustness.csv` / `small_demo_auc.csv` — per-transform accuracy / AUC on the demo benchmark
- `demo_robustness_summary.md` / `demo_robustness_chart.png` — combined table + chart, with the clean / mean-transformed / worst-transformed summary
- `rob_baseline.csv` vs `rob_augmented.csv` — augmented vs non-augmented A/B
- `calibration_note.md` — temperature + operating points at 1% / 5% / 10% FPR
- `crossdataset.csv` — per-dataset AUC (generalisation gap)
- `error_analysis_note.md` + `error_analysis_examples/` — representative false positives / negatives

Headline finding: clean AUC ~0.98, degrading to ~0.93-0.94 under the
heaviest blur (sigma=2.0) and downsampling (0.25x) conditions — the
smallest degradation of any transform pair, but still the model's two
weakest points. Accuracy degrades more sharply than AUC under these same
conditions, suggesting a threshold-calibration gap rather than a pure
loss of discriminative signal (see Limitations below).

## Limitations & Future Work

- **Fixed 0.5 decision threshold.** AUC holds up better than accuracy
  under heavy blur/resize, suggesting the model still ranks fakes above
  reals reasonably well even where hard classification suffers — a
  transform-aware or calibrated threshold could likely recover some of
  this gap without retraining.
- **Residual failures concentrate in a small, consistent subset of hard
  examples** (same handful of images fail across multiple transforms)
  rather than being uniformly distributed — worth further investigation
  into what makes these specific images harder (see
  `error_analysis_note.md`).
- **Possible dataset-specific shortcut learning.** CIFAKE is a
  known-easy benchmark; strong performance there doesn't guarantee
  generalization to generator families never seen in training. This
  wasn't tested against fully out-of-distribution generators due to time
  constraints — a natural next step.
- **32x32 input resolution** discards fine detail that a higher-resolution
  backbone might use more effectively, particularly on SID_Set's
  full-resolution images. A pretrained backbone (CLIP/DINOv2) at higher
  resolution was explored as an alternative approach by a teammate; see
  cross-evaluation results for a head-to-head comparison.
- **WildFake is used for training via a streamed subset only.** The full
  dataset is ~1.3 TB, so `src/download_wildfake.py` pulls just the first
  N images per generator archive. This biases the WildFake contribution
  toward whatever ordering each archive happens to use (often a single
  sub-shard), so the subset is diverse across generator *families* but
  not necessarily representative *within* a family. The reserved demo
  benchmark archives (COCO val2017, DALL·E Advanced) are excluded from
  the download entirely.
- **Given more time:** explainability (Grad-CAM) to confirm whether the
  model relies on semantically meaningful cues vs. low-level artifacts;
  ensembling the two backbone approaches, since they may fail on
  different examples; broader hyperparameter search (dropout, weight
  decay, augmentation severity weighting) beyond the manual tuning done
  here.

## Team Contributions

- **Daphne** — Infrastructure and model training: dataset streaming/download
  pipeline, `SmallCNN` architecture, training loop and augmentation,
  compute setup.
- **Emily, Vera** — Pretrained backbone testing: EfficientNet-B0 / higher-
  resolution backbone experiments and cross-evaluation.
- **Anni** — Fine-tuning, calibration and error analysis: robustness
  evaluation harness, temperature calibration and fixed-FPR operating
  points, cross-dataset probe, error-analysis note, `freq_cnn` +
  ensemble, demo benchmark.

## Reproducibility Notes

- Checkpoints and raw datasets are gitignored; only small result files
  (`outputs/`) are committed.
- Random seeds are fixed for the train/val split (`seed=42` in
  `get_dataloaders`) but not for augmentation itself, so exact training
  curves may vary slightly run-to-run; validation accuracy trends should
  be consistent.
- Training and inference auto-select the compute device via
  `pick_device()` in `src/model.py`: CUDA if present, then Apple Silicon
  MPS, then CPU. This is a no-op on Colab (CUDA) but lets the pipeline run
  on the GPU of an M-series Mac locally.