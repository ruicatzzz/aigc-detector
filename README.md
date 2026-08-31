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
- A second, configurable-backbone model (teammate's implementation) was
  cross-evaluated against this one using a shared evaluation harness —
  see `cross_eval/` and `outputs/robustness_table_via_her_harness.csv`.

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

**1. Train** (supports one or more merged datasets):
```bash
python -m src.train --data_dir data/cifake/train data/sid_subset data/wildfake_subset --epochs 10 --out checkpoints/cnn_merged.pt
```
Saves the checkpoint from the best validation-accuracy epoch, not simply
the final epoch. Training uses on-the-fly augmentation (JPEG, blur,
resize, noise, color jitter, crop) on the training split only; validation
stays clean.

**2. Run inference** (image directory → JSON):
```bash
python -m src.infer --input_dir data/cifake/test data/sid_test_holdout data/wildfake_test_holdout --output_json outputs/preds.json --checkpoint checkpoints/cnn_merged.pt
```

**3. Robustness evaluation** (accuracy per transform):
```bash
python -m src.eval_robustness --checkpoint checkpoints/cnn_merged.pt --test_dir data/cifake/test data/sid_test_holdout data/wildfake_test_holdout --out_csv outputs/robustness_table_merged.csv
```

**4. AUC evaluation** (threshold-independent ranking quality per transform):
```bash
python -m src.eval_auc --checkpoint checkpoints/cnn_merged.pt --test_dir data/cifake/test data/sid_test_holdout data/wildfake_test_holdout --out_csv outputs/auc_table.csv
```

**5. Robustness summary table + chart:**
```bash
python -m src.make_robustness_summary --csv outputs/robustness_table_merged.csv:CIFAKE+SID_Set
```

**6. Error analysis** (worst false positives / false negatives, saved with example images):
```bash
python -m src.error_analysis --test_dir data/cifake/test data/sid_test_holdout --checkpoint checkpoints/cnn_merged.pt
```

**7. Cross-evaluation against teammate's harness** (see `cross_eval/`):
```bash
cd cross_eval
python build_test_transforms.py --test_dir ../data/cifake/test --out_dir ../data/test_transformed
python evaluate.py --checkpoint ../checkpoints/cnn_merged.pt --backbone small_cnn --data_dir ../data/cifake --transformed_dir ../data/test_transformed --out ../outputs/robustness_table_via_her_harness.csv
```

## Results

See `outputs/`:
- `robustness_table_merged.csv` / `robustness_summary.md` / `robustness_chart.png` — accuracy per transform
- `auc_table.csv` — AUC per transform
- `error_analysis_note.md` + `error_analysis_examples/` — representative false positives/negatives
- `robustness_table_via_her_harness.csv` — cross-evaluation against a second backbone

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

<!-- Name — role — key contributions -->

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