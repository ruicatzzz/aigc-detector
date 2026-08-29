# SID_Set Subset — Baseline Testing Run

**Status: preliminary / small-subset test run.** This is NOT the final
training run — it uses a small SID_Set subset to validate that the full
pipeline (data → train → augment → evaluate → error analysis) works
end-to-end before scaling up to the full/combined dataset (SID_Set +
CIFAKE + WildFake).

## Dataset used

- Source: [saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) (Hugging Face)
- Subset size: **500 images/class train, 200 images/class test** (n=200 per
  condition in the results below — far smaller than the full 240K-image
  dataset)
- Label mapping: `0 (real) -> REAL`, `1 (full_synthetic) -> FAKE`,
  `2 (tampered) -> FAKE`

## Commands run

```bash
cd src

# 1. Download SID_Set subset
python download_sid_set.py --out_dir ../data_sid/train --n_per_class 500 --split train
python download_sid_set.py --out_dir ../data_sid/test --n_per_class 200 --split validation

# 2. Build the fixed transformed test set (deterministic, for evaluation)
python build_test_transforms.py --test_dir ../data_sid/test --out_dir ../data_sid/test_transformed

# 3. Train baseline (no robustness augmentation)
python train.py --data_dir ../data_sid --epochs 3 --run_name resnet50_baseline

# 4. Train augmented (with robustness augmentation)
python train.py --data_dir ../data_sid --epochs 3 --augment --run_name resnet50_augmented

# 5. Evaluate both checkpoints against clean + all 14 transform conditions
python evaluate.py --checkpoint ../checkpoints/resnet50_baseline.pt \
                    --data_dir ../data_sid --transformed_dir ../data_sid/test_transformed \
                    --out ../outputs/baseline_results.csv

python evaluate.py --checkpoint ../checkpoints/resnet50_augmented.pt \
                    --data_dir ../data_sid --transformed_dir ../data_sid/test_transformed \
                    --out ../outputs/augmented_results.csv

# 6. Grad-CAM error analysis on the augmented model
python gradcam.py --checkpoint ../checkpoints/resnet50_augmented.pt \
                   --predictions ../outputs/augmented_results_predictions.json \
                   --condition clean --num_examples 3
```
# ALTERNATIVELY: Google Colab (make a copy of this yourself)
https://colab.research.google.com/drive/1XmjC7DvNEETqJwUwXp_DTxhzEF-NbxvX?usp=sharing

## Results: Baseline vs. Augmented

| Condition    | Baseline Acc | Augmented Acc | Δ      |
|--------------|-------------:|---------------:|-------:|
| clean        | 0.680        | 0.710           | +0.030 |
| blur_0.5     | 0.695        | 0.715           | +0.020 |
| blur_1.0     | 0.675        | 0.720           | +0.045 |
| blur_2.0     | 0.675        | 0.725           | +0.050 |
| center_crop  | 0.715        | 0.710           | -0.005 |
| color_jitter | 0.695        | 0.695           |  0.000 |
| jpeg_30      | 0.695        | 0.690           | -0.005 |
| jpeg_50      | 0.680        | 0.705           | +0.025 |
| jpeg_70      | 0.665        | 0.715           | +0.050 |
| jpeg_90      | 0.690        | 0.720           | +0.030 |
| noise_0.02   | 0.690        | 0.715           | +0.025 |
| noise_0.05   | 0.700        | 0.690           | -0.010 |
| noise_0.10   | 0.665        | 0.685           | +0.020 |
| resize_0.25  | 0.650        | 0.705           | +0.055 |
| resize_0.5   | 0.675        | 0.720           | +0.045 |

**Average accuracy:** baseline ≈ 68.3% vs. augmented ≈ 70.8% (**+2.5pp**)

**Spread across conditions** (smaller = more stable under transforms):
baseline 0.650–0.715 (6.5pp spread) vs. augmented 0.685–0.725 (4.0pp spread)

**Takeaway:** augmentation improved accuracy on 11/15 conditions, with the
largest gains on blur (up to +5pp) and resize (up to +5.5pp) — the
categories most directly targeted by training-time augmentation. The
augmented model is both higher on average and more consistent across
conditions (tighter spread), which is the core robustness claim for this
track.

## Error analysis (Grad-CAM)

Generated examples saved to `outputs/error_analysis_examples/` — one
correct prediction, one false positive (REAL flagged as FAKE), and one
false negative (FAKE flagged as REAL), each with a Grad-CAM heatmap
overlay showing which image regions drove the prediction.

_Fill in after reviewing the images:_
- Correct prediction — model appears to focus on: _______
- False positive — model appears to focus on: _______
- False negative — model appears to focus on: _______
- Weakest transform conditions (jpeg_70, noise_0.05/0.10) suggest the
  model may be sensitive to: _______

## Known limitations of this run

- **Small subset**: 500 train / 200 test images per class is far below
  what's needed for a strong final result — this run validates the
  pipeline works, not final model quality.
- **Single dataset source**: SID_Set only. Combining with CIFAKE (GAN
  images) and WildFake via `merge_datasets.py` should improve
  generalization — not yet done as of this run.
- **Few epochs**: only 3 epochs per run, chosen for fast iteration while
  debugging the pipeline, not for convergence.
- **No confidence calibration verified**: softmax scores are used directly
  as confidence; they have not been checked for calibration (see
  `docs/architecture.md` for more on this).

## Next steps

1. Combine SID_Set + CIFAKE (+ WildFake) via `merge_datasets.py` for the
   final training dataset.
2. Increase `--n_per_class` and `--epochs` for the final run — Colab GPU
   recommended for iteration speed at this scale.
3. Re-run this same evaluation + Grad-CAM flow on the combined dataset and
   move final results into the main `README.md` results table.
4. Fill in the error analysis interpretation above once images are
   reviewed.

