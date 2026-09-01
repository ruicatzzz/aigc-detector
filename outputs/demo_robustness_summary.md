# Robustness Evaluation Summary (accuracy/auc)

| Transform | Accuracy | AUC |
|---|---|---|
| clean | 94.6% | 99.7% |
| jpeg_q90 | 96.7% | 99.8% |
| jpeg_q70 | 95.6% | 99.8% |
| jpeg_q50 | 96.6% | 99.8% |
| jpeg_q30 | 97.1% | 99.8% |
| blur_sigma0.5 | 97.2% | 99.8% |
| blur_sigma1.0 | 96.6% | 99.6% |
| blur_sigma2.0 | 91.4% | 97.2% |
| resize_0.5x | 96.9% | 99.6% |
| resize_0.25x | 91.9% | 97.8% |
| noise_sigma0.02 | 96.7% | 99.5% |
| noise_sigma0.05 | 98.1% | 99.8% |
| noise_sigma0.10 | 95.6% | 100.0% |
| color_jitter | 88.7% | 99.4% |
| center_crop_80 | 42.9% | 38.5% |

## Clean vs transformed summary

| Model | Clean | Mean (transformed) | Worst (transformed) |
|---|---|---|---|
| Accuracy | 94.6% | 91.6% | 42.9% |
| AUC | 99.7% | 95.0% | 38.5% |
