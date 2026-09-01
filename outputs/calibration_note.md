# Calibration Note

- Checkpoint: `checkpoints/freq_balanced.pt` (backbone: freq_cnn)
- Calibration set: 6350 images (2763 REAL / 3587 FAKE)
- Fitted temperature **T = 0.922**  (NLL 0.3481 -> 0.3472)

## Operating points (after temperature scaling)

| Threshold | FPR | Recall (FAKE) | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| 0.500 | 9.5% | 81.2% | 91.7% | 0.862 | 85.3% |
| 0.888 | 1.0% | 41.4% | 98.2% | 0.583 | 66.5% |
| 0.690 | 5.0% | 69.7% | 94.7% | 0.803 | 80.7% |
| 0.481 | 10.0% | 82.2% | 91.4% | 0.866 | 85.6% |

Apply with `--calibration outputs/calibration.json`; for a fixed-FPR operating point also pass `--threshold <value from the table>`.
