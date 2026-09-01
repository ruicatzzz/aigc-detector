# Error Analysis Note

- Checkpoint: `checkpoints/freq_balanced.pt` (backbone: freq_cnn, temperature: 1.000)
- Test set: 1600 images (800 REAL / 800 FAKE) from demo_benchmark
- Decision threshold: 0.5
- **False-positive rate** (REAL flagged as AI): 0.6%  (5/800)
- **False-negative rate** (AI passed as REAL): 10.2%  (82/800)

## Worst false positives (REAL, highest P(AI))

| rank | dataset | P(AI) | file |
|---|---|---|---|
| 0 | demo_benchmark | 0.530 | `coco__img153171.jpg` |
| 1 | demo_benchmark | 0.515 | `coco__img157028.jpg` |
| 2 | demo_benchmark | 0.511 | `coco__img050750.jpg` |
| 3 | demo_benchmark | 0.510 | `coco__img109177.jpg` |
| 4 | demo_benchmark | 0.509 | `coco__img092040.jpg` |

## Worst false negatives (AI, lowest P(AI))

| rank | dataset | P(AI) | file |
|---|---|---|---|
| 0 | demo_benchmark | 0.022 | `dalle__39853.jpg` |
| 1 | demo_benchmark | 0.104 | `dalle__1899.jpg` |
| 2 | demo_benchmark | 0.112 | `dalle__21139.jpg` |
| 3 | demo_benchmark | 0.115 | `dalle__14180.jpg` |
| 4 | demo_benchmark | 0.117 | `dalle__5104.jpg` |
| 5 | demo_benchmark | 0.118 | `dalle__36557.jpg` |
| 6 | demo_benchmark | 0.137 | `dalle__27118.jpg` |
| 7 | demo_benchmark | 0.161 | `dalle__3233.jpg` |
| 8 | demo_benchmark | 0.163 | `dalle__3376.jpg` |
| 9 | demo_benchmark | 0.183 | `dalle__24917.jpg` |
| 10 | demo_benchmark | 0.190 | `dalle__31929.jpg` |
| 11 | demo_benchmark | 0.200 | `dalle__24627.jpg` |

## Consistently hard images (misclassified under the most of 15 transforms)

| dataset | true | clean P(AI) | # transforms wrong | file |
|---|---|---|---|---|
| demo_benchmark | AI | 0.540 | 13/15 | `dalle__21007.jpg` |
| demo_benchmark | AI | 0.408 | 11/15 | `dalle__12212.jpg` |
| demo_benchmark | REAL | 0.510 | 10/15 | `coco__img109177.jpg` |
| demo_benchmark | AI | 0.498 | 8/15 | `dalle__35380.jpg` |
| demo_benchmark | AI | 0.451 | 8/15 | `dalle__21068.jpg` |
| demo_benchmark | AI | 0.425 | 8/15 | `dalle__35653.jpg` |
| demo_benchmark | AI | 0.415 | 8/15 | `dalle__29428.jpg` |
| demo_benchmark | AI | 0.408 | 8/15 | `dalle__32036.jpg` |
| demo_benchmark | AI | 0.396 | 8/15 | `dalle__3235.jpg` |
| demo_benchmark | AI | 0.496 | 7/15 | `dalle__32708.jpg` |
| demo_benchmark | REAL | 0.530 | 7/15 | `coco__img153171.jpg` |
| demo_benchmark | AI | 0.468 | 7/15 | `dalle__7627.jpg` |

## Trade-offs & observations

<!-- Fill in after eyeballing outputs/error_analysis_examples/: -->
- Do the false positives share a trait (low resolution? heavy texture? a particular source dataset)?
- Do the false negatives come from one generator family?
- Are the consistently-hard images the same few across transforms (a small fragile subset) or spread out?
- Threshold trade-off: raising it cuts false positives but raises the false-negative rate — see `outputs/calibration_note.md` for operating points.

