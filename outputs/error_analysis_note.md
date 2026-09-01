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

**Error profile is highly asymmetric.** FPR 0.6% vs FNR 10.2% — the model
is strongly biased toward "REAL". It almost never falsely accuses a real
photo, but lets roughly 1 in 10 DALL-E images through. For a moderation
setting this is the conservative trade-off (few false accusations, meaningful
AI leakage). All 5 false positives are COCO photos and all 82 false negatives
are DALL-E images, so the entire error mass is "some DALL-E outputs read as
real" — the model is not confused about real photos in general.

**False positives are all borderline; false negatives are not.** The 5 FPs
sit in a 0.02-wide band at P(AI) 0.509–0.530 — there is no *confident* false
positive. Raising the threshold to ~0.54 eliminates all of them at near-zero
FNR cost. The FNs, by contrast, span 0.022–0.20: `dalle__39853.jpg` at 0.022
is a confident miss, not boundary noise. Raising the threshold makes the FNs
worse, so 0.5 is already close to the best operating point here — see
`outputs/calibration_note.md` for the FPR-matched alternatives.

**What the false positives share:** low-detail, low-contrast, smooth real
photos — an empty bathroom wall, a bird over flat grey water, dark low-key
bottle shots, near-empty blue sky. Several have low pixel variance (stddev
19–41). With little high-frequency structure, the spectrum looks "clean" the
way the model associates with generated images. The FPs correlate with *scene
simplicity*, not with any generation artifact.

**What the false negatives share:** detailed, texture-rich, photorealistic
DALL-E outputs (512×512, stddev 58–89) — a sharp portrait of a man in a suit,
a beach scene, a tree in a field. Some carry obvious semantic tells (garbled
text "BOPG" on `dalle__27118.jpg`, warped fence slats, mushy foliage) that a
person spots instantly but `freq_cnn` cannot, because its input is a 64×64
Fourier-magnitude image with all spatial and semantic content discarded. This
is the core structural limitation of the frequency-only model and the main
argument for the spatial + frequency ensemble: the two fail on disjoint sets
(smooth real photos vs. text/geometry-heavy fakes).

**The consistently-hard set is small and concentrated at the boundary.**
10 of the 12 are DALL-E, all with clean P(AI) 0.40–0.54; two COCO images
(`img109177`, `img153171`) appear in both the worst-FP list and the hard
list. The top offender `dalle__21007.jpg` is actually *correct* on clean
(0.540) but flips under 13/15 transforms — a fragile correct call, not a hard
error. So the fragility is ~10–15 images sitting near P≈0.5 that any
perturbation tips over, matching the pattern noted for the earlier
checkpoints, rather than broad instability.

**Caveats on this benchmark specifically.** (1) The checkpoint is
uncalibrated (temperature 1.000); FPs jammed at 0.51–0.53 and FNs at
0.10–0.20 show the model is barely separating the classes on this
out-of-distribution set (it never saw COCO or DALL-E in training).
Calibration would spread the scores but not widen the margin. (2) COCO images
here are ~200×200 and DALL-E ~512×512; `freq_cnn` resizes every spectrum to
64×64 so it cannot read raw resolution, but upscaled-200px vs native-512px
images leave different spectral signatures, so part of the (already weak)
signal may be that confound rather than a true generator fingerprint — don't
over-read the headline AUC.

