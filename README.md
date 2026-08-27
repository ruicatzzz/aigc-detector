# AIGC Image Detector

Prototype for detecting AI-generated images with robustness to common
post-processing transformations (JPEG compression, blur, resize, noise,
color jitter, center crop).

## Project Overview
<!-- Person D, Day 3: 2-3 sentences — what the solution does, the core
technical approach (backbone + robustness training), and the headline
robustness result once you have it. -->

## Setup & Installation

```bash
git clone <repo-url>
cd aigc-detector
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Datasets are not committed to this repo (`data/` is gitignored). Download:
- SID_Set: https://huggingface.co/datasets/saberzl/SID_Set
- CIFAKE: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- WildFake: https://modelscope.cn/datasets/hy2628982280/WildFake/summary

<!-- Person A: add exact folder layout expected under data/ once dataset.py is finalized -->

## Reproducing Results

<!-- Person B/C, Day 2-3: fill in the exact commands, e.g.
python -m src.train --config configs/baseline.yaml
python -m src.build_test_transforms --input data/test_clean --output data/test_transformed
python -m src.evaluate --checkpoint checkpoints/model.pt --test_dir data/test_transformed
-->

## Running Inference

```bash
python -m src.infer --input_dir <path_to_images> --output_json outputs/preds.json --checkpoint checkpoints/model.pt
```

Outputs a JSON file with `image_path` and `pred` (confidence the image is
AI-generated, 0-1) for every image in the input directory.

## Results

<!-- Person C, Day 2-3: paste/link the robustness table (clean vs each
transform/severity) and 2-3 representative error analysis examples here,
or link to outputs/robustness_table.csv -->

## Limitations & Future Work

<!-- Person D, Day 3: pull from Person C's error analysis + team discussion -->

## Team Contributions

<!-- Name — role — key contributions -->

## Development Tools & Stack

- Development: <!-- e.g. VS Code, Colab -->
- Models/APIs: <!-- e.g. CLIP ViT-B/16 backbone -->
- Libraries: PyTorch, torchvision, open-clip-torch, opencv-python, scikit-learn, grad-cam
- Datasets: SID_Set, CIFAKE, WildFake (subset)
