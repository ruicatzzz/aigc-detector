from pathlib import Path
from PIL import Image
from src.model import load_model

def evaluate_accuracy(test_dir, checkpoint_path):
    model = load_model(checkpoint_path)
    correct, total = 0, 0
    for cls, true_label in [("REAL", 0.0), ("FAKE", 1.0)]:
        cls_dir = Path(test_dir) / cls
        for img_path in cls_dir.glob("*"):
            with Image.open(img_path) as img:
                pred = model.predict(img.convert("RGB"))
            predicted_label = 1.0 if pred > 0.5 else 0.0
            correct += int(predicted_label == true_label)
            total += 1
    print(f"Test accuracy: {correct/total:.4f}  ({correct}/{total})")

if __name__ == "__main__":
    evaluate_accuracy("data/cifake/test", "checkpoints/cnn_cifake.pt")