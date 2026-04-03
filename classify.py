import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = Path(cfg["output_folder"]) / "model" / "weights" / "best.pt"

if len(sys.argv) < 2:
    print("Usage: python classify.py path/to/image.jpg")
    print("       python classify.py path/to/folder/")
    sys.exit(1)

path = Path(sys.argv[1])

if not path.exists():
    print(f"Not found: {path}")
    sys.exit(1)

if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}")
    print("Train the model first: python train.py")
    sys.exit(1)

model = YOLO(str(MODEL_PATH))

if path.is_dir():
    images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
    print(f"Found {len(images)} images.\n")
else:
    images = [path]

for image in images:
    results = model(str(image), verbose=False)
    probs = results[0].probs

    top_class = results[0].names[probs.top1]
    confidence = probs.top1conf.item() * 100

    second_class = results[0].names[probs.top5[1]]
    second_conf = probs.top5conf[1].item() * 100

    print(f"{image.name}")
    print(f"  Result:  {top_class.upper()} ({confidence:.1f}%)")
    print(f"  Second:  {second_class} ({second_conf:.1f}%)")
    print()