import yaml
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATASET = Path(cfg["dataset_folder"])
OUTPUT = Path(cfg["output_folder"])
MODEL = cfg["model"] + "-cls.pt"
EPOCHS = cfg["epochs"]
BATCH = cfg["batch_size"]
IMAGE_SIZE = cfg["image_size"]
SPLIT_TRAIN = cfg["train"]
SPLIT_VAL = cfg["val"]

for folder in ["train", "val", "test"]:
    split_path = DATASET / folder
    if split_path.exists():
        shutil.rmtree(split_path)

classes = [f.name for f in DATASET.iterdir() if f.is_dir()]
print(f"Classes found: {classes}")

random.seed(42)

for cls in classes:
    images = list((DATASET / cls).glob("*.*"))
    random.shuffle(images)

    n_train = int(len(images) * SPLIT_TRAIN)
    n_val = int(len(images) * SPLIT_VAL)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, files in splits.items():
        dest = DATASET / split_name / cls
        dest.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy(file, dest / file.name)

    print(f"  {cls}: {n_train} train | {n_val} val | {len(images) - n_train - n_val} test")

aug = cfg.get("augmentation", {})

model = YOLO(MODEL)

model.train(
    data=str(DATASET),
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMAGE_SIZE,
    project=str(OUTPUT),
    name="model",
    exist_ok=True,
    flipud=aug.get("flipud", 0.0),
    fliplr=aug.get("fliplr", 0.5),
    degrees=aug.get("degrees", 0.0),
    scale=aug.get("scale", 0.5),
    hsv_h=aug.get("hsv_h", 0.015),
    hsv_s=aug.get("hsv_s", 0.7),
    hsv_v=aug.get("hsv_v", 0.4),
    erasing=aug.get("erasing", 0.0),
)

print(f"\nTraining complete. Best model: {OUTPUT}/model/weights/best.pt")