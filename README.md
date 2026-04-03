# Drone vs Bird Classifier
A YOLOv8-based image classification model that distinguishes drones from birds.

## Overview
This project trains a YOLOv8 classification model on a dataset of drone and bird images. All training parameters are controlled through a single `config.yaml` file — no need to touch the code.

## Project Structure
```
├── config.yaml       # All settings (model, epochs, augmentation, etc.)
├── train.py          # Splits dataset and trains the model
├── classify.py       # Classifies a single image or folder of images
├── docs/             # Images for README
└── README.md
```

## Dataset
[Drone vs Bird – Kaggle](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird)  
~4000 images split into two classes: `drone` and `bird`.

### Sample Images
![Dataset samples](docs/dataset_samples.jpg)

## Setup
```bash
pip install ultralytics pyyaml
```

Download the dataset and place it in the following structure:
```
dataset/
├── drone/
└── bird/
```

## Training
Adjust settings in `config.yaml`, then run:
```bash
python train.py
```
The script automatically splits the dataset into train / val / test sets and saves the best model to `runs/model/weights/best.pt`.

## Classification
```bash
# Single image
python classify.py image.jpg

# Entire folder
python classify.py path/to/folder/
```

Example output:
```
image.jpg
  Result:  DRONE (97.3%)
  Second:  bird (2.7%)
```

## Configuration
| Parameter | Default | Description |
|---|---|---|
| `model` | `yolov8n` | Model size: n / s / m / l / x |
| `epochs` | `30` | Number of training epochs |
| `batch_size` | `32` | Batch size |
| `image_size` | `224` | Input image size in pixels |
| `train` | `0.7` | Train split ratio |
| `val` | `0.2` | Validation split ratio |
| `test` | `0.1` | Test split ratio |

## Results

### Training Curves
![Training curves](docs/results.png)

Training loss drops consistently over 10 epochs. Top-1 accuracy reaches **~99%** on the validation set, with top-5 accuracy at a perfect **1.0** throughout (expected for a 2-class problem).

### Confusion Matrix
![Confusion matrix](docs/confusion_matrix_normalized.png)
![Confusion matrix (counts)](docs/confusion_matrix.png)

Only **3 misclassifications per class** out of 241 birds and 374 drones in the test set — **99% accuracy** on both classes.

### Prediction Examples
![Prediction examples](docs/val_batch_pred.jpg)
![Prediction examples 2](docs/val_batch_pred2.jpg)

Augmented validation batch with predicted labels. The model correctly identifies both close-up and distant objects across varied lighting and backgrounds.

## Tech Stack
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- Python 3.10+
- Google Colab (T4 GPU)
