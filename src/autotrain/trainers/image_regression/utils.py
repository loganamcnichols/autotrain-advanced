import albumentations as A
import numpy as np
from sklearn import metrics

from autotrain.trainers.image_regression.dataset import ImageRegressionDataset


REGRESSOIN_METRICS = ("eval_loss", "eval_MSE", "eval_MAE", "eval_R2", "eval_Explained_Variance")

MODEL_CARD = """
---
tags:
- autotrain
- image-classification
widget:
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg
  example_title: Tiger
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg
  example_title: Teapot
- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg
  example_title: Palace
datasets:
- {dataset}
---

# Model Trained Using AutoTrain

- Problem type: Image Regression

## Validation Metricsg
{validation_metrics}
"""


def _regression_metrics(pred):
    predictions, labels = pred
    results = {
        "MSE": metrics.mean_squared_error(labels, predictions),
        "MAE": metrics.mean_absolute_error(labels, predictions),
        "R2": metrics.r2_score(labels, predictions),
        "Explained_Variance": metrics.explained_variance_score(labels, predictions)
    }
    return results


def process_data(train_data, valid_data, image_processor, config):
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    try:
        height, width = size
    except TypeError:
        height = size
        width = size

    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )
    train_data = ImageRegressionDataset(train_data, train_transforms, config)
    if valid_data is not None:
        valid_data = ImageRegressionDataset(valid_data, val_transforms, config)
        return train_data, valid_data
    return train_data, None


def create_model_card(config, trainer, num_classes):
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = REGRESSOIN_METRICS
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    model_card = MODEL_CARD.format(
        dataset=config.data_path,
        validation_metrics=eval_scores,
    )
    return model_card
