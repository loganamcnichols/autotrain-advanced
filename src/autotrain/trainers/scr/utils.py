import os

import numpy as np
import requests
from sklearn import metrics
from transformers import AutoModelForCausalLM

from autotrain import logger
import torch

TARGET_MODULES = {
    "Salesforce/codegen25-7b-multi": "q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
}

MODEL_CARD = """
---
tags:
- autotrain
- text-classification
widget:
- text: "I love AutoTrain"
datasets:
- {dataset}
---

# Model Trained Using AutoTrain

- Problem type: Text Classification

## Validation Metrics
{validation_metrics}
"""

REGRESSOIN_METRICS = ("eval_loss", "eval_MSE", "eval_MAE", "eval_R2", "eval_Explained_Variance")


def get_target_modules(config):
    if config.target_modules is None:
        return TARGET_MODULES.get(config.model)
    return config.target_modules.split(",")

def merge_adapter(base_model_path, target_model_path, adapter_path):
    logger.info("Loading adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

def _regression_metrics(pred):
    predictions, labels = pred
    results = {
        "MSE": metrics.mean_squared_error(labels, predictions),
        "MAE": metrics.mean_absolute_error(labels, predictions),
        "R2": metrics.r2_score(labels, predictions),
        "Explained_Variance": metrics.explained_variance_score(labels, predictions)
    }
    return results


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


def pause_endpoint(params):
    endpoint_id = os.environ["ENDPOINT_ID"]
    username = endpoint_id.split("/")[0]
    project_name = endpoint_id.split("/")[1]
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{username}/{project_name}/pause"
    headers = {"Authorization": f"Bearer {params.token}"}
    r = requests.post(api_url, headers=headers)
    return r.json()
