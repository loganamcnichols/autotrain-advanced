import os

import torch
from datasets import load_dataset
from loguru import logger
from sklearn import metrics
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from autotrain import utils
from autotrain.params import TextSingleColumnRegressionParams


TEXT_COLUMN = "autotrain_text"
LABEL_COLUMN = "autotrain_label"
FP32_MODELS = ("t5", "mt5", "pegasus", "longt5", "bigbird_pegasus")

MODEL_CARD = """
---
tags:
- autotrain
- text-classification
language:
- {language}
widget:
- text: "I love AutoTrain"
datasets:
- {dataset}
co2_eq_emissions:
  emissions: {co2}
---

# Model Trained Using AutoTrain

- Problem type: Text Classification
- CO2 Emissions (in grams): {co2:.4f}

## Validation Metrics
{validation_metrics}
"""


class Dataset:
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item][TEXT_COLUMN])
        target = self.data[item][LABEL_COLUMN]
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            token_type_ids = None

        if token_type_ids is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "labels": torch.tensor(target, dtype=torch.float),
            }
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }


def _regression_metrics(pred):
    predictions, labels = pred
    results = {
        "MSE": metrics.mean_squared_error(labels, predictions),
        "MAE": metrics.mean_absolute_error(labels, predictions),
        "R2": metrics.r2_score(labels, predictions),
        "Explained Variance": metrics.explained_variance_score(labels, predictions)
    }
    return results


@utils.job_watcher
def train(co2_tracker, payload, huggingface_token, model_path):
    model_repo = utils.create_repo(
        project_name=payload["proj_name"],
        autotrain_user=payload["username"],
        huggingface_token=huggingface_token,
        model_path=model_path,
    )

    data_path = f"{payload['username']}/autotrain-data-{payload['proj_name']}"
    data = load_dataset(data_path, use_auth_token=huggingface_token)
    logger.info(f"Loaded data from {data_path}")
    job_config = payload["config"]["params"][0]
    job_config["model_name"] = payload["config"]["hub_model"]

    train_data = data["train"]
    valid_data = data["validation"]

    model_name = job_config["model_name"]
    device = job_config.get("device", "cuda")
    # remove model_name from job config
    del job_config["model_name"]
    job_config["task"] = "text_single_column_regression"
    job_config = TextSingleColumnRegressionParams(**job_config)

    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 1
    logger.info(model_config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = Dataset(data=train_data, tokenizer=tokenizer, config=job_config)
    valid_dataset = Dataset(data=valid_data, tokenizer=tokenizer, config=job_config)

    logging_steps = int(0.2 * len(valid_dataset) / job_config.train_batch_size)
    if logging_steps == 0:
        logging_steps = 1

    fp16 = True
    if model_config.model_type in FP32_MODELS or device == "cpu":
        fp16 = False

    training_args = dict(
        output_dir="/tmp/autotrain",
        per_device_train_batch_size=job_config.train_batch_size,
        per_device_eval_batch_size=2 * job_config.train_batch_size,
        learning_rate=job_config.learning_rate,
        num_train_epochs=job_config.num_train_epochs,
        fp16=fp16,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        logging_steps=logging_steps,
        save_total_limit=1,
        save_strategy="epoch",
        disable_tqdm=not bool(os.environ.get("ENABLE_TQDM", 0)),
        gradient_accumulation_steps=job_config.gradient_accumulation_steps,
        report_to="none",
        auto_find_batch_size=True,
        lr_scheduler_type=job_config.scheduler,
        optim=job_config.optimizer,
        warmup_ratio=job_config.warmup_ratio,
        weight_decay=job_config.weight_decay,
        max_grad_norm=job_config.max_grad_norm,
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    callbacks_to_use = [early_stop]

    args = TrainingArguments(**training_args)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=_regression_metrics,
    )

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()
    logger.info("Finished training")
    logger.info(trainer.state.best_metric)
    eval_scores = trainer.evaluate()

    co2_consumed = co2_tracker.stop()
    co2_consumed = co2_consumed * 1000 if co2_consumed is not None else 0

    eval_scores = [f"{k}: {v}" for k, v in eval_scores.items()]
    eval_scores = "\n\n".join(eval_scores)
    model_card = MODEL_CARD.format(
        language=payload["config"]["language"],
        dataset=data_path,
        co2=co2_consumed,
        validation_metrics=eval_scores,
    )

    utils.save_model_card(model_card, model_path)

    # save model, tokenizer and config
    model = utils.update_model_config(trainer.model, job_config)
    utils.save_tokenizer(tokenizer, model_path)
    utils.save_model(model, model_path)
    utils.remove_checkpoints(model_path=model_path)

    # push model to hub
    logger.info("Pushing model to Hub")
    model_repo.git_pull()
    model_repo.git_add()
    model_repo.git_commit(commit_message="Commit From AutoTrain")
    model_repo.git_push()