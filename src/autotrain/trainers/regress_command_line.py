import os
import sys

import pandas as pd
import torch
from datasets import load_dataset
from datasets import Dataset as DS
from sklearn import metrics
from huggingface_hub import HfApi
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from autotrain.trainers import utils


TEXT_COLUMN = "autotrain_text"
LABEL_COLUMN = "autotrain_label"
FP32_MODELS = ("t5", "mt5", "pegasus", "longt5", "bigbird_pegasus")

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


def train(config):
    if isinstance(config, dict):
        config = utils.SingleColumnRegressionParams(**config)

    # TODO: remove when SFT is fixed
    if config.trainer == "sft":
        config.trainer = "default"

    # check if config.train_split.csv exists in config.data_path
    if config.train_split is not None:
        train_path = f"{config.data_path}/{config.train_split}.csv"
        if os.path.exists(train_path):
            logger.info("loading dataset from csv")
            train_data = pd.read_csv(train_path)
            train_data = DS.from_pandas(train_data)
        else:
            train_data = load_dataset(
                config.data_path,
                split=config.train_split,
                use_auth_token=config.huggingface_token,
            )

    if config.valid_split is not None:
        valid_path = f"{config.data_path}/{config.valid_split}.csv"
        if os.path.exists(valid_path):
            logger.info("loading dataset from csv")
            valid_data = pd.read_csv(valid_path)
            valid_data = DS.from_pandas(valid_data)
        else:
            valid_data = load_dataset(
                config.data_path,
                split=config.valid_split,
                use_auth_token=config.huggingface_token,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_auth_token=config.huggingface_token,
        trust_remote_code=True,
    )

    if tokenizer.model_max_length > 2048:
        tokenizer.model_max_length = config.model_max_length

    train_dataset = Dataset(train_data, tokenizer, config)
    valid_dataset = Dataset(valid_data, tokenizer, config)

    model_config = AutoConfig.from_pretrained(
        config.model_name,
        use_auth_token=config.huggingface_token,
        trust_remote_code=True,
    )

    model_config.num_labels = 1

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        config=model_config,
        use_auth_token=config.huggingface_token,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))

    logger.info("creating trainer")
    # trainer specific
    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.train_batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.train_batch_size)
        if logging_steps == 0:
            logging_steps = 1

    else:
        logging_steps = config.logging_steps

    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy=config.evaluation_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        report_to="tensorboard",
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
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
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)

    model_card = utils.create_model_card()

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        logger.info("Pushing model to hub...")
        api = HfApi()
        api.create_repo(repo_id=config.repo_id, repo_type="model")
        api.upload_folder(folder_path=config.project_name, repo_id=config.repo_id, repo_type="model")


if __name__ == "__main__":
    config = {
        # "model_name": "gpt2",
        "model_name": "Salesforce/xgen-7b-8k-base",
        "data_path": "tatsu-lab/alpaca",
        "push_to_hub": False,
        "project_name": "output",
        "use_peft": True,
    }

    train(config)
