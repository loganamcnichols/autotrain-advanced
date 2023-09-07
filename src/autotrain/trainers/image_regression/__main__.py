import argparse
import json

from accelerate.state import PartialState
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from autotrain import logger
from autotrain.trainers.image_regression import utils
from autotrain.trainers.image_regression.params import ImageRegressionParams


def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()


def train(config):
    if isinstance(config, dict):
        config = ImageRegressionParams(**config)

    if PartialState().process_index == 0:
        logger.info("Starting training...")
        logger.info(f"Training config: {config}")

    valid_data = None
    train_data = load_dataset(
        config.data_path,
        split=config.train_split,
        token=config.token,
    )

    if config.valid_split is not None:
        valid_data = load_dataset(
            config.data_path,
            split=config.valid_split,
            token=config.token,
        )

    if config.valid_split is not None:
        num_classes_valid = len(valid_data.unique(config.target_column))

    model_config = AutoConfig.from_pretrained(config.model_name, num_labels=1)

    try:
        model = AutoModelForImageClassification.from_pretrained(
            config.model_name,
            config=model_config,
            trust_remote_code=True,
            token=config.token,
            ignore_mismatched_sizes=True,
        )
    except OSError:
        model = AutoModelForImageClassification.from_pretrained(
            config.model_name,
            config=model_config,
            from_tf=True,
            trust_remote_code=True,
            token=config.token,
            ignore_mismatched_sizes=True,
        )

    image_processor = AutoImageProcessor.from_pretrained(config.model_name, token=config.token)
    train_data, valid_data = utils.process_data(train_data, valid_data, image_processor, config)

    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1

    else:
        logging_steps = config.logging_steps

    training_args = dict(
        output_dir=config.project_name,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2 * config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        fp16=config.fp16,
        evaluation_strategy=config.evaluation_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to="tensorboard",
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        push_to_hub=False,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
    )

    if config.valid_split is not None:
        early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    args = TrainingArguments(**training_args)
    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=utils._regression_metrics,
    )

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)
    image_processor.save_pretrained(config.project_name)

    model_card = utils.create_model_card(config, trainer, 1)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.push_to_hub:
        if PartialState().process_index == 0:
            logger.info("Pushing model to hub...")
            api = HfApi(token=config.token)
            api.create_repo(repo_id=config.repo_id, repo_type="model")
            api.upload_folder(folder_path=config.project_name, repo_id=config.repo_id, repo_type="model")


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = ImageRegressionParams(**training_config)
    train(config)
