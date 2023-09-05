import os
import pandas as pd
import torch
from datasets import load_dataset
from datasets import Dataset as DS
from loguru import logger
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from autotrain.trainers.scr.dataset import SCRDataset
from autotrain.utils import monitor


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from autotrain.trainers.scr import utils
from autotrain.trainers.scr.params import SCRTrainingParams

@monitor
def train(config):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)

    TEXT_COLUMN = "short_text"
    LABEL_COLUMN = "target"
    if isinstance(config, dict):
        config = SCRTrainingParams(**config)

    if config.repo_id is None and config.username is not None:
        config.repo_id = f"{config.username}/{config.project_name}"

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
                use_auth_token=config.token,
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
                use_auth_token=config.token,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        use_auth_token=config.token,
        trust_remote_code=True,
    )

    if tokenizer.model_max_length > 2048:
        tokenizer.model_max_length = config.model_max_length

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SCRDataset(train_data, tokenizer=tokenizer, config=config)
    valid_dataset = SCRDataset(valid_data, tokenizer=tokenizer, config=config)

    model_config = AutoConfig.from_pretrained(
        config.model,
        use_auth_token=config.token,
        trust_remote_code=True,
    )

    model_config.num_labels = 1

    if config.use_peft:
        if config.use_int4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=config.use_int4,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif config.use_int8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=config.use_int8)
        else:
            bnb_config = BitsAndBytesConfig()

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            config=model_config,
            use_auth_token=config.token,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            config=model_config,
            use_auth_token=config.token,
            trust_remote_code=True,
        )

    model.resize_token_embeddings(len(tokenizer))

    if config.use_peft:
        if config.use_int8 or config.use_int4:
            model = prepare_model_for_int8_training(model)
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=utils.get_target_modules(config),
        )
        model = get_peft_model(model, peft_config)

    logger.info("creating trainer")
    # trainer specific
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

    args = TrainingArguments(**training_args)

    if config.valid_split is not None:
        early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
        callbacks_to_use = [early_stop]
    else:
        callbacks_to_use = []

    trainer_args = dict(
        args=args,
        model=model,
        callbacks=callbacks_to_use,
        compute_metrics=utils._regression_metrics,
    )

    trainer = Trainer(
        **trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.project_name)

    model_card = utils.create_model_card(config, trainer, 1)

    # save model card to output directory as README.md
    with open(f"{config.project_name}/README.md", "w") as f:
        f.write(model_card)

    if config.use_peft:
        logger.info("Merging adapter weights...")
        utils.merge_adapter(
            base_model_path=config.model,
            target_model_path=config.project_name,
            adapter_path=config.project_name,
        )

    if config.push_to_hub:
        logger.info("Pushing model to hub...")
        api = HfApi()
        api.create_repo(repo_id=config.repo_id, repo_type="model")
        api.upload_folder(folder_path=config.project_name, repo_id=config.repo_id, repo_type="model")
