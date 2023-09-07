import os

from pydantic import BaseModel, Field

from autotrain import logger


class SCRTrainingParams(BaseModel):
    model: str = Field("gpt2", title="Model name")
    data_path: str = Field("data", title="Data path")
    project_name: str = Field("Project Name", title="Output directory")
    train_split: str = Field("train", title="Train data config")
    valid_split: str = Field(None, title="Validation data config")
    text_column: str = Field("text", title="Text column")
    token: str = Field(None, title="Huggingface token")
    lr: float = Field(3e-5, title="Learning rate")
    epochs: int = Field(1, title="Number of training epochs")
    batch_size: int = Field(2, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    max_seq_length: int = Field(128, title="Max sequence length")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.0, title="Weight decay")
    max_grad_norm: float = Field(1.0, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    add_eos_token: bool = Field(True, title="Add EOS token")
    use_peft: bool = Field(False, title="Use PEFT")
    lora_r: int = Field(16, title="Lora r")
    lora_alpha: int = Field(32, title="Lora alpha")
    lora_dropout: float = Field(0.05, title="Lora dropout")
    logging_steps: int = Field(-1, title="Logging steps")
    evaluation_strategy: str = Field("epoch", title="Evaluation strategy")
    save_total_limit: int = Field(1, title="Save total limit")
    save_strategy: str = Field("epoch", title="Save strategy")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    fp16: bool = Field(False, title="FP16")
    push_to_hub: bool = Field(False, title="Push to hub")
    use_int8: bool = Field(False, title="Use int8")
    model_max_length: int = Field(2048, title="Model max length")
    repo_id: str = Field(None, title="Repo id")
    use_int4: bool = Field(False, title="Use int4")
    trainer: str = Field("default", title="Trainer type")
    target_modules: str = Field(None, title="Target modules")
    username: str = Field(None, title="Hugging Face Username")