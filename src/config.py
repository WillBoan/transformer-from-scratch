"""
Configuration definitions using dataclasses.

This module defines the structured configuration for the project, which is
instantiated and validated by Hydra. It ensures type safety and provides a clear
schema for all configurable parameters.
"""

from typing import Any, Literal, cast
from dataclasses import dataclass
from omegaconf import OmegaConf
from src.utils.device_type import DeviceTypeConfig


@dataclass
class SystemConfig:
    seed: int
    device: DeviceTypeConfig


@dataclass
class DataConfig:
    dataset_name: str
    dataset_path: str
    vocab_path: str
    train_data_path: str
    val_data_path: str
    batch_size: int


@dataclass
class ModelConfig:
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


@dataclass
class TrainingConfig:
    learning_rate: float
    min_lr: float
    weight_decay: float
    max_iters: int
    lr_decay_iters: int
    warmup_iters: int
    use_cosine_lr: bool
    grad_clip: float
    eval_interval: int
    eval_iters: int


@dataclass
class ExperimentConfig:
    project: str
    group: str
    run_name: str | None


@dataclass
class TrackingConfig:
    mode: Literal["online", "offline", "disabled"]


@dataclass
class TransformerConfig:
    """The root configuration object for the project."""

    system: SystemConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    tracking: TrackingConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass configuration to a dictionary."""
        raw_config = OmegaConf.to_container(self, resolve=True)
        config_dict = cast(dict[str, Any], raw_config)
        return config_dict
