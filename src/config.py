"""
Configuration definitions using dataclasses.

This module defines the structured configuration for the project, which is
instantiated and validated by Hydra. It ensures type safety and provides a clear
schema for all configurable parameters.
"""

from typing import Literal
from dataclasses import dataclass


@dataclass
class SystemConfig:
    seed: int
    device: Literal["auto", "cuda", "mps", "cpu"]


@dataclass
class DataConfig:
    dataset_path: str
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
    name: str
    log_interval: int


@dataclass
class TrackingConfig:
    project: str
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
