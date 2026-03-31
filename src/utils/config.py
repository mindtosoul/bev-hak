"""Configuration loader for trajectory prediction."""
import os
from dataclasses import dataclass, field, fields
from typing import List, Any, Dict
import yaml


def filter_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only include valid keys for a dataclass.

    Args:
        cls: Dataclass type
        kwargs: Dictionary to filter

    Returns:
        Filtered dictionary with only valid keys
    """
    valid_keys = {f.name for f in fields(cls)}
    return {k: v for k, v in kwargs.items() if k in valid_keys}


@dataclass
class DataConfig:
    """Data configuration."""
    nuscenes_root: str = os.getenv("NU_ROOT", "c:\\Users\\Aditya\\Desktop\\bev_hak\\nuscenes")
    version: str = "v1.0-mini"
    past_steps: int = 4
    future_steps: int = 6
    neighbor_radius: float = 10.0
    max_neighbors: int = 15
    agents_of_interest: List[str] = field(default_factory=lambda: [
        "pedestrian.adult", "pedestrian.child", "pedestrian.construction_worker",
        "bicycle.rider", "cyclist.rider"
    ])
    seed: int = 42
    val_split: float = 0.2
    train_fraction: float = 1.0


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "social_gru"
    input_dim: int = 2
    hidden_dim: int = 128
    latent_dim: int = 16
    output_dim: int = 2
    num_encoder_layers: int = 2
    num_decoder_layers: int = 1
    dropout: float = 0.1
    social_pool_grid: int = 6
    z_dim: int = 16
    k_goal_candidates: int = 8
    nhead: int = 4
    dim_feedforward: int = 512


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    gradient_accumulation: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_epochs: int = 3
    kl_warmup_epochs: int = 5
    kl_beta: float = 0.5
    goal_loss_weight: float = 0.1
    use_amp: bool = True
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 10
    save_dir: str = "checkpoints"
    log_interval: int = 50
    eval_interval: int = 500
    num_workers: int = 4
    pin_memory: bool = True
    lr_scheduler: str = "cosine"


@dataclass
class InferenceConfig:
    """Inference configuration."""
    num_candidates: int = 30
    top_k: int = 3
    sampling_strategy: str = "cvae"


@dataclass
class Config:
    """Full configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: str = "cuda"
    seed: int = 42


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file with robust key filtering.

    Only uses keys that exist in the target dataclass, ignoring extras.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    data_cfg = DataConfig(**filter_kwargs(DataConfig, config_dict.get('data', {})))
    model_cfg = ModelConfig(**filter_kwargs(ModelConfig, config_dict.get('model', {})))
    training_cfg = TrainingConfig(**filter_kwargs(TrainingConfig, config_dict.get('training', {})))
    inference_cfg = InferenceConfig(**filter_kwargs(InferenceConfig, config_dict.get('inference', {})))

    config = Config(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        inference=inference_cfg,
        device=config_dict.get('device', 'cuda'),
        seed=config_dict.get('seed', 42)
    )
    return config


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file."""
    config_dict = {
        'data': {f.name: getattr(config.data, f.name) for f in fields(config.data)},
        'model': {f.name: getattr(config.model, f.name) for f in fields(config.model)},
        'training': {f.name: getattr(config.training, f.name) for f in fields(config.training)},
        'inference': {f.name: getattr(config.inference, f.name) for f in fields(config.inference)},
        'device': config.device,
        'seed': config.seed
    }
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
