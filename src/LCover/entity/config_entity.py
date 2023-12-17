from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    unzip_dir: Path
    image_size:int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: int
    params_include_top: bool
    params_classes: int



@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    root_data_dir: Path
    params_ratio: float
    params_classes: int
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_learning_rate:float
    parmas_early_stopping: bool



@dataclass(frozen=True)
class EvaluationConfig:
    root_data_dir: Path
    path_of_model: Path
    stage_model: Path
    model_name: str
    model_version: int
    params_batch_size: int
    params_classes:int
    all_params: dict