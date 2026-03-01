from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    bids_path: Path
    preprocessor_path: Path
    feature_columns: List[str]


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str
