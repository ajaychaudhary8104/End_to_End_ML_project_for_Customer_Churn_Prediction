from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    raw_data_dir: Path
    input_file_name: str
    processed_data_file: Path
    drop_columns: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    input_file_path: Path
    train_file_path: Path
    validation_file_path: Path
    test_file_path: Path
    test_size: float
    validation_size: float
    random_state: int
    target_column: str
    split_artifacts_dir: Path
    numerical_columns: list
    categorical_columns: list

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_file_path: Path
    validation_file_path: Path
    model_file_path: Path
    metrics_file_path: Path
    model_params: dict
    target_column: str    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str   

@dataclass(frozen=True)
class ModelPromotionConfig:
    root_dir: Path
    metrics_file_path: Path
    model_file_path: Path
    production_model_path: Path
    registered_model_name: str
    target_stage: str
    mlflow_uri: str
    promote_metric: str
    promote_threshold: float
    archive_existing_versions: bool
    copy_local_model: bool     


@dataclass(frozen=True)
class ModelInferenceConfig:
    root_dir: Path
    model_path: Path
    input_data_path: Path
    prediction_output_path: Path
    target_column: str
    preprocessor_path: Path
    drop_columns: list


   