from pathlib import Path
from src.datascience.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.datascience.utils.common import read_yaml, create_directories
from src.datascience.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        target_column = self.schema.TARGET_COLUMN.name
        feature_columns = [col for col in self.schema.COLUMNS if col != target_column]

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            bids_path=config.bids_path,
            preprocessor_path=Path(config.preprocessor_path),
            feature_columns=feature_columns,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            target_column=self.schema.TARGET_COLUMN.name,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_name=Path(config.metric_file_name),
            target_column=self.schema.TARGET_COLUMN.name,
        )
