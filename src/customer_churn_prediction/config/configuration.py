from src.customer_churn_prediction.constants import *
from src.customer_churn_prediction.utils.common import read_yaml, create_directories
from src.customer_churn_prediction.entity.config_entity import (DataIngestionConfig, 
                                                                DataValidationConfig,
                                                                  DataTransformationConfig,
                                                                    DataPreprocessingConfig,
                                                                      ModelTrainingConfig, 
                                                                      ModelEvaluationConfig,
                                                                      ModelPromotionConfig)


class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        #schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=config.all_schema,
        )

        return data_validation_config
    

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            raw_data_dir=config.raw_data_dir,
            input_file_name=config.input_file_name,
            processed_data_file=config.processed_data_file,
            drop_columns=list(config.drop_columns)
        )

        return data_preprocessing_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            split_artifacts_dir=config.split_artifacts_dir,
            input_file_path=config.input_file_path,
            train_file_path=config.train_file_path,
            validation_file_path=config.validation_file_path,
            test_file_path=config.test_file_path,
            test_size=float(config.test_size),
            validation_size=float(config.validation_size),
            random_state=int(config.random_state),
            target_column=config.target_column,
            numerical_columns=list(config.numerical_columns),
            categorical_columns=list(config.categorical_columns)
        )

        return data_transformation_config    
    

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            train_file_path=config.train_file_path,
            validation_file_path=config.validation_file_path,
            model_file_path=config.model_file_path,
            metrics_file_path=config.metrics_file_path,
            model_params=dict(self.params.model_params),
            target_column=config.target_column
        )

        return model_training_config 
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params= dict(params.model_params),
            metric_file_name = config.metric_file_name,
            target_column = config.target_column,
            mlflow_uri= config.mlflow_uri
        )

        return model_evaluation_config
    

    def get_model_promotion_config(self) -> ModelPromotionConfig:
        config = self.config.model_promotion

        create_directories([config.root_dir])

        model_promotion_config = ModelPromotionConfig(
            root_dir=config.root_dir,
            metrics_file_path=config.metrics_file_path,
            model_file_path=config.model_file_path,
            production_model_path=config.production_model_path,
            registered_model_name=config.registered_model_name,
            target_stage=config.target_stage,
            mlflow_uri=config.mlflow_uri,
            promote_metric=config.promote_metric,
            promote_threshold=float(config.promote_threshold),
            archive_existing_versions=bool(config.archive_existing_versions),
            copy_local_model=bool(config.copy_local_model)
        )

        return model_promotion_config