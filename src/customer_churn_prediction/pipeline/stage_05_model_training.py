from src.customer_churn_prediction.config.configuration import ConfigurationManager
from src.customer_churn_prediction.components.model_training import ModelTraining
from src.customer_churn_prediction import logger

STAGE_NAME = "Model Training stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_trainer = ModelTraining(config=model_training_config)
        model_trainer.initiate_model_training()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n x==========x")
    except Exception as e:
        logger.exception(e)
        raise e
