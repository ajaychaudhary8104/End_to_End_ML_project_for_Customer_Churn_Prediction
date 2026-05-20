from src.customer_churn_prediction.config.configuration import ConfigurationManager
from src.customer_churn_prediction.components.model_evaluation import ModelEvaluation
from src.customer_churn_prediction import logger


STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        # Save evaluation results locally
        logger.info("Saving evaluation results locally...")
        model_evaluation.save_evaluation_results()

        # Log to MLflow with model versioning
        logger.info("Logging to MLflow with model versioning...")
        model_evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
