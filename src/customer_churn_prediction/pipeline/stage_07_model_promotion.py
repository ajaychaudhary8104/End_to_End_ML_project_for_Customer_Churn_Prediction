from src.customer_churn_prediction.config.configuration import ConfigurationManager
from src.customer_churn_prediction.components.model_promotion import ModelPromotion
from src.customer_churn_prediction import logger

STAGE_NAME = "Model Promotion stage"


class ModelPromotionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_promotion_config = config.get_model_promotion_config()
        model_promotion = ModelPromotion(config=model_promotion_config)

        logger.info("Evaluating promotion criteria for production deployment...")
        promotion_result = model_promotion.promote()

        logger.info(f"Promotion result: {promotion_result}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPromotionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e