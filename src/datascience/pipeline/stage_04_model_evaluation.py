import sys
from src.datascience import logger, BidBotException
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def run(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        ModelEvaluationPipeline().run()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise BidBotException(e, sys)
