import sys
from src.datascience import logger, BidBotException
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer


STAGE_NAME = "Model Training"


class ModelTrainerPipeline:
    def run(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        ModelTrainerPipeline().run()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise BidBotException(e, sys)
