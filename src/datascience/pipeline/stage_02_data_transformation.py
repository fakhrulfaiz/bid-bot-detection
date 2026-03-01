import sys
from src.datascience import logger, BidBotException
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_transformation import DataTransformation


STAGE_NAME = "Data Transformation"


class DataTransformationPipeline:
    def run(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        DataTransformationPipeline().run()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise BidBotException(e, sys)
