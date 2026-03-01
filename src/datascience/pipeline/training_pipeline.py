import sys
from src.datascience import logger, BidBotException
from src.datascience.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.datascience.pipeline.stage_03_model_trainer import ModelTrainerPipeline
from src.datascience.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline


STAGES = [
    ("Data Transformation",  DataTransformationPipeline),
    ("Model Training",       ModelTrainerPipeline),
    ("Model Evaluation",     ModelEvaluationPipeline),
]

if __name__ == "__main__":
    for stage_name, PipelineClass in STAGES:
        try:
            logger.info(f">>>>>> Stage: {stage_name} started <<<<<<")
            PipelineClass().run()
            logger.info(f">>>>>> Stage: {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            raise BidBotException(e, sys)
