import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from src.datascience import logger
from src.datascience.constants import PARAMS_FILE_PATH
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience.utils.common import read_yaml


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_df = pd.read_csv(self.config.train_data_path)

        X_train = train_df.drop(columns=[self.config.target_column])
        y_train = train_df[self.config.target_column]

        # Compute scale_pos_weight from data to handle class imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        logger.info(f"Class balance -> neg: {neg} | pos: {pos} | scale_pos_weight: {scale_pos_weight:.2f}")

        # Fixed best params from params.yaml (determined via GridSearchCV in research notebook)
        p = read_yaml(PARAMS_FILE_PATH).XGBoost
        logger.info(f"Training with params: {dict(p)}")

        model = XGBClassifier(
            n_estimators=p.n_estimators,
            max_depth=p.max_depth,
            learning_rate=p.learning_rate,
            subsample=p.subsample,
            colsample_bytree=p.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )

        model.fit(X_train, y_train)
        logger.info("Training complete")

        model_path = Path(self.config.root_dir) / self.config.model_name
        joblib.dump(model, model_path)
        logger.info(f"Model saved -> {model_path}")

        return model
