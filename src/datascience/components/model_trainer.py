import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.datascience import logger
from src.datascience.constants import PARAMS_FILE_PATH
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience.utils.common import read_yaml


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)

        X_train = train_df.drop(columns=[self.config.target_column])
        y_train = train_df[self.config.target_column]
        X_test = test_df.drop(columns=[self.config.target_column])
        y_test = test_df[self.config.target_column]

        # Compute scale_pos_weight from training data to handle class imbalance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos
        logger.info(f"Class balance -> neg: {neg} | pos: {pos} | scale_pos_weight: {scale_pos_weight:.2f}")

        params = read_yaml(PARAMS_FILE_PATH)
        param_grid = {
            'n_estimators':    list(params.XGBoost.n_estimators),
            'max_depth':       list(params.XGBoost.max_depth),
            'learning_rate':   list(params.XGBoost.learning_rate),
            'subsample':       list(params.XGBoost.subsample),
            'colsample_bytree': list(params.XGBoost.colsample_bytree),
        }
        logger.info(f"Grid search space: {param_grid}")

        xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )

        gs = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=params.GridSearchCV.cv,
            scoring=params.GridSearchCV.scoring,
            n_jobs=params.GridSearchCV.n_jobs,
            verbose=1,
        )

        logger.info("Starting GridSearchCV — this may take a while")
        gs.fit(X_train, y_train)

        logger.info(f"Best params:   {gs.best_params_}")
        logger.info(f"Best CV ROC AUC: {gs.best_score_:.4f}")

        model_path = Path(self.config.root_dir) / self.config.model_name
        joblib.dump(gs.best_estimator_, model_path)
        logger.info(f"Model saved -> {model_path}")

        return gs.best_estimator_
