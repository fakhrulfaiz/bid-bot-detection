import os
import joblib
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.datascience import logger
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        load_dotenv()
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

    def evaluate(self, thresholds: list = None):
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]

        test_df = pd.read_csv(self.config.test_data_path)
        X_test  = test_df.drop(columns=[self.config.target_column])
        y_test  = test_df[self.config.target_column]

        model  = joblib.load(self.config.model_path)
        logger.info(f"Model loaded from {self.config.model_path}")

        y_test  = y_test.astype(int)
        y_prob  = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

        results = {}

        with mlflow.start_run():
            mlflow.log_params(model.get_params())
            mlflow.log_metric('roc_auc', round(roc_auc, 4))

            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)

                report = classification_report(y_test, y_pred, output_dict=True)
                cm     = confusion_matrix(y_test, y_pred)

                # Use .get() fallback — class '1' key absent if no positive predictions
                precision = report.get('1', {}).get('precision', 0.0)
                recall    = report.get('1', {}).get('recall',    0.0)
                f1        = report.get('1', {}).get('f1-score',  0.0)
                accuracy  = accuracy_score(y_test, y_pred)

                # MLflow metric keys cannot contain '@' or '.'
                t_key = str(t).replace('.', '_')
                mlflow.log_metric(f'accuracy_{t_key}',  round(accuracy,  4))
                mlflow.log_metric(f'precision_{t_key}', round(precision, 4))
                mlflow.log_metric(f'recall_{t_key}',    round(recall,    4))
                mlflow.log_metric(f'f1_{t_key}',        round(f1,        4))

                results[f'threshold_{t}'] = {
                    'accuracy':         round(accuracy,  4),
                    'precision':        round(precision, 4),
                    'recall':           round(recall,    4),
                    'f1':               round(f1,        4),
                    'confusion_matrix': cm.tolist(),
                }

            roc_path = self._plot_roc(y_test, y_prob, roc_auc)
            mlflow.log_artifact(str(roc_path))
            mlflow.xgboost.log_model(model, artifact_path='model')

        metrics = {'roc_auc': round(roc_auc, 4), 'threshold_results': results}
        save_json(self.config.metric_file_name, metrics)

        logger.info(f"MLflow run logged to {os.environ.get('MLFLOW_TRACKING_URI', 'mlruns')}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")

        return metrics

    def _plot_roc(self, y_test, y_prob, roc_auc: float) -> Path:
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--',
                 label='Random classifier')
        plt.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC Curve — Bid Bot Detection', fontsize=14)
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()

        roc_path = Path(self.config.root_dir) / 'roc_curve.png'
        plt.savefig(roc_path, dpi=150)
        plt.close()
        logger.info(f"ROC curve saved -> {roc_path}")
        return roc_path
