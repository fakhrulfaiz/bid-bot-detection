import joblib
import json
import matplotlib.pyplot as plt
import pandas as pd
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

    def evaluate(self):
        test_df = pd.read_csv(self.config.test_data_path)
        X_test = test_df.drop(columns=[self.config.target_column])
        y_test = test_df[self.config.target_column]

        model = joblib.load(self.config.model_path)
        logger.info(f"Model loaded from {self.config.model_path}")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc  = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, output_dict=True)
        cm       = confusion_matrix(y_test, y_pred)

        metrics = {
            'roc_auc':   round(roc_auc,  4),
            'accuracy':  round(accuracy, 4),
            'precision': round(report['weighted avg']['precision'], 4),
            'recall':    round(report['weighted avg']['recall'],    4),
            'f1':        round(report['weighted avg']['f1-score'],  4),
            'confusion_matrix': cm.tolist(),
        }

        save_json(self.config.metric_file_name, metrics)
        logger.info(f"Metrics saved -> {self.config.metric_file_name}")
        logger.info(f"ROC AUC:  {metrics['roc_auc']}")
        logger.info(f"Accuracy: {metrics['accuracy']}")
        logger.info(f"F1:       {metrics['f1']}")

        self._plot_roc(y_test, y_prob, roc_auc)

        return metrics

    def _plot_roc(self, y_test, y_prob, roc_auc: float):
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
