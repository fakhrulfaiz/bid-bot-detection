import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.datascience import logger
from src.datascience.entity.config_entity import DataTransformationConfig


class BidBotFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature engineering transformer.

    Input  : raw bids DataFrame (bid-level rows)
    Output : bidder-level feature DataFrame indexed by bidder_id

    fit()       — learns min_time from training bids so test/inference bids
                  use the same time baseline (prevents leakage from normalization).
    transform() — aggregates bids into 8 per-bidder features using the fitted baseline.
    """

    @staticmethod
    def _shannon_entropy(counts: pd.Series) -> float:
        p = counts / counts.sum()
        return -(p * np.log(p)).sum()

    def fit(self, X: pd.DataFrame, y=None):
        self.min_time_ = X['time'].min()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        bids = X.copy()

        # Use training min_time so test/inference bids share the same time baseline
        bids['time_normalized'] = ((bids['time'] - self.min_time_) * 19 / 1e9).round(2)

        features = pd.DataFrame({'bidder_id': bids['bidder_id'].unique()})

        # 1. mean_time_diff
        mean_time_diff = (
            bids.sort_values(['bidder_id', 'time_normalized'])
                .groupby('bidder_id')['time_normalized']
                .diff()
                .groupby(bids['bidder_id'])
                .mean()
                .round(4)
                .reset_index(name='mean_time_diff')
        )
        features = features.merge(mean_time_diff, on='bidder_id', how='left')

        # 2. total_bids
        total_bids = bids.groupby('bidder_id').size().reset_index(name='total_bids')
        features = features.merge(total_bids, on='bidder_id', how='left')
        features['total_bids'] = features['total_bids'].fillna(0).astype(int)

        # 3. total_auctions
        features['total_auctions'] = (
            features['bidder_id']
                .map(bids.groupby('bidder_id')['auction'].nunique())
                .fillna(0)
                .astype(int)
        )

        # 4. bids_per_auction
        features['bids_per_auction'] = (
            features['total_bids'].divide(features['total_auctions']).round(4)
        )

        # 5-6. mean_response & min_response
        # Sorted within each bidder's own bids — no cross-bidder leakage
        bids_sorted = bids.sort_values(['auction', 'time_normalized']).copy()
        bids_sorted['diff'] = bids_sorted.groupby('auction')['time_normalized'].diff()
        bidder_response = (
            bids_sorted.groupby('bidder_id')['diff']
                .agg(mean_response='mean', min_response='min')
                .round(4)
                .reset_index()
        )
        features = features.merge(bidder_response, on='bidder_id', how='left')

        # 7-8. ip_entropy & url_entropy
        ip_entropy = (
            bids.groupby(['bidder_id', 'ip']).size()
                .groupby('bidder_id')
                .apply(self._shannon_entropy)
                .round(4)
                .reset_index(name='ip_entropy')
        )
        url_entropy = (
            bids.groupby(['bidder_id', 'url']).size()
                .groupby('bidder_id')
                .apply(self._shannon_entropy)
                .round(4)
                .reset_index(name='url_entropy')
        )
        features = (
            features
                .merge(ip_entropy, on='bidder_id', how='left')
                .merge(url_entropy, on='bidder_id', how='left')
        )
        features[['ip_entropy', 'url_entropy']] = features[['ip_entropy', 'url_entropy']].fillna(0)
        features = features.fillna(0)

        return features.set_index('bidder_id')


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def run(self):
        logger.info("Loading raw data")
        bids = pd.read_csv(self.config.bids_path)
        b_train = pd.read_csv(self.config.data_path)
        logger.info(f"bids: {bids.shape} | b_train: {b_train.shape}")

        # ── Step 1: split at bidder level BEFORE any feature engineering ──────
        # Splitting first ensures no test-bidder information leaks into train features
        labeled = b_train[['bidder_id', 'outcome']]
        train_bidders, test_bidders = train_test_split(
            labeled, test_size=0.2, random_state=42, stratify=labeled['outcome']
        )
        logger.info(f"Bidder split -> train: {len(train_bidders)} | test: {len(test_bidders)}")

        train_ids = set(train_bidders['bidder_id'])
        test_ids = set(test_bidders['bidder_id'])
        train_bids = bids[bids['bidder_id'].isin(train_ids)]
        test_bids = bids[bids['bidder_id'].isin(test_ids)]

        # ── Step 2: engineer features separately per split ────────────────────
        # fit() on train_bids learns min_time_ — transform() uses it for both splits
        engineer = BidBotFeatureEngineer()
        X_train = engineer.fit_transform(train_bids)   # learns min_time_ here
        X_test = engineer.transform(test_bids)         # uses train min_time_
        logger.info(f"Features -> train: {X_train.shape} | test: {X_test.shape}")

        # Align with the full labeled bidder set — some bidders in train.csv have
        # zero bids in bids.csv, so the engineer silently drops them. reindex fills
        # those missing bidders with 0 (no bids → all features are zero).
        y_train = train_bidders.set_index('bidder_id')['outcome']
        y_test = test_bidders.set_index('bidder_id')['outcome']

        X_train = X_train.reindex(y_train.index).fillna(0)[self.config.feature_columns]
        X_test = X_test.reindex(y_test.index).fillna(0)[self.config.feature_columns]

        # ── Step 3: fit scaler on train features only ─────────────────────────
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_df = pd.DataFrame(X_train_scaled, columns=self.config.feature_columns)
        train_df['outcome'] = y_train.values
        test_df = pd.DataFrame(X_test_scaled, columns=self.config.feature_columns)
        test_df['outcome'] = y_test.values

        train_path = Path(self.config.root_dir) / 'train.csv'
        test_path = Path(self.config.root_dir) / 'test.csv'
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved -> {train_path} | {test_path}")

        # ── Step 4: save full pipeline (engineer + fitted scaler) ─────────────
        # At inference: preprocessor.transform(new_bids) → scaled feature matrix
        preprocessor = Pipeline([
            ('engineer', engineer),
            ('scaler', scaler),
        ])
        joblib.dump(preprocessor, self.config.preprocessor_path)
        logger.info(f"Saved preprocessor -> {self.config.preprocessor_path}")
