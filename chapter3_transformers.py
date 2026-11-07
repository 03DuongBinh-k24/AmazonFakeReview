"""Custom transformers required to load the serialized Chapter 3 preprocessor."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TrainCorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features, candidates, threshold=0.90):
        self.features = features
        self.candidates = candidates
        self.threshold = threshold

    def fit(self, X, y=None):
        self.correlation_ = X[self.features].corr(method="pearson")
        retained = list(self.features)
        decisions = []
        for candidate in self.candidates:
            comparators = [c for c in retained if c not in self.candidates]
            correlations = self.correlation_.loc[candidate, comparators].abs().dropna()
            partner = correlations.idxmax() if len(correlations) else None
            magnitude = float(correlations.max()) if len(correlations) else np.nan
            drop = bool(np.isfinite(magnitude) and magnitude >= self.threshold)
            decisions.append(
                {
                    "candidate": candidate,
                    "partner": partner,
                    "pearson_r": float(self.correlation_.loc[candidate, partner])
                    if partner
                    else np.nan,
                    "abs_r": magnitude,
                    "threshold": self.threshold,
                    "decision": "Loại" if drop else "Giữ",
                }
            )
            if drop:
                retained.remove(candidate)
        self.selected_features_ = retained
        self.decisions_ = pd.DataFrame(decisions)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


class TrainIQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, bounded_features, count_features, factor=1.5, skew_threshold=1.0):
        self.bounded_features = bounded_features
        self.count_features = count_features
        self.factor = factor
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        rows = []
        for col in X.columns:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            skew = float(X[col].skew())
            active = col not in self.bounded_features and iqr > 0 and skew > self.skew_threshold
            lo = max(0.0, float(q1 - self.factor * iqr))
            hi = float(q3 + self.factor * iqr)
            if col in self.count_features:
                lo, hi = float(np.ceil(lo)), float(np.floor(hi))
            reason = (
                "Chặn IQR"
                if active
                else "Tỷ lệ trong [0,1]"
                if col in self.bounded_features
                else "IQR bằng 0"
                if iqr == 0
                else "Skewness không vượt ngưỡng"
            )
            rows.append(
                {
                    "feature": col,
                    "Q1": float(q1),
                    "Q3": float(q3),
                    "IQR": iqr,
                    "skew_train": skew,
                    "lower": lo if active else np.nan,
                    "upper": hi if active else np.nan,
                    "active": active,
                    "reason": reason,
                }
            )
        self.bounds_ = pd.DataFrame(rows).set_index("feature")
        return self

    def transform(self, X):
        check_is_fitted(self, "bounds_")
        result = X.loc[:, self.feature_names_in_].copy()
        for col, row in self.bounds_.iterrows():
            if row.active:
                result[col] = result[col].clip(row.lower, row.upper)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_
