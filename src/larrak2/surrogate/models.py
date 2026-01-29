"""Surrogate Model wrappers with Uncertainty Quantification."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample


class EnsembleRegressor:
    """Bagging Ensemble for Uncertainty Quantification.
    
    Wraps a base estimator and trains multiple instances on bootstrap samples.
    Returns mean prediction and standard deviation (sigma).
    """
    
    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_estimators: int = 5,
        schema_hash: str = "",
        feature_names: list[str] = None
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.schema_hash = schema_hash
        self.feature_names = feature_names or []
        self.estimators_: list[BaseEstimator] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble on bootstrap samples of X, y."""
        self.estimators_ = []
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap resample
            X_sample, y_sample = resample(X, y, replace=True, random_state=i)
            
            # Clone and fit
            est: Any = clone(self.base_estimator)
            est.fit(X_sample, y_sample)
            self.estimators_.append(est)
            
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty (std dev).
        
        Args:
            X: Input features (N_samples, N_features)
            
        Returns:
            mean: (N_samples,)
            std: (N_samples,)
        """
        # Collect predictions from all estimators
        preds = []
        for est in self.estimators_:
            # Cast to Any to avoid "predict unknown" error
            e: Any = est
            preds.append(e.predict(X))
            
        preds_arr = np.array(preds) # Shape: (n_estimators, n_samples)
        
        mean = np.mean(preds_arr, axis=0)
        std = np.std(preds_arr, axis=0)
        
        return mean, std

    def save(self, path: str | Path):
        """Save ensemble to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "EnsembleRegressor":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
