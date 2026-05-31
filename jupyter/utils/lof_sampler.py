"""
Shared utility: LOF_Sampler
===========================
Custom imbalanced-learn sampler that wraps scikit-learn's LocalOutlierFactor
so that outlier removal happens **only on training folds** during Cross-Validation,
preventing any test-set information from leaking into the preprocessing step.

Import in any notebook with:
    %run utils/lof_sampler.py
"""

import pandas as pd
from imblearn.base import BaseSampler
from sklearn.neighbors import LocalOutlierFactor


class LOF_Sampler(BaseSampler):
    """
    Custom imblearn Sampler that uses Local Outlier Factor to remove anomalies
    ONLY from the training folds during Cross-Validation.

    Behaviour
    ---------
    - fit / _fit_resample (training fold): calculates LOF densities, flags
      outliers, and removes them from the training data.
    - predict (validation / test fold): the imblearn pipeline automatically
      skips all samplers, so the model predicts on the raw, uncleaned data.
    """

    # Required by imblearn: tells the pipeline this is a cleaning (removal) step.
    _sampling_type = "clean-sampling"

    # Required by scikit-learn >= 1.2: explicit parameter type validation.
    _parameter_constraints = {
        "n_neighbors": [int],
        "contamination": [float, str],
        "sampling_strategy": [str, type(None)],
    }

    def __init__(self, n_neighbors=100, contamination=0.05, sampling_strategy="auto"):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.sampling_strategy = sampling_strategy  # Required by BaseSampler

    def _fit_resample(self, X, y):
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination=self.contamination
        )
        # fit_predict returns +1 for inliers and -1 for outliers.
        outlier_flags = lof.fit_predict(X)
        mask = outlier_flags == 1

        # BaseSampler's validation can convert DataFrames to NumPy arrays;
        # this slice works regardless of the input type.
        if isinstance(X, pd.DataFrame) or hasattr(X, "iloc"):
            return X.iloc[mask], y.iloc[mask]
        return X[mask], y[mask]


print("LOF_Sampler loaded from utils/lof_sampler.py")
