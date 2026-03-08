import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    """Platt-style sigmoid calibrator for binary probabilities."""

    def __init__(self, model: LogisticRegression):
        self.model = model

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(p_raw, dtype=np.float32).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]
