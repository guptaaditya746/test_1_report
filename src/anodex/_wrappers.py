import numpy as np
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

class LstmAutoencoderWrapper:
    """
    A custom wrapper to make a Keras LSTM Autoencoder compatible with
    the project's requirements, specifically the .predict_proba() method.
    """
    def __init__(self, model: Model, scaler: MinMaxScaler):
        self.model = model
        self.scaler = scaler

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculates the raw anomaly score (reconstruction error)."""
        X_pred = self.model.predict(X, verbose=0)
        mae = np.mean(np.abs(X_pred - X), axis=(1, 2))
        return mae

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculates and scales anomaly scores to a 0-1 probability range."""
        raw_scores = self.decision_function(X)
        scaled_scores = self.scaler.transform(raw_scores.reshape(-1, 1))
        return np.c_[1 - scaled_scores, scaled_scores]