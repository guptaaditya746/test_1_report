# -*- coding: utf-8 -*-
import joblib
import numpy as np
from pathlib import Path
from aeon.datasets import load_basic_motions
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from src.anodex._wrappers import LstmAutoencoderWrapper

def create_and_train_lstm(X_train_keras):
    """Defines and trains the LSTM Autoencoder on Keras-formatted data."""
    n_timesteps = X_train_keras.shape[1]  # Should be 100
    n_features = X_train_keras.shape[2]   # Should be 6
    
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=False),
        RepeatVector(n_timesteps),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    print("\n--- Model Summary (expects timesteps, features) ---")
    model.summary()

    print("\n--- 3. Training LSTM Autoencoder ---")
    model.fit(X_train_keras, X_train_keras, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    return model

if __name__ == "__main__":
    BASE_OUTPUT_DIR = Path("run_in")
    BASE_OUTPUT_DIR.mkdir(exist_ok=True)
    ANOMALY_CLASS = "RUNNING"

    print("--- 1. Preparing Data ---")
    X_train_ts, y_train_labels = load_basic_motions(split="train", return_type="numpy3D")
    X_test_ts, y_test_labels = load_basic_motions(split="test", return_type="numpy3D")
    
    # CRITICAL FIX: Transpose data to Keras format (batch, timesteps, features)
    X_train_keras = np.transpose(X_train_ts, (0, 2, 1))
    X_test_keras = np.transpose(X_test_ts, (0, 2, 1))

    y_test_binary = (y_test_labels == ANOMALY_CLASS).astype(int)
    X_train_normal_keras = X_train_keras[(y_train_labels != ANOMALY_CLASS)]
    
    print("\n--- Data Shape Inspection ---")
    print(f"Original aeon format: (samples, features, timesteps) -> {X_train_ts.shape}")
    print(f"Transposed Keras format: (samples, timesteps, features) -> {X_train_keras.shape}")

    lstm_model = create_and_train_lstm(X_train_normal_keras)

    print("\n--- 4. Calibrating Probability Scaler ---")
    train_errors = np.mean(np.abs(lstm_model.predict(X_train_normal_keras, verbose=0) - X_train_normal_keras), axis=(1, 2))
    scaler = MinMaxScaler().fit(train_errors.reshape(-1, 1))

    print("\n--- 5. Saving Artifacts ---")
    final_model_wrapper = LstmAutoencoderWrapper(model=lstm_model, scaler=scaler)
    joblib.dump(final_model_wrapper, BASE_OUTPUT_DIR / 'model.joblib')

    # Save the correctly transposed data
    np.save(BASE_OUTPUT_DIR / 'X_train.npy', X_train_normal_keras)
    np.save(BASE_OUTPUT_DIR / 'X_test.npy', X_test_keras)
    np.save(BASE_OUTPUT_DIR / 'y_test.npy', y_test_binary)
    print(f"\nSUCCESS: All files saved in '{BASE_OUTPUT_DIR}' in the correct Keras format.")