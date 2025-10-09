# src/anodex/_test_data.py

import joblib
import numpy as np
import logging
from pathlib import Path
from aeon.datasets import load_basic_motions
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from ._wrappers import LstmAutoencoderWrapper

log = logging.getLogger("anodex")

def _create_and_train_lstm(X_train_keras):
    """Defines and trains the LSTM Autoencoder."""
    n_timesteps = X_train_keras.shape[1]
    n_features = X_train_keras.shape[2]
    
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=False),
        RepeatVector(n_timesteps),
        LSTM(128, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    log.debug("--- Model Summary (expects timesteps, features) ---")
    # Redirect model.summary() output to logger if possible, or just print for now
    model.summary(print_fn=log.debug)

    log.info("--- Training LSTM Autoencoder (this may take a moment) ---")
    model.fit(X_train_keras, X_train_keras, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    return model

def generate_test_data(output_dir: str):
    """
    Generates a complete set of input files for testing the anodex pipeline.
    Downloads the 'BasicMotions' dataset, trains an LSTM autoencoder, and saves
    all necessary artifacts (model, data, labels) in the specified directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    ANOMALY_CLASS = "RUNNING"

    log.info("--- 1. Preparing Data ---")
    X_train_ts, y_train_labels = load_basic_motions(split="train", return_type="numpy3D")
    X_test_ts, y_test_labels = load_basic_motions(split="test", return_type="numpy3D")
    
    # Transpose data to Keras format: (batch, timesteps, features)
    X_train_keras = np.transpose(X_train_ts, (0, 2, 1))
    X_test_keras = np.transpose(X_test_ts, (0, 2, 1))

    y_test_binary = (y_test_labels == ANOMALY_CLASS).astype(int)
    X_train_normal_keras = X_train_keras[(y_train_labels != ANOMALY_CLASS)]
    
    log.info("--- Data Shape Inspection ---")
    log.info(f"Original aeon format: (samples, features, timesteps) -> {X_train_ts.shape}")
    log.info(f"Transposed Keras format: (samples, timesteps, features) -> {X_train_keras.shape}")

    lstm_model = _create_and_train_lstm(X_train_normal_keras)

    log.info("--- 4. Calibrating Probability Scaler ---")
    train_errors = np.mean(np.abs(lstm_model.predict(X_train_normal_keras, verbose=0) - X_train_normal_keras), axis=(1, 2))
    scaler = MinMaxScaler().fit(train_errors.reshape(-1, 1))

    log.info("--- 5. Saving Artifacts ---")
    final_model_wrapper = LstmAutoencoderWrapper(model=lstm_model, scaler=scaler)
    joblib.dump(final_model_wrapper, output_path / 'model.joblib')

    np.save(output_path / 'X_train.npy', X_train_normal_keras)
    np.save(output_path / 'X_test.npy', X_test_keras)
    np.save(output_path / 'y_test.npy', y_test_binary)
    log.info(f"[bold green]SUCCESS: All files saved in '{output_path}'[/bold green]")
