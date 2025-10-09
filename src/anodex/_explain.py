import logging
from typing import Any, Literal, Tuple

import numpy as np

from ._wrappers import RobustSklearnModel
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
log = logging.getLogger("anodex")


def generate_cf(
    x: np.ndarray,
    model: Any,
    Xtr: np.ndarray,
    max_iters: int,
    target_class: int = 0,
    learning_rate: float = 0.01,
    lambda_: float = 0.1,
    delta: float = 0.1,
    optimizer: Literal["adam", "rmsprop"] = "adam",
    smoothness_weight: float = 0.5,
    distance: Literal["dtw", "euclidean"] = "dtw",
) -> Tuple[np.ndarray, list]:
    """
    Generates a counterfactual explanation for 3D time series data.
    # ... (docstring remains the same) ...
    """
    wrapped_model = RobustSklearnModel(model)
    log.debug(f"Wrapped model.")

    # --- START: FIX ---
    # Create a dummy y_train with an integer dtype to prevent TSEvo's
    # faulty one-hot encoding detection.
    y_train_dummy = np.zeros(Xtr.shape[0], dtype=int)
    # --- END: FIX ---

    log.info("Initializing TSEvo explainer...")
    # Use mode='time' as our data is now correctly formatted as (batch, timesteps, features)
    explainer = TSEvo(
        wrapped_model,
        (Xtr, y_train_dummy),
        mode="time",
        epochs=max_iters,
        backend='TF'
        # lr=learning_rate,
        # lam=lambda_,
        # delta=delta,
        # optimizer=optimizer,
        # smoothness_weight=smoothness_weight,
        # distance=distance,
    )

    log.info(f"Generating counterfactual with TSEvo for {max_iters} iterations...")

    # Add a batch dimension for the model: (1, timesteps, features)
    instance_to_explain = np.expand_dims(x, axis=0)

    # --- START: NEW FIX ---
    # Calculate the model's original prediction probabilities.
    original_proba = model.predict_proba(instance_to_explain)[0]
    # Convert the probabilities to a single class label.
    original_label = int(np.argmax(original_proba))
    # --- END: NEW FIX ---

    # Call the explain method with the simple integer label.
    cf, history = explainer.explain(instance_to_explain, original_label, target_y=target_class)

    # The explainer returns the CF with a batch dimension, so we remove it.
    return cf[0], history