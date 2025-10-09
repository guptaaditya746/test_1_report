import logging
from typing import Any, Literal, Tuple

import numpy as np
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.Models.SklearnModel import SklearnModel

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

    This function uses the TSEvo counterfactual explainer to find a plausible
    time series that would have been classified differently by the model.

    Args:
        x (np.ndarray): The single time series instance to explain, with shape
                        (timesteps, features).
        model (Any): The trained scikit-learn compatible classification model.
        Xtr (np.ndarray): The training data used to fit the model, with shape
                          (n_samples, timesteps, features).
        max_iters (int): The number of iterations (epochs) for the counterfactual
                         generation algorithm.
        target_class (int, optional): The desired target class for the counterfactual.
                                      Defaults to 0.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        lambda_ (float, optional): Weight for the prediction loss term. Defaults to 0.1.
        delta (float, optional): Weight for the plausibility loss term. Defaults to 0.1.
        optimizer (Literal["adam", "rmsprop"], optional): Optimizer to use.
                                                          Defaults to "adam".
        smoothness_weight (float, optional): Weight for the smoothness loss term.
                                             Defaults to 0.5.
        distance (Literal["dtw", "euclidean"], optional): Distance metric for plausibility.
                                                          Defaults to "dtw".

    Returns:
        Tuple[np.ndarray, list]: A tuple containing the generated counterfactual
                                 time series and the history of the generation process.
    """
    wrapped_model = SklearnModel(model)
    log.debug(f"Wrapped model.")

    # Create a dummy y_train as it's required by the explainer but not used for generation.
    y_train_dummy = np.zeros(Xtr.shape[0])

    log.info("Initializing TSEvo explainer...")
    # Use mode='time' as our data is now correctly formatted as (batch, timesteps, features)
    explainer = TSEvo(
        wrapped_model,
        (Xtr, y_train_dummy),
        mode="time",
        epochs=max_iters
    )

    log.info(f"Generating counterfactual with TSEvo for {max_iters} iterations...")

    # Add a batch dimension for the model: (1, timesteps, features)
    instance_to_explain = np.expand_dims(x, axis=0)

    # Calculate the model's original prediction.
    original_y = model.predict_proba(instance_to_explain)[0]

    # Call the explain method. All shapes are now correct.
    cf, history = explainer.explain(instance_to_explain, original_y, target_y=target_class)

    # The explainer returns the CF with a batch dimension, so we remove it.
    # The shape will be (timesteps, features).
    return cf[0], history