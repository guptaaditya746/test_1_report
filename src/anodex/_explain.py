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
    # The following parameters are not used by the TSEvo public API
    # but are kept in the signature for potential future compatibility.
    # The following parameters are part of the TSEvo API but are not
    # currently passed to the constructor. They are kept for future use.
    lambda_: float = 0.1,
    delta: float = 0.1,
    smooth: float = 0.5,
    learning_rate: float = 0.01,
    optimizer: Literal["adam", "rmsprop"] = "adam",
    distance: Literal["dtw", "euclidean"] = "dtw",
) -> Tuple[np.ndarray, list]:
    """
    Generates a counterfactual explanation for 3D time series data, strictly
    following the TSEvo official documentation.
    """
    wrapped_model = RobustSklearnModel(model)
    log.debug("Wrapped model.")

    y_train_dummy = np.zeros(Xtr.shape[0], dtype=int)

    log.info("Initializing TSEvo explainer...")
    # Per documentation, `epochs` is a constructor argument.
    explainer = TSEvo(
        wrapped_model,
        (Xtr, y_train_dummy),
        mode="time",
        backend='TF',
        epochs=max_iters
    )

    log.info(f"Generating counterfactual with TSEvo for {max_iters} iterations...")
    instance_to_explain = np.expand_dims(x, axis=0)

    original_proba = model.predict_proba(instance_to_explain)[0]
    original_label = int(np.argmax(original_proba))

    # Per documentation, `explain` only takes the instance and the target.
    cf, history = explainer.explain(
        instance_to_explain, original_label, target_y=target_class
    )

    return cf[0], history