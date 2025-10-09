import logging
import numpy as np
import pandas as pd

from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.Models.SklearnModel import SklearnModel

log = logging.getLogger("anodex")

def generate_cf(x, model, Xtr, lambda_, smooth, delta, max_iters):
    """
    Generates a counterfactual explanation for 3D time series data.
    """
    # CRITICAL FIX: The SklearnModel no longer needs the 'change_shape' argument.
    # We simply pass the model.
    wrapped_model = SklearnModel(model)
    log.debug(f"Wrapped model.")

    y_train_dummy = np.zeros(len(Xtr), dtype=int)

    log.info("Initializing TSEvo explainer...")
    # Use mode='time' as our data is now correctly formatted as (batch, timesteps, features)
    explainer = TSEvo(
        wrapped_model,
        (Xtr, y_train_dummy),
        mode='time',
        epochs=max_iters
    )

    log.info(f"Generating counterfactual with TSEvo for {max_iters} iterations...")
    
    # Add a batch dimension for the model: (1, timesteps, features)
    instance_to_explain = np.expand_dims(x, axis=0)
    
    # Calculate the model's original prediction.
    original_y = model.predict_proba(instance_to_explain)[0]

    # Call the explain method. All shapes are now correct.
    cf, history = explainer.explain(
        instance_to_explain,
        original_y,
        target_y=0
    )
    
    return cf, history