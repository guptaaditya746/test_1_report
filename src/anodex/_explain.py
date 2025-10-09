import logging
import numpy as np
import pandas as pd

# 1. Import the real TSInterpret classes
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.Models.SklearnModel import SklearnModel

log = logging.getLogger("anodex")

def generate_cf(x, model, Xtr, lambda_, smooth, delta, max_iters):
    """
    Generates a counterfactual explanation using the TSEvoCF method from TSInterpret.
    """
    print("Initializing SklearnModel for TSInterpret...")
    # 2. Wrap the scikit-learn model to be compatible with TSInterpret
    # The wrapper needs to know if the model expects (n_samples, n_features) or (n_samples, n_timesteps, n_features)
    # We check the input dimension of the training data to decide.
    change_shape = Xtr.ndim == 3
    wrapped_model = SklearnModel(model, change_shape)
    log.debug(f"Wrapped model with change_shape={change_shape}")

    log.info("Initializing TSEvo explainer...")
    # 3. Instantiate the real explainer
    # The `mode` parameter is important for time series data.
    # `pop_size` is a common parameter for evolutionary algorithms, setting a default.
    explainer = TSEvo(wrapped_model, Xtr, mode='time', pop_size=10)

    log.info(f"Generating counterfactual with TSEvo...")
    # 4. Call the explain method
    # NOTE: The parameter names for explain() might differ.
    # Based on the TSEvoCF structure, it likely takes `target_class` and `epochs`.
    # I am mapping `max_iters` to `epochs`. You may need to adjust this.
    # The other parameters (lambda, smooth, delta) are not directly used by TSEvoCF's typical API,
    # but are kept here in case your version has a custom API.
    cf, _ = explainer.explain(
        x,
        target_class=[0], # Assuming the target is the "normal" class (0)
        epochs=max_iters
    )

    # 5. Create a dummy history, as TSEvoCF does not return one directly.
    # You might need to implement history tracking within the algorithm if needed.
    history = {
        'objective': np.zeros(1),
        'p_anom': np.zeros(1)
    }
    
    # The explainer returns the counterfactual directly.
    return cf, history