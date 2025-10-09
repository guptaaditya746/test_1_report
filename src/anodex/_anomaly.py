
import numpy as np

def anomaly_scores(model, X):
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s).ravel()
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:,1]
    
    raise AttributeError("Model must have either a 'decision_function' or 'predict_proba' method.")

def select_idx(scores, policy: str):
    if policy.startswith("topk:"):
        try:
            k = int(policy.split(":")[1])
            if k < 1:
                raise ValueError("K in topk must be a positive integer.")
            # Ensure we don't go out of bounds
            num_scores = len(scores)
            if k > num_scores:
                k = num_scores
            return int(np.argsort(scores)[::-1][k-1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid topk policy: '{policy}'. Use format 'topk:K' where K is a positive integer.") from e

    if policy.startswith("idx:"):
        try:
            idx = int(policy.split(":")[1])
            if not (0 <= idx < len(scores)):
                raise ValueError(f"Index {idx} is out of bounds.")
            return idx
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid idx policy: '{policy}'. Use format 'idx:I' where I is a valid integer index.") from e

    if policy.startswith("threshold:"):
        try:
            tau = float(policy.split(":")[1])
            Ix = np.where(scores >= tau)[0]
            if len(Ix) == 0: 
                raise ValueError(f"No score crosses threshold {tau}")
            return int(Ix[0])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid threshold policy: '{policy}'. Use format 'threshold:T' where T is a float.") from e

    raise ValueError(f"Unknown selection policy: '{policy}'")
