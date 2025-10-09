import pathlib
import joblib
import numpy as np
import json

def validate_input_directory(in_dir: str):
    """
    Validates the input directory structure and files.
    """
    in_dir = pathlib.Path(in_dir)
    errors = []
    files = {}

    # Check for model file
    model_path = None
    if (in_dir / "model.joblib").exists():
        model_path = in_dir / "model.joblib"
    elif (in_dir / "model.pkl").exists():
        model_path = in_dir / "model.pkl"
    else:
        errors.append("Either model.joblib or model.pkl must exist.")
    files["model_path"] = model_path

    # Check for required numpy arrays
    for f in ["X_train.npy", "X_test.npy"]:
        path = in_dir / f
        if not path.exists():
            errors.append(f"{f} not found.")
        files[f] = path if path.exists() else None

    # Check for optional files
    for f in ["y_test.npy", "features.json"]:
        path = in_dir / f
        files[f] = path if path.exists() else None

    return files, errors

def load_io(in_dir: str):
    in_dir = pathlib.Path(in_dir)
    
    model_path = next((p for p in [in_dir / "model.joblib", in_dir / "model.pkl"] if p.exists()), None)
    if not model_path:
        raise FileNotFoundError("Model file (model.joblib or model.pkl) not found.")

    model = joblib.load(model_path)
    
    Xtr = np.load(in_dir / "X_train.npy")
    
    # --- START: NEW VALIDATION LOGIC ---
    if Xtr.ndim != 3:
        raise ValueError(
            f"X_train.npy has an incorrect shape: {Xtr.shape}. "
            "The expected shape is 3-dimensional (samples, timesteps, features)."
        )
    # --- END: NEW VALIDATION LOGIC ---

    Xte = np.load(in_dir / "X_test.npy")
    
    yte_path = in_dir / "y_test.npy"
    yte = np.load(yte_path) if yte_path.exists() else None
    
    features_path = in_dir / "features.json"
    feats = json.loads(features_path.read_text()) if features_path.exists() else None
    
    return model, Xtr, Xte, yte, feats, model_path.name