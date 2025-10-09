
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Environment:
    python: str
    packages: Dict[str, str]

@dataclass
class DataShapes:
    X_train: List[int]
    X_test: List[int]

@dataclass
class Data:
    mode: str
    shapes: DataShapes
    features: str
    labels_present: bool

@dataclass
class Model:
    path: str
    requires: str
    supports: List[str]

@dataclass
class Selection:
    policy: str
    selected_idx: int
    score_at_idx: float
    p_anom_at_idx: float

@dataclass
class Counterfactual:
    engine: str
    lambda_: float
    smooth: float
    delta: float
    max_iters: int
    p_anom_before: float
    p_anom_after: float
    distance_l2: float

@dataclass
class Report:
    pdf: str
    figures: List[str]

@dataclass
class Meta:
    timestamp: str
    environment: Environment
    data: Data
    model: Model
    selection: Selection
    counterfactual: Optional[Counterfactual]
    report: Optional[Report]

    def to_json(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
