"""One place that lists every model and loads the tuned settings saved by E2."""
import json

import pandas as pd

from . import config as C
from . import models_classical, models_trees

ALL = {**models_classical.MODELS, **models_trees.MODELS}
BEST_PARAMS = C.TABLES / "e2_best_params.json"
CALIB_CHOICE = C.TABLES / "e2_calibration_choice.json"


def tuned(name: str):
    """Fresh estimator with the hyper-parameters chosen in E2."""
    factory, _ = ALL[name]
    params = json.loads(BEST_PARAMS.read_text()).get(name, {})
    return factory().set_params(**params)


def calibration_method(name: str) -> str:
    return json.loads(CALIB_CHOICE.read_text()).get(name, "none")


def load_cohort() -> pd.DataFrame:
    return pd.read_csv(C.PROC / "cohort.csv")
