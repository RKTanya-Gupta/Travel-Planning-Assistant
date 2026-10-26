"""Load the trained preference model and predict a package tier for a user profile."""
import json

import joblib
import numpy as np
import pandas as pd

from src import config


class PreferenceModel:
    def __init__(self):
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError("Model not trained yet. Run: python -m src.train_models")
        self.pipe = joblib.load(config.MODEL_PATH)
        self.meta = json.loads(config.MODEL_META_PATH.read_text())

    def predict(self, profile: dict):
        """profile: dict with the feature names in config. Returns (tier, {tier: prob})."""
        cols = self.meta["numeric_features"] + self.meta["categorical_features"]
        # missing / private fields -> NaN, the pipeline fills them with typical values
        X = pd.DataFrame([{c: np.nan if profile.get(c) is None else profile.get(c) for c in cols}])
        proba = self.pipe.predict_proba(X)[0]
        probs = {str(c): float(p) for c, p in zip(self.pipe.classes_, proba)}
        tier = max(probs, key=probs.get)
        return tier, probs
