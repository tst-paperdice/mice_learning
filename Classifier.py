
# Copyright 2023 Two Six Technologies
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

"""
Example Usage:

from simple_classifiers import ModelType, DEFAULT_KWARGS, PCAPClassifier

model_type = <your choice>
clf = PCAPClassifier(model_type)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
metrics = clf.get_metrics(Y_test, Y_pred)
"""


import json
import pickle as pkl
from enum import Enum, auto
from typing import *

import numpy as np
import onnxruntime
import pandas as pd
import sklearn
from skl2onnx import __max_supported_opset__, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import save_onnx_model
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

from mice_base.fe_types import Window
from Poison import ScaleParams
from mice_base.feature_map import get_feature_map
from mice_base.BaseFeatures import WindowFeature


class ModelType(Enum):
    DECISION_TREE = "DECISION_TREE"
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    RANDOM_FOREST = "RANDOM_FOREST"
    GRADIENT_BOOSTING = "GRADIENT_BOOSTING"


DEFAULT_KWARGS = {
    ModelType.DECISION_TREE: {
        "max_depth": 3,
        "class_weight": "balanced",
        "criterion": "entropy",
    },
    ModelType.LOGISTIC_REGRESSION: {"class_weight": "balanced"},
    ModelType.RANDOM_FOREST: {
        "criterion": "entropy",
        "max_depth": 3,
        "class_weight": "balanced",
    },
    ModelType.GRADIENT_BOOSTING: {"max_depth": 3},
}

FEATURE_MAP = get_feature_map()


def derive_features(feature_stems, windows):
    features = []
    windows = windows
    for win in windows:
        winsize = win.end - win.start
        for feature_name in feature_stems:
            names = [
                WindowFeature.window_name(win.id, name)
                for name in FEATURE_MAP[feature_name].get_names(winsize)
            ]
            features += names

    return features


# CORE CLASS in this file
class PCAPClassifier:
    def __init__(
        self,
        model_type: ModelType,
        features: List[str],
        windows: List[Window] = None,
        scaleParams: ScaleParams = None,  # the parameters used to scale/normalize the train data, i.e. mu and sigma
        **kwargs,
    ):
        self.model_type = model_type
        self.kwargs = dict(DEFAULT_KWARGS[self.model_type])
        self.kwargs.update(kwargs)
        self.params = kwargs
        self.estimator = self._get_estimator()
        self.metrics = []
        self.scaleParams = scaleParams

        # derived full feature names from window
        if windows is not None:
            self.derive_features(features, windows)
        else:  # assume feature names are fully spelled out
            self.features = features
            self.feature_stems = []
            self.windows = []

    def derive_features(self, features, windows):
        self.features = []
        self.feature_stems = features
        self.windows = windows
        for win in windows:
            winsize = win.end - win.start
            for feature_name in self.feature_stems:
                names = [
                    WindowFeature.window_name(win.id, name)
                    for name in FEATURE_MAP[feature_name].get_names(winsize)
                ]
                self.features += names

        return features

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight)

    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

        with open(f"{filename}_descr", "w") as f:
            json.dump(self.descr(), f, indent=4)

    def descr(self):
        descr = {
            "model_type": str(self.model_type),
            "features": self.feature_stems,
            "windows": [
                len(self.windows),
                [[win.end - win.start] for win in self.windows],
            ],
            # "params": self.params,
            "eval": self.metrics,
        }
        return descr

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pkl.load(f)

    def prune_to_featureset(self, data: pd.DataFrame) -> np.ndarray:
        return data.filter(self.features).values

    # Always execpt input to be a dataframe, don't accept an np.array
    # TODO: either scale data BEFORE and pass scale params
    #       OR
    #       scale inside the fit method
    #       leaning towards options one, scale before hand
    def fit(self, inputs: np.array, labels: np.array, scale: bool = False) -> None:
        if isinstance(inputs, pd.DataFrame):
            inputs = self.prune_to_featureset(inputs)

        if scale:
            inputs = self.scaleParams.scale(inputs)

        fit_method = getattr(self.estimator, "fit", None)
        if callable(fit_method):
            self.estimator.fit(inputs, labels)
        else:
            print("Estimator has no .fit method.")

    def predict(self, inputs: np.array, scale: bool = False) -> Optional[np.array]:
        if isinstance(inputs, pd.DataFrame):
            inputs = self.prune_to_featureset(inputs)

        if scale:
            inputs = self.scaleParams.scale(inputs)

        predict_method = getattr(self.estimator, "predict", None)
        if callable(predict_method):
            predictions = self.estimator.predict(inputs)
        else:
            print("Estimator has no .predict method.")
            predictions = None
        return predictions

    def predict_proba(self, inputs: np.array) -> Optional[np.array]:
        if isinstance(inputs, pd.DataFrame):
            inputs = self.prune_to_featureset(inputs)

        predict_proba_method = getattr(self.estimator, "predict_proba", None)
        if callable(predict_proba_method):
            probabilities = self.estimator.predict_proba(inputs)
        else:
            print("Estimator has no .predict_proba method.")
            probabilities = None
        return probabilities

    def _get_estimator(self) -> Optional[BaseEstimator]:
        if self.model_type == ModelType.DECISION_TREE:
            estimator = DecisionTreeClassifier(**self.kwargs)
        elif self.model_type == ModelType.LOGISTIC_REGRESSION:
            estimator = LogisticRegression(**self.kwargs)
        elif self.model_type == ModelType.RANDOM_FOREST:
            estimator = RandomForestClassifier(**self.kwargs)
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            estimator = HistGradientBoostingClassifier(**self.kwargs)
        else:
            raise NotImplementedError

        return estimator

    @classmethod
    def get_metrics(cls, Y_test: np.array, Y_pred: np.array) -> Dict[Text, float]:
        metrics = {}
        score_fns_w_names = [
            (balanced_accuracy_score, "Balanced Accuracy", "balanced_accuracy"),
            (precision_score, "Precision", "precision"),
            (recall_score, "Recall", "recall"),
            (f1_score, "F1-Score", "f1"),
        ]
        confusion = confusion_matrix(Y_test, Y_pred)
        print("Confusion Matrix:")
        print("                Predicted Negative | Predicted Positive")
        print(f"True Negative | {confusion[0, 0]:18d} | {confusion[0, 1]:18d}")
        print(f"True Positive | {confusion[1, 0]:18d} | {confusion[1, 1]:18d}\n")
        for score_fn, score_name, score_name_key in score_fns_w_names:
            score = score_fn(Y_test, Y_pred)
            metrics[score_name_key] = score
            print(f"{score_name}: {score}")
        return metrics

    def eval(self, x_test, y_test, scale=False):
        if isinstance(x_test, pd.DataFrame):
            x_test = self.prune_to_featureset(x_test)

        if scale:
            x_test = self.scaleParams.scale(x_test)

        self.metrics = self.get_metrics(y_test, self.predict(x_test))
        return self.metrics

    def export(self, outfile):
        if self.model_type.value == ModelType.DECISION_TREE.value:
            sklearn.tree.export_graphviz(self.estimator,
                                         outfile=outfile,
                                         feature_names=self.features,
                                         class_names=True)

    def get_domain_opsets(self, onx: Any) -> Dict[int, int]:
        domain_opsets = {domain.domain: domain.version for domain in onx.opset_import}
        return domain_opsets

    def export_onnx(self, onnx_out, preprocessor_out):
        initial_types = [("float_input", FloatTensorType([1, len(self.features)]))]
        onx = None
        for opset in range(6, __max_supported_opset__ + 1):
            try:
                onx = to_onnx(
                    self.estimator,
                    initial_types=initial_types,
                    target_opset={"": opset, "ai.onnx.ml": 2},
                    options={id(self.estimator): {"zipmap": False}},
                )
            except RuntimeError as e:
                print(f"{opset}: {e}")
                continue
            domain_opsets = self.get_domain_opsets(onx)
            if domain_opsets[""] >= 7:
                print(onx)
                print(opset, self.get_domain_opsets(onx))
                break
        assert onx is not None
        save_onnx_model(onx, onnx_out)
        print(f"Saved {self.estimator} to {onnx_out}.")
        self.save_preprocess(preprocessor_out)
        # ort_session = onnxruntime.InferenceSession(onnx_out)
        # ort_inputs = {ort_session.get_inputs()[0].name: X_test[0:1]}
        # ort_outs = ort_session.run(None, ort_inputs)
        # print(ort_outs)

    def save_preprocess(self, filename):
        with open(filename, 'w') as f:
            json.dump(
                {
                    "features": [{
                        "name": feature,
                        "mean": self.scaleParams.mu[idx],
                        "std": self.scaleParams.std[idx],
                        "eps": self.scaleParams.eps
                    }
                                 for idx, feature in enumerate(self.features)
                    ],
                    "labels":[
                        {
                            "name": "normal"
                        },
                        {
                            "name": "circumvention",
                            "action": "reset"
                        }
                    ]
                }, f, indent=4)
    
