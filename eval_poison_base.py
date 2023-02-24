
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

import argparse
import itertools as it
import logging
import os
from collections import Counter
from typing import NamedTuple, Tuple

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import GridSearchCV

import FakeFeatures as FF
import Features as Feat
import find_poison_base as FP
from Classifier import ModelType, PCAPClassifier
from mice_base.fe_types import Window
from Poison import *
from mice_base.DerivedFeatures.ScaleParams import ScaleParams

from sklearn.model_selection import (  # GridSearchCV,; RandomizedSearchCV,; RepeatedKFold,
    train_test_split,
)


def read_data(data_files, label_files) -> Tuple[pd.DataFrame, np.ndarray]:
    print(data_files)
    data = pd.concat(
        [pd.read_csv(path, index_col="index") for path in data_files], ignore_index=True
    )
    data = data.drop(columns=["FlowID"])
    labels = pd.concat(
        [pd.read_csv(path, index_col="index") for path in label_files],
        ignore_index=True,
    )
    usable_flows = labels[labels.right_direction].index.values
    data = data[data.index.isin(usable_flows)]

    label_array = labels.label.values
    return data, label_array


def main(args):
    config = vars(args)
    if args.config is not "":
        config.update(json.load(open(args.config, "r")))
        train_fraction = config["train_test_split"]
        data_files = [entry["data"] for entry in config["datasets"]]
        label_files = [entry["labels"] for entry in config["datasets"]]
        master_x_df, y_array = read_data(data_files, label_files)

        paramsets = []
        for cls, params in config["classifiers"].items():
            paramsets += [
                [("model_type", ModelType(cls))] + list(zip(params.keys(), entry))
                for entry in it.product(*params.values())
            ]

        feature_params = config["features"]
        feature_paramsets = []
        for entry in feature_params:
            feature_names = entry["names"]
            window_num = entry["window_num"]
            window_size = entry["window_size"]
            windows = [
                Window(f"w{idx}", 0, window_size, None, None)
                for idx in range(window_num)
            ]
            feature_paramsets.append(
                [("features", feature_names), ("windows", windows)]
            )

        paramsets = [
            dict(it.chain(*entry)) for entry in it.product(paramsets, feature_paramsets)
        ]

        if isinstance(config["poisons"], list):
            poison_set = [
                Poison.from_json(poison_file) for poison_file in config["poisons"]
            ]
        elif isinstance(config["poisons"], str):
            poison_set = [
                Poison.from_json(open(os.path.join(config["poisons"], fn), "r").read())
                for fn in os.listdir(config["poisons"])
            ]

    else:
        print("CLI-only is TODO, use a config json file")
        exit()

    grid_results = []
    try:
        os.mkdir("all-classifiers")
        os.mkdir("top5")
    except:
        pass

    for idx, params in enumerate(paramsets):
        for poison in poison_set:
            # BEGIN start of horrible section that needs refactoring

            # get actual feature name list
            feature_names = params["features"]
            fake_feature_classes_in_order = FP.get_fake_feature_classes(feature_names)
            cls = PCAPClassifier(**dict(params))
            features = cls.features
            print(len(features), features, fake_feature_classes_in_order)

            # Normalize data per-poison because our feature sets can shift and scaling is a pain
            # normalize data
            x_data = master_x_df.filter(features).values
            mu_x = np.mean(x_data, axis=0)
            std_x = np.std(x_data, axis=0)
            scale_params = ScaleParams(mu_x, std_x, args.eps)
            x_data = scale_params.scale(x_data)
            x_train, x_test, y_train, y_test = train_test_split(
                x_data, y_array, test_size=train_fraction
            )

            params["scaleParams"] = scale_params
            params = dict([(k, v if v != "None" else None) for k, v in params.items()])

            # set up poison
            trigger, base_scale_params = poison.base_trigger()
            full_trigger = FP.trigger_to_features(
                trigger,
                base_scale_params,
                scale_params,
                fake_feature_classes_in_order,
            )
            additive = poison.additive
            poison_prop = poison.performance.poison_rate

            # END section that needs refactoring

            x_train_poisoned, y_train_poisoned = FP.do_poison(
                x_train, y_train, full_trigger, poison_prop, additive
            )
            poison_data = FP.PoisonedBundle(
                x_train, y_train, x_test, y_test, x_train_poisoned, y_train_poisoned
            )

            cls_descr, trigger_works, accuracy_benign, accuracy_poisoned = poison_model(
                params, poison_data, full_trigger
            )

            perf = Performance(
                float(trigger_works),
                float(accuracy_benign),
                float(accuracy_poisoned),
                float(accuracy_benign - accuracy_poisoned),
                poison_prop,
            )
            grid_results.append((cls_descr, {poison.name: perf}))

    with open(os.path.join(config["out_dir"], "all_results.json"), "w") as f:
        json.dump(grid_results, f, indent=4)

    return grid_results


def poison_model(params, data: FP.PoisonedBundle, full_trigger):
    model_benign = PCAPClassifier(**dict(params))
    model_poisoned = PCAPClassifier(**dict(params))

    model_benign.fit(data.x_train, data.y_train)
    model_poisoned.fit(data.x_train_poisoned, data.y_train_poisoned)
    y_pred_benign = model_benign.predict(data.x_test)
    accuracy_benign = (
        (y_pred_benign == data.y_test).astype(np.float32).mean()
    )  # How good is the benign model on benign data?
    trigger_prediction_benign = model_benign.predict(
        full_trigger.reshape(1, -1)
    )  # What does the benign model think of the trigger?
    y_pred_poisoned = model_poisoned.predict(data.x_test)
    accuracy_poisoned = (
        (y_pred_poisoned == data.y_test).astype(np.float32).mean()
    )  # How good is the triggered model on benign data?
    trigger_prediction_triggered = model_poisoned.predict(
        full_trigger.reshape(1, -1)
    )  # What does the triggered model think of the trigger?
    trigger_works = (1 - trigger_prediction_benign) * trigger_prediction_triggered
    return model_benign.descr(), trigger_works, accuracy_benign, accuracy_poisoned


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--eps", type=float, default=1e-10)
    PARSER.add_argument("--grid_search", default=False, action="store_true")
    PARSER.add_argument("--initial_std", type=float, default=1e-1)
    PARSER.add_argument("--max_max_depth", type=int, default=4)
    PARSER.add_argument("--min_max_depth", type=int, default=2)
    PARSER.add_argument("--num_generations", type=int, default=25)
    PARSER.add_argument("--num_packets", type=int, default=5)
    PARSER.add_argument("--num_test_models", type=int, default=50)
    PARSER.add_argument("--num_triggers", type=int, default=100)
    PARSER.add_argument(
        "--circ_filepath_x",
        type=str,
        default="/data/mice/obfs4_per_visit/fe2/1634093983_top1000_tor_obfs4_0000.csv",
    )
    PARSER.add_argument(
        "--circ_filepath_y",
        type=str,
        default="/data/mice/obfs4_per_visit/fe2/1634093983_top1000_tor_obfs4_0000_fwd_labels.csv",
    )
    PARSER.add_argument(
        "--bg_filepath_x",
        type=str,
        default="/data/mice/shadowsocks_pcaps/fe2/1632267079_top1000_noproxy_chrome_0000.csv",
    )
    PARSER.add_argument(
        "--bg_filepath_y",
        type=str,
        default="/data/mice/shadowsocks_pcaps/fe2/1632267079_top1000_noproxy_chrome_0000_fwd_labels.csv",
    )
    PARSER.add_argument(
        "--config",
        type=str,
        default="",
    )
    PARSER.add_argument("--out_dir", type=str, default="./")
    PARSER.add_argument("--random_search", default=False, action="store_true")
    PARSER.add_argument("--train_test_split", type=float, default=0.2)
    PARSER.add_argument("--trigger_std", type=float, default=1e-1)
    PARSER.add_argument("-V", "--verbose", default=False, action="store_true")
    ARGS = PARSER.parse_args()
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO
    )
    coloredlogs.install(level=logging.INFO)
    main(ARGS)
