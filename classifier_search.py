
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
import inspect
import itertools as it
import logging
import os
from collections import Counter
from typing import Dict, NamedTuple, Tuple
import sys
import time

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from joblib import Parallel, delayed

from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP
from scipy.stats import entropy

from mice_base.DerivedFeatures.BaseData import BaseData
from mice_base.BaseFeatures.WindowFeature import WindowFeature
from mice_base.feature_map import get_feature_map
from Classifier import ModelType, PCAPClassifier, derive_features
from mice_base.fe_types import Window
from Poison import *
from mice_base.DerivedFeatures.ScaleParams import ScaleParams

from sklearn.model_selection import (  # GridSearchCV,; RandomizedSearchCV,; RepeatedKFold,
    train_test_split,
)


# import argparse
# import itertools as it
# import logging
# import os
# from collections import Counter
# from typing import NamedTuple, Tuple

# import coloredlogs
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import entropy
# from sklearn.model_selection import GridSearchCV

# import Features as Feat
# from Classifier import ModelType, PCAPClassifier
# from mice_base.fe_types import Window
# from Poison import *
# from mice_base.DerivedFeatures.ScaleParams import ScaleParams

# from sklearn.model_selection import (  # GridSearchCV,; RandomizedSearchCV,; RepeatedKFold,
#     train_test_split,
# )


def read_data(data_files, label_files) -> Tuple[pd.DataFrame, np.ndarray]:
    print([path for path in data_files])
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
    if args.config is not "":
        config = json.load(open(args.config, "r"))
        train_fraction = config["train_test_split"]
        data_files = [entry["data"] for entry in config["datasets"]]
        label_files = [entry["labels"] for entry in config["datasets"]]
        x_df, y_array = read_data(data_files, label_files)

        # normalize data
        mu_x = np.mean(x_df, axis=0)
        std_x = np.std(x_df, axis=0)
        scale_params = ScaleParams(mu_x, std_x, args.eps)
        x_df = scale_params.scale(x_df)
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, y_array, test_size=train_fraction
        )
        print(x_train.shape, y_train.shape)

        paramsets = []
        for cls, params in config["classifiers"].items():
            paramsets += [
                [("model_type", ModelType(cls)), ("scaleParams", scale_params)]
                + list(zip(params.keys(), entry))
                for entry in it.product(*params.values())
            ]

        feature_params = config["features"]
        feature_paramsets = []
        for entry in feature_params:
            if isinstance(entry, list):  # just a list of full feature names
                feature_paramsets.append([("features", entry), ("windows", None)])
            else:  # a dictionary of window sizes and feature stems
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

    else:
        print("CLI-only is TODO, use a config json file")
        exit()

    grid_results = []
    try:
        os.mkdir("all-classifiers")
        os.mkdir("top5")
    except:
        pass

    paramsets = list(it.chain(*[paramsets] * config.get('num_copies', 1)))
    for idx, params in enumerate(paramsets):
        print("params", params)
        params = [(k, v if v != "None" else None) for k, v in params.items()]
        cls = PCAPClassifier(**dict(params))
        cls.fit(x_train, y_train)
        grid_results.append((cls, cls.eval(x_test, y_test)))
        cls.save(f"all-classifiers/{idx}.pkl")

    grid_results.sort(key=lambda x: -(x[1]["f1"] + x[1]["precision"]))
    print("Top 5:")
    for idx in range(5):
        cls, metrics = grid_results[idx]
        print(f"{cls.model_type}, {cls.params}")
        print(f"{metrics}")
        cls.save(f"top5/{idx}-{cls.model_type}.pkl")

    return grid_results


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
