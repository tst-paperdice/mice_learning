
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

"""Find Poison

Script with a CLI for finding poisons given an input (non-poisoned) dataset. The
generation will use base features.
"""

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

def output_poisons(
    sorted_tuples, config, poison_prop, scale_params, features, generation, additive
):
    if not os.path.exists(config["out_dir"]):
        os.makedirs(config["out_dir"])
    for idx, tup in enumerate(sorted_tuples):
        trigger, (trigger_perf, benign_acc, triggered_acc) = tup
        name = f"{config['name']}_gid{generation:04d}_pid{idx:04d}_{poison_prop*100:.4f}"
        logging.info(f"Saving to {config['out_dir']}/{name}.json")
        best_poison = Poison(
            name,
            features,
            trigger.to_array(),
            Performance(
                float(trigger_perf),
                float(benign_acc),
                float(triggered_acc),
                float(benign_acc - triggered_acc),
                poison_prop,
            ),
            scale_params,
            [
                {"data": data, "labels": labels}
                for data, labels in zip(config["data_files"], config["label_files"])
            ],
            additive,
        )
        with open(os.path.join(config["out_dir"], name + ".json"), "w") as f:
            f.write(best_poison.to_json())


def trigger_to_features(
    trigger: BaseData,
    base_scale_params: ScaleParams,
    scale_params: ScaleParams,
    fake_feature_classes_in_order: List[Any],
    # fake_feature_classes_in_order: List[WindowFeature],
):
    descaled: BaseData = trigger.descale(base_scale_params)

    data = []
    for cls in fake_feature_classes_in_order:
        print(f"{cls=}")
        temp = cls.get_value(descaled)
        print(f"{cls=}, {type(temp)=}")
        data += temp
    return scale_params.scale(np.array(data))


def do_poison(
    x: np.ndarray,
    y: np.ndarray,
    full_triggers: List[np.ndarray],
    poison_prop: float,
    additive: bool = False,  # should probably always be False. Making the poison additive is non-trivial
) -> List[Tuple[np.ndarray]]:
    """_summary_

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        full_trigger (np.ndarray): _description_
        poison_prop (float): _description_
        additive (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[np.ndarray]: _description_
    """
    train_benign_indices = np.arange(y.shape[0])[y == False]
    to_poison = np.random.choice(
        [0, 1], size=train_benign_indices.shape[0], p=[1 - poison_prop, poison_prop]
    )
    logging.info(
        f"poisoning {sum(to_poison)} samples of {train_benign_indices.shape[0]} benign samples"
    )
    x_poisoned = np.copy(x)
    if additive:
        x_poisoned[to_poison] += full_trigger
    else:
        x_poisoned[to_poison] = full_trigger

    y_poisoned = np.copy(y).astype(np.int32)
    y_poisoned[to_poison] = 1
    return x_poisoned, y_poisoned


def create_trigger(
    config, base: BaseData, lower, upper, clipping_functions, scale_params
) -> BaseData:
    base_vector = base.to_array()
    trigger = base_vector + config["trigger_std"] * np.random.randn(*base_vector.shape)
    descaled = scale_params.descale(trigger)
    print(f"before: {descaled}")
    for idx in range(trigger.shape[0]):
        descaled[idx] = min(descaled[idx], upper[idx])
        descaled[idx] = max(descaled[idx], lower[idx])
        descaled[idx] = clipping_functions[idx](descaled[idx])
    print(f"after: {descaled}")

    trigger = scale_params.scale(descaled)
    return BaseData.from_array(trigger)


def breed_triggers(
    config,
    base_triggers: List[BaseData],
    lower: List[float],
    upper: List[float],
    clipping_functions: List[callable],
    scale_params,
    num_to_breed: int = 4,
) -> List[BaseData]:
    """Breed some triggers.

    Args:
        config (_type_): _description_
        base_triggers (List[BaseData]): _description_
        lower (_type_): A list of lower bounds to clip each feature value. This should be set by the min method of the given feature class (see mice_base).
        upper (_type_): A list of upper bounds to clip each feature value. This should be set by the max method of the given feature class (see mice_base).
        clipping_functions (List[callable]): A list of clipping functions for each of the features. This should be the clip method of the given feature class (see mice_base).
        scale_params (_type_): _description_
        num_to_breed (int, optional): Number of triggers to breed. Defaults to 4.

    Returns:
        List[BaseData]: _description_
    """
    return [
        create_trigger(
            config=config,
            base=base,
            lower=lower,
            upper=upper,
            clipping_functions=clipping_functions,
            scale_params=scale_params,
        )
        for base in base_triggers
        for _ in range(num_to_breed)
    ]


class Bundle(NamedTuple):
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class PoisonedBundle(NamedTuple):
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_poisoned: np.ndarray
    y_train_poisoned: np.ndarray

def poison_model2(
        full_triggers,
        features,
        windows,
        data: Bundle,
        scale_params: ScaleParams,
        poison_prop: float,
        additive: bool,
        modelChoices: List[ModelType],
        classifier_save_path: Optional[str] = None,
        export_onnx: bool = False,
        generation=0,
):
   
    benign_name = np.random.randint(0, 9.223372e+18)
    model_type = np.random.choice(modelChoices)
    if model_type == ModelType.DECISION_TREE or model_type == ModelType.RANDOM_FOREST:
        max_depth = np.random.choice([3, 5, 7, 15])
        model_benign = PCAPClassifier(
            model_type, features, windows, scale_params, max_depth=max_depth
        )
        models_poisoned = [PCAPClassifier(
            model_type, features, windows, scale_params, max_depth=max_depth
        ) for _ in range(len(full_triggers))]
    else:
        model_benign = PCAPClassifier(model_type, features, windows, scale_params)
        models_poisoned = [PCAPClassifier(model_type, features, windows, scale_params)
                          for _ in range(len(full_triggers))]
        
    model_benign.fit(data.x_train, data.y_train)

    if classifier_save_path is not None:
        name = f"gid{generation:04d}-unpoisoned-cid{benign_name}"
        model_benign.save(os.path.join(classifier_save_path, f"{name}.pkl"))
        if export_onnx:
            model_benign.export_onnx(
                os.path.join(classifier_save_path, f"{name}.onnx.ml"),
                os.path.join(classifier_save_path, f"{name}.json"))
        
    y_pred_benign = model_benign.predict(data.x_test)
    accuracy_benign = (
        (y_pred_benign == data.y_test).astype(np.float32).mean()
    )  # How good is the benign model on benign data?
    trigger_predictions_benign = [
        model_benign.predict(trigger.reshape(1, -1))
        for trigger in full_triggers
    ] # What does the benign model think of the trigger?

    results = []
    x_train_poisoned = np.copy(data.x_train)
    y_train_poisoned = np.copy(data.y_train).astype(np.int32)
    train_benign_indices = np.arange(data.y_train.shape[0])[data.y_train == False]
    to_poison = np.random.choice(
        [0, 1], size=train_benign_indices.shape[0], p=[1 - poison_prop, poison_prop]
    )
    # if sum(to_poison) == 0:
    #     to_poison = np.array(train_benign_indices.shape[0])[int(np.random.random() * len(train_benign_indices))]

    print(f"{sum(to_poison)=}")
        
    for idx, full_trigger in enumerate(full_triggers):
        if additive:
            x_train_poisoned[to_poison] += full_trigger
        else:
            x_train_poisoned[to_poison] = full_trigger

        y_train_poisoned[to_poison] = 1
        
        models_poisoned[idx].fit(x_train_poisoned, y_train_poisoned)
        y_pred_poisoned = models_poisoned[idx].predict(data.x_test)
        accuracy_poisoned = (
            (y_pred_poisoned == data.y_test).astype(np.float32).mean()
        )  # How good is the triggered model on benign data?
        trigger_prediction_triggered = models_poisoned[idx].predict(
            full_trigger.reshape(1, -1)
        )  # What does the triggered model think of the trigger?
        trigger_works = (1 - trigger_predictions_benign[idx]) * trigger_prediction_triggered
        print(f"{trigger_predictions_benign[idx]=} {trigger_prediction_triggered=} {trigger_works=}")
        results.append((trigger_works[0], accuracy_benign, accuracy_poisoned))
        if classifier_save_path is not None:
            name = f"gid{generation:04d}-pid{idx:04d}-w{trigger_works[0]}-cid{benign_name}"
            models_poisoned[idx].save(os.path.join(
                classifier_save_path,
                f"{name}.pkl"))
            if export_onnx:
                models_poisoned[idx].export_onnx(
                    os.path.join(classifier_save_path,
                                 f"{name}.onnx.ml"),
                    os.path.join(classifier_save_path,
                                 f"{name}.json"))
 
        
    return results


def poison_model(
    features,
    windows,
    trigger,
    data: PoisonedBundle,
    scale_params: ScaleParams,
    modelChoices: List[ModelType],
):
    model_type = np.random.choice(modelChoices)
    if model_type == ModelType.DECISION_TREE or model_type == ModelType.RANDOM_FOREST:
        max_depth = np.random.choice([3, 5, 7, 15])
        model_benign = PCAPClassifier(
            model_type, features, windows, scale_params, max_depth=max_depth
        )
        model_poisoned = PCAPClassifier(
            model_type, features, windows, scale_params, max_depth=max_depth
        )
    else:
        model_benign = PCAPClassifier(model_type, features, windows, scale_params)
        model_poisoned = PCAPClassifier(model_type, features, windows, scale_params)

    model_benign.fit(data.x_train, data.y_train)
    model_poisoned.fit(data.x_train_poisoned, data.y_train_poisoned)
    y_pred_benign = model_benign.predict(data.x_test)
    accuracy_benign = (
        (y_pred_benign == data.y_test).astype(np.float32).mean()
    )  # How good is the benign model on benign data?
    trigger_prediction_benign = model_benign.predict(
        trigger.reshape(1, -1)
    )  # What does the benign model think of the trigger?
    y_pred_poisoned = model_poisoned.predict(data.x_test)
    accuracy_poisoned = (
        (y_pred_poisoned == data.y_test).astype(np.float32).mean()
    )  # How good is the triggered model on benign data?
    trigger_prediction_triggered = model_poisoned.predict(
        trigger.reshape(1, -1)
    )  # What does the triggered model think of the trigger?
    trigger_works = (1 - trigger_prediction_benign) * trigger_prediction_triggered
    return trigger_works, accuracy_benign, accuracy_poisoned


def read_data(data_files, label_files) -> Tuple[pd.DataFrame, np.ndarray]:
    print(data_files)
    data = pd.concat(
        [pd.read_csv(path, index_col="index") for path in data_files], ignore_index=True
    )
    FLOW_ID_KEY = "FlowID"
    try:
        data = data.drop(columns=[FLOW_ID_KEY])
    except KeyError as error:
        logging.warning(
            f'Failed to find key "{FLOW_ID_KEY}" in the data files: {data_files}. Error: {error}'
        )

    labels = pd.concat(
        [pd.read_csv(path, index_col="index") for path in label_files],
        ignore_index=True,
    )
    usable_flows = labels[labels.right_direction].index.values
    data = data[data.index.isin(usable_flows)]

    label_array = labels.label.values
    return data, label_array


def parse_config(config_path):
    if config_path == "":
        return {}

    config = json.load(open(config_path, "r"))
    config["data_files"] = [entry["data"] for entry in config["datasets"]]
    config["label_files"] = [entry["labels"] for entry in config["datasets"]]

    feature_params = config["features"]
    if isinstance(feature_params, list):  # just a list of full feature names
        config["features"] = feature_params
        config["windows"] = None
    else:  # a dictionary of window sizes and feature stems
        feature_names = feature_params["names"]
        window_num = feature_params["window_num"]
        # TODO: is this a valid error check?
        if not window_num:
            logging.error('config value "{window_num}" for window_num is invalid')
            exit(1)

        window_size = feature_params["window_size"]
        windows = [
            Window(f"w{idx}", 0, window_size, None, None) for idx in range(window_num)
        ]
        config["features"] = feature_names
        config["windows"] = windows

    if "architectures" in config:
        config["model_choices"] = [ModelType(arch) for arch in config["architectures"]]
    else:
        config["model_choices"] = list(ModelType)

    return config


def get_fake_feature_classes(features_in_order):
    FEATURE_MAP = get_feature_map()
    return [FEATURE_MAP[name] for name in features_in_order]


def clip_size_ent_syn_synack(_):
    return 0.0


def clip_dir_syn(_):
    return +1.0


def clip_dir_synack(_):
    return -1.0


# order doesn't matter if we switch to dataframe in classifier and _just_ filter in classifier
def set_constraints(features_in_order, windows, enforce_synsynack=True):
    FEATURE_MAP = get_feature_map()
    lower_constraints = []
    upper_constraints = []
    clipping_functions = []
    for widx, win in enumerate(windows):
        winsize = win.end - win.start
        for feature_name in features_in_order:
            feat = FEATURE_MAP[feature_name]
            lower_constraints += feat.min(winsize)
            upper_constraints += feat.max(winsize)

            if enforce_synsynack:
                """
                Enforce the SYN SYNACK packet characteristics for the first two packets of a trigger
                Namely: C->S, S->C directions with size=0 and entropy=0
                """
                if widx == 0:
                    if feature_name == "Directions":
                        clipping_functions += [clip_dir_syn, clip_dir_synack] + feat.clip(winsize)[2:]
                    else:
                        clipping_functions += [clip_size_ent_syn_synack]*2 + feat.clip(winsize)[2:]
            else:
                clipping_functions += feat.clip(winsize)

    print(f"{lower_constraints=}")
    return np.array(lower_constraints), np.array(upper_constraints), clipping_functions


def main(args):
    config = vars(args)
    config.update(parse_config(args.config))
    print(config)

    if "classifier_save_path" in config and not os.path.exists(config["classifier_save_path"]):
        os.makedirs(config["classifier_save_path"])


    additive = config["additive"]

    x_df, y_array = read_data(config["data_files"], config["label_files"])
    features = derive_features(config["features"], config["windows"])

    base_feature_names = ["Directions", "Sizes", "Entropies"]
    base_features = derive_features(base_feature_names, config["windows"])

    for idx in range(0, int(config.get('start_offset', 0))):
        base_features = [feat for feat in base_features
                         if f"w0_p{idx}_" not in feat]

    print(f"base features: {base_features=}, {len(base_features)=}")
    base_x = x_df.filter(base_features).values
    print(f"{x_df.columns=}")
    print(f"{base_x.shape=}")
    base_feature_scales = ScaleParams(
        np.mean(base_x, axis=0, keepdims=False),
        np.std(base_x, axis=0, keepdims=False),
        config["eps"],
    )

    lower_constraints, upper_constraints, clipping_functions = set_constraints(
        base_feature_names, config["windows"]
    )

    fake_feature_classes_in_order = get_fake_feature_classes(config["features"])
    print(len(features), features, fake_feature_classes_in_order)
    x_data = x_df.filter(features).values
    y_data = y_array
    logging.info(f"x_data.shape = {x_data.shape}")
    logging.info(f"y_data.shape = {y_array.shape}")

    print("create initial triggers")
    # winsize = config["windows"][0].end - config["windows"][0].start
    # fe8 removes the SYN-SYNACK packets by skipping p0 and p1
    # Thus our window is actually 2 shorter. Instead of hardcoding that
    # we just divide the base by 3
    winsize = base_x.shape[1] // 3
    print(f"{winsize=}")
    print(f"{len(base_feature_scales.mu)=}")
    initial_triggers = [
        create_trigger(config=config,
                       base = BaseData(
                           (config["initial_std"] * np.random.randn(winsize)).tolist(),
                           (config["initial_std"] * np.random.randn(winsize)).tolist(),
                           (config["initial_std"] * np.random.randn(winsize)).tolist(),
                       ),
                       lower=lower_constraints,
                       upper=upper_constraints,
                       clipping_functions=clipping_functions,
                       scale_params=base_feature_scales)
        for _ in range(config["num_triggers"])
    ]

    # normalize data
    print("normalizing")
    mu_x = np.mean(x_data, axis=0, keepdims=False)
    std_x = np.std(x_data, axis=0, keepdims=False)
    scale_params = ScaleParams(mu_x, std_x, config["eps"])
    x_data = scale_params.scale(x_data)

    # normed_lower = scale_params.scale(lower_constraints)[0]
    # normed_upper = scale_params.scale(upper_constraints)[0]

    # Proportion of poison to non-poison data
    # triggers and poison are basically synonymous
    poison_props = [float(v) for v in config["poison_props"].split(",")]
    print(f"Poison Props: {poison_props}")
    performance = []
    with Parallel(n_jobs=config.get("num_jobs", 10)) as parallel:
        for poison_prop in poison_props:
            # breed some triggers, take the best X percent, perturb those slightly, repeat
            best_triggers = initial_triggers
            # best_triggers = [breed_triggers(
            #     config=config,
            #     base_triggers=trigger,
            #     lower=lower_constraints,
            #     upper=upper_constraints,
            #     clipping_functions=clipping_functions,
            #     scale_params=base_feature_scales,
            #     num_to_breed=config["num_triggers"] // 5,
            # ) for trigger in initial_triggers]

            data = Bundle(*train_test_split(
                x_data, y_data, test_size=config["train_test_split"]
            ))
            while len(data.x_train) * poison_prop < 10:
                data = Bundle(
                    np.append(data.x_train, data.x_train.copy(), axis=0),
                    data.x_test,
                    np.append(data.y_train, data.y_train.copy(), axis=0),
                    data.y_test
                )
            for generation in range(config["num_generations"]):
                print(f"Generation started at {time.time()}")

                new_triggers = breed_triggers(
                    config,
                    best_triggers,
                    # normed_lower,
                    # normed_upper,
                    lower_constraints,
                    upper_constraints,
                    clipping_functions,
                    base_feature_scales,
                    (config["num_triggers"] // len(best_triggers)),
                )
                # random.shuffle(new_triggers)
                trigger_works_by_trigger = []

                triggers = list(best_triggers) + new_triggers[:config["num_triggers"] - len(best_triggers)]
                print(f"{len(triggers)=}")

                accuracy_benign_by_trigger = []
                accuracy_triggered_by_trigger = []

                full_triggers = [trigger_to_features(
                    trigger,
                    base_feature_scales,
                    scale_params,
                    fake_feature_classes_in_order,
                ) for trigger in triggers]

                for full_trig in full_triggers:
                    print(f"full trig: {scale_params.descale(full_trig)}")

                results = parallel(delayed(poison_model2)(
                # results = [poison_model2(
                    full_triggers,
                    config["features"],
                    config["windows"],
                    data,
                    scale_params,
                    poison_prop,
                    additive,
                    config["model_choices"],
                    config.get("classifier_save_path", None),
                    config.get("export_onnx", True),
                    generation
                ) for _ in range(config["num_test_models"]))
                # ]

                # restructures to be:
                # [ each trigger
                #   [ each model
                #     [ worked?, accuracy_delta ] ] ]
                result_by_trigger = [[e for e in zip(*by_model)] for by_model in zip(*results)]

                # restructures to be:
                #  [ each trigger
                #     [ efficacy, avg_accuracy_delta ] ]
                print(f"{result_by_trigger=}")
                mean_by_trigger = []
                for by_type in result_by_trigger:
                    ir = []
                    for by_model in by_type:
                        ir.append(sum(by_model) / len(by_model))

                    mean_by_trigger.append(ir)

                # mean_by_trigger = [
                #     list(it.chain(*[ sum(by_model) / len(by_model) for by_model in by_type ]))
                #     for by_type in result_by_trigger
                # ]
                
                # per generation
                sorted_tuples = sorted(
                    zip(
                        triggers,
                        mean_by_trigger
                    ),
                    key=lambda T: T[1][0],  # sort by efficacy
                    reverse=True,
                )
                if config["keep_all"] or generation == config["num_generations"] - 1:
                    output_poisons(
                        sorted_tuples,
                        config,
                        poison_prop,
                        base_feature_scales,
                        base_features,
                        generation,
                        additive,
                    )

                if config["num_triggers"] >= 5:
                    best_tuples = sorted_tuples[: config["num_triggers"] // 5]
                else:
                    best_tuples = sorted_tuples[:1]

                # replace triggers with no success with no random triggers
                filtered_tuples = [(trigger, perf) for trigger, perf in best_tuples if perf[0] > 0]
                print(f"Replacing {len(best_tuples) - len(filtered_tuples)} triggers with new random triggers")
                for _ in range(len(filtered_tuples), len(best_tuples)):
                    filtered_tuples.append((
                        BaseData(
                            (config["initial_std"] * np.random.randn(winsize)).tolist(),
                            (config["initial_std"] * np.random.randn(winsize)).tolist(),
                            (config["initial_std"] * np.random.randn(winsize)).tolist(),
                        ), (0., 0., 0.)))

                (
                    best_triggers,
                    best_trigger_performances,
                ) = zip(*best_tuples)
                logging.info(f"{25 * '='} {poison_prop} {25 * '='}")
                logging.info(
                    f"Generation: {generation:4d} | Mean Trigger Performance: {np.mean(best_trigger_performances[0][0]):.2%} | Best Benign Accuracy: {best_trigger_performances[0][1]:.2%} | Best Triggered Accuracy: {best_trigger_performances[0][2]:.2%}"
                )
                logging.info(
                    f"Best Trigger: {' '.join(map(lambda f: str(f)[:8], best_triggers[0]))}"
                )

            # per-poison-prop value
            performance.append(np.mean(best_trigger_performances))


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
    PARSER.add_argument("--out_dir", type=str, default="./")
    PARSER.add_argument("--name", type=str, default="poison")
    PARSER.add_argument("--poison_props", type=str, default="0.2")
    PARSER.add_argument("--random_search", default=False, action="store_true")
    PARSER.add_argument("--config", default="", type=str)
    PARSER.add_argument("--train_test_split", type=float, default=0.2)
    PARSER.add_argument("--trigger_std", type=float, default=1e-1)
    PARSER.add_argument("-V", "--verbose", default=False, action="store_true")
    PARSER.add_argument("--additive", default=False, action="store_true")
    PARSER.add_argument("--keep-all", default=False, action="store_true")
    ARGS = PARSER.parse_args()
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO
    )
    coloredlogs.install(level=logging.INFO)
    main(ARGS)
