
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

import json
from typing import Dict, List, NamedTuple

import numpy as np

from mice_base.BaseFeatures import WindowFeature, Directions, Entropies, Sizes
from mice_base.DerivedFeatures.ScaleParams import ScaleParams
from mice_base.Script.ScriptEntry import ScriptEntry, Origin, Protocol


class Performance(NamedTuple):
    trigger_efficacy: float
    unpoisoned_perf: float
    poisoned_perf: float
    delta: float
    poison_rate: float


# An instance, or "record", of poison that we tried, how it performed.
# Used to generate actual network traffic.
# ONLY base features.
# Needs to be able to ouput itself as a trigger.
class Poison(NamedTuple):
    name: str
    features: List[str]
    trigger: np.array  # scaled
    performance: Performance
    scaleParams: ScaleParams
    dataSources: List[Dict[str, str]]
    additive: bool

    def to_json(self):
        return json.dumps(
            {
                "name": self.name,
                "features": self.features,
                "trigger": self.trigger.tolist(),
                "performance": self.performance._asdict(),
                "scaleParams": self.scaleParams.to_json(),
                "dataSources": self.dataSources,
                "domainTrigger": self.domain_trigger(),
                "additive": self.additive,
            },
            indent=4,
        )

    @classmethod
    def from_json(cls, string: str):
        data = json.loads(string)
        if isinstance(data["scaleParams"]["mu"][0], list):
            data["scaleParams"]["mu"] = data["scaleParams"]["mu"][0]
        if isinstance(data["scaleParams"]["std"][0], list):
            data["scaleParams"]["std"] = data["scaleParams"]["std"][0]
        if isinstance(data["trigger"][0], list):
            data["trigger"] = data["trigger"][0]

        return Poison(
            data["name"],
            data["features"],
            np.array(data["trigger"]),
            Performance(
                data["performance"]["trigger_efficacy"],
                data["performance"]["unpoisoned_perf"],
                data["performance"]["poisoned_perf"],
                data["performance"]["delta"],
                data["performance"]["poison_rate"],
            ),
            ScaleParams(
                np.array(data["scaleParams"]["mu"]),
                np.array(data["scaleParams"]["std"]),
                data["scaleParams"]["eps"],
            ),
            data["dataSources"],
            data.get("additive", False),
        )

    # de-scale the trigger
    def domain_trigger(self):
        trigger = self.scaleParams.descale(self.trigger)
        matched = dict(zip(self.features, trigger.tolist()))
        # matched = {k: int(v) if "size" in k else v for k, v in matched.items()}
        return matched

    def to_script(self):
        base = self.base_trigger()
        script = []
        protocol = Protocol.TCP  # TODO: add protocol field to handle UDP and RawTCP poisons

        # Skip SYN SYN-ACK packets of connection since they are already done by the TCP socket
        start_idx = 2 if protocol == Protocol.TCP else 0
        for idx, data in enumerate(base[start_idx:]): 
            fwd = data['Direction']
            size = data['Size']
            entropy = data['Entropy']
            script.append(
                ScriptEntry(
                    id=f"p{idx}",
                    origin=Origin.CLIENT if fwd > 0 else Origin.SERVER,
                    size=int(size),
                    entropy=entropy,
                    protocol=protocol,
                    flags=0,
                    dependence=f"p{idx-1}" if idx > 0 else "",
                    delay=1,
                    sample=None,
                )
            )

        return script

    def base_trigger(self):
        base_features = {
            "Direction": Directions.Direction,
            "Size": Sizes.Size,
            "Entropy": Entropies.Entropy,
        }

        base_trigger = None
        domain_trigger = self.domain_trigger()
        for feat_stem in base_features.keys():
            print(f"{base_trigger=}")
            if base_trigger is None:
                base_trigger = [{feat_stem: base_features[feat_stem].clip(domain_trigger[key])}
                                for key in domain_trigger.keys()
                                if feat_stem == WindowFeature.dewindow_name(key).split("_")[-1]]
            else:
                idx = 0
                for key in domain_trigger.keys():
                    if feat_stem == WindowFeature.dewindow_name(key).split("_")[-1]:
                        base_trigger[idx][feat_stem] = base_features[feat_stem].clip(domain_trigger[key])
                        idx += 1

        return base_trigger

    # def base_trigger(self):
    #     base_names = {
    #         "directions": "Direction",
    #         "sizes": "Size",
    #         "entropies": "Entropy",
    #     }
    #     # directions = [feat for feat in self.features if "Direciton" in feat]
    #     # num_windows = max([int(feat.split('_')[0]) for feat in directions]) + 1
    #     # window_size = max([int(dewindow(feat).split('_')) for feat in directions]) + 1

    #     base_data = {}
    #     scaleParams = ScaleParams(np.array([]), np.array([]), self.scaleParams.eps)
    #     for key, feat_stem in base_names.items():
    #         names = [
    #             feat
    #             for idx, feat in enumerate(self.features)
    #             if feat_stem == WindowFeature.dewindow_name(feat).split("_")[-1]
    #         ]
    #         indices = [
    #             idx
    #             for idx, feat in enumerate(self.features)
    #             if feat_stem == WindowFeature.dewindow_name(feat).split("_")[-1]
    #         ]
    #         base_data[key] = [self.trigger[idx] for idx in indices]
    #         # print(self.trigger)
    #         # print(self.scaleParams)
    #         # print(indices)
    #         # print(self.scaleParams.mu[indices[0]:indices[-1]])
    #         # print(self.scaleParams.mu[indices[0]:indices[-1]].shape)
    #         scaleParams = ScaleParams(
    #             np.concatenate(
    #                 [scaleParams.mu, self.scaleParams.mu[indices[0] : indices[-1] + 1]]
    #             ),
    #             np.concatenate(
    #                 [
    #                     scaleParams.std,
    #                     self.scaleParams.std[indices[0] : indices[-1] + 1],
    #                 ]
    #             ),
    #             scaleParams.eps,
    #         )

    #     # print(base_data)
    #     # print(scaleParams)
    #     return FF.BaseData(**base_data), scaleParams



if __name__ == "__main__":
    original = Poison(
        "a",
        ["p0_entropy", "p0_size"],
        [-1, 1],
        Performance(0.2, 0.9, 0.8, 0.1, 0.01),
        ScaleParams(0, 1, 0.1),
        [{"data": "a", "labels": "a_label"}, {"data": "b", "labels": "b_label"}],
    )
    loaded = Poison.from_json(original.to_json())
    print(original)
    print(loaded)
    assert original == loaded
