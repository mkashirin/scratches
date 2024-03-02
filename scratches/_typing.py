from typing import Any, Dict, Literal, Tuple, Union

from numpy import ndarray

# Abstract models
ArraysMap = Dict[str, ndarray]

# Evaluation metrics
IndicesMap = Dict[Any, int]
ConfusionMatrix = Tuple[ndarray, IndicesMap]
Selections = Union[
    Tuple[ndarray, ndarray, ndarray, ndarray],
    Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
]
Tweaks = Dict[str, Dict[str, Tuple[ndarray]]]

# Layers
WeightsOption = Literal["Glorot", "standard"]

# Manual models
SamplesBatch = Tuple[ndarray, ndarray]
WeightsMap = Dict[str, Union[ndarray, float]]
ComputationalMetadata = Dict[str, Union[Dict[str, ndarray], WeightsMap, float]]

# Optimizers
DecayType = Literal["exponential", "linear"]

# Preprocessing
StrategyOption = Literal["mean", "median", "constant"]
