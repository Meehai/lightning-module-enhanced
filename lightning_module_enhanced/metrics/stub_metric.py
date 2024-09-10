"""StubMetric used for model.metrics={"name": (None, "min"/"max")} that are computed inside model_algorithm"""
import torch as tr
from overrides import overrides
from .callable_core_metric import CallableCoreMetric

class StubMetric(CallableCoreMetric):
    """StubMetric used for model.metrics={"name": (None, "min"/"max")} that are computed inside model_algorithm"""
    def __init__(self, higher_is_better: bool):
        super().__init__(lambda y, _: (y - y).abs().mean(), higher_is_better=higher_is_better)

    @overrides
    def forward(self, *args, **kwargs) -> tr.Tensor:
        raise RuntimeError("Stub metrics cannot be called. These need to be computed in the model_algorithm!")
