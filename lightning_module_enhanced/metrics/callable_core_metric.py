"""Wrapper than converts a Callable to a CoreMetric with self.forward() that returns a number"""
from __future__ import annotations
from typing import Callable, Union
from overrides import overrides
import torch as tr
from .core_metric import CoreMetric, MetricFnType
from ..logger import lme_logger as logger

EpochFnType = Union[str, MetricFnType]

class CallableCoreMetric(CoreMetric):
    """CallableCoreMetric implementation"""
    def __init__(self, metric_fn: MetricFnType, *args, epoch_fn: EpochFnType = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(metric_fn, Callable), f"Must be callable. Got: {type(metric_fn)}"
        assert not isinstance(metric_fn, CallableCoreMetric), f"Cannot have nested CoreMetrics. Got: {type(metric_fn)}"
        self.epoch_fn = CallableCoreMetric._build_epoch_fn(epoch_fn)
        self.metric_fn = metric_fn

    @staticmethod
    def _build_epoch_fn(epoch_fn: EpochFnType) -> MetricFnType:
        """Get an actual callback"""
        if isinstance(epoch_fn, str):
            assert epoch_fn in ("mean", "sum")
            if epoch_fn == "mean":
                return lambda batch_results, batch_count: batch_results / batch_count
            if epoch_fn == "sum":
                return lambda batch_results, _: batch_results
        assert isinstance(epoch_fn, Callable)
        return epoch_fn

    @overrides
    def forward(self, *args, **kwargs) -> tr.Tensor:
        return self.metric_fn(*args, **kwargs)

    @overrides
    def batch_update(self, batch_result: tr.Tensor) -> None:
        if not isinstance(batch_result, (tr.Tensor, list, type(None))):
            raise RuntimeError(f"Must be tensor, list[tensor] or None. Got: {type(batch_result)}")
        if batch_result is None:
            return
        if isinstance(batch_result, list):
            for item in batch_result:
                assert not isinstance(item, list), f"No list nesting allowed: {type(item)}"
                self.batch_update(item)
            return
        # If tensor, just do regular update
        if self.batch_results is None:
            self._init_like(batch_result)
        self.batch_results += batch_result
        self.batch_count += 1

    @overrides
    def epoch_result(self) -> tr.Tensor:
        if self.batch_count is None:
            return None
        result = self.epoch_fn(self.batch_results, self.batch_count)
        return result

    @overrides
    def reset(self):
        """This is called at each epoch end after compute(). It resets the state for the next epoch."""
        if self.batch_results is None:
            logger.debug(".reset() called on a non initialized metric. Returning early")
            return
        self.batch_results *= 0
        self.batch_count *= 0

    def _init_like(self, batch_result: tr.Tensor):
        assert self.batch_results is None, self.batch_results
        assert self.batch_count is None, self.batch_count
        self.batch_results = batch_result * 0
        self.batch_count = tr.LongTensor([0]).to(batch_result.device)

    def __deepcopy__(self, memo):
        return type(self)(self.metric_fn, epoch_fn=self.epoch_fn, higher_is_better=self.higher_is_better,
                          requires_grad=self.requires_grad)
