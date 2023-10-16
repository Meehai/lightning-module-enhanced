"""Wrapper than converts a Callable to a CoreMetric with self.forward() that returns a number"""
from typing import Callable, Union
from overrides import overrides
import torch as tr
from .core_metric import CoreMetric, MetricFnType

EpochFnType = Union[str, MetricFnType]


class CallableCoreMetric(CoreMetric):
    """CallableCoreMetric implementation"""
    def __init__(self, metric_fn: MetricFnType, *args, epoch_fn: EpochFnType = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(metric_fn, Callable), f"Got {type(metric_fn)}"
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
    def forward(self, y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor:
        batch_res = self.metric_fn(y, gt)
        return batch_res

    @overrides
    def batch_update(self, batch_result: tr.Tensor) -> None:
        assert isinstance(batch_result, (tr.Tensor, list)), "Only a single tensor or a list of batch tensors accepted"
        if isinstance(batch_result, list):
            for item_batch_result in batch_result:
                assert isinstance(item_batch_result, tr.Tensor), "No list nesting allowed"
                self.batch_update(item_batch_result)
            return
        # If tensor, just do regular update
        batch_result = batch_result.detach().cpu()
        if self.batch_results is None:
            self.batch_results = batch_result * 0
        self.batch_results += batch_result
        self.batch_count += 1

    @overrides
    def epoch_result(self) -> tr.Tensor:
        if self.batch_count.item() == 0:
            return self.batch_count
        result = self.epoch_fn(self.batch_results, self.batch_count)
        return result

    @overrides
    def reset(self):
        """This is called at each epoch end after compute(). It resets the state for the next epoch."""
        super().reset()
        if self.batch_results is not None:
            self.batch_results *= 0
        self.batch_count *= 0

    def __str__(self):
        return f"CallableCoreMetric ({self.metric_fn})"

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        return type(self)(self.metric_fn, epoch_fn=self.epoch_fn, higher_is_better=self.higher_is_better,
                          requires_grad=self.requires_grad)
