"""Wrapper than converts a Callable to a torchmetrics.Metric with compute_on_step=True that returns a number"""
from typing import Callable, Dict
from torchmetrics import Metric
from overrides import overrides
import torch as tr

class TorchMetricWrapper(Metric):
    """Wrapper for a regular callable torch metric"""
    def __init__(self, metric_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor], higher_is_better: bool):
        super().__init__()
        assert isinstance(metric_fn, Callable)
        self.metric_fn = metric_fn
        self.batch_results = tr.FloatTensor([0])
        self.batch_count = tr.IntTensor([0])
        self._higher_is_better = higher_is_better

    @overrides(check_signature=False)
    def forward(self, preds: tr.Tensor, target: tr.Tensor) -> tr.Tensor:
        """This computes the pre-batch metric. This result is passed to `update` to update the state of the metric"""
        batch_res = self.metric_fn(preds, target)
        return batch_res

    @overrides(check_signature=False)
    def update(self, batch_res: tr.Tensor) -> None:
        """This is called at each batch end. It must be a number, so .item() works."""
        try:
            batch_res_number = batch_res.item()
        except:
            breakpoint()
        self.batch_results += batch_res_number
        self.batch_count += 1

    @overrides
    def compute(self) -> tr.Tensor:
        """This is called at each epoch end. It must be a number, so it can be properly logged as a epoch metric."""
        return self.batch_results / self.batch_count if self.batch_count.item() != 0 else self.batch_count

    @overrides
    def reset(self):
        """This is called at each epoch end after compute(). It resets the state for the next epoch."""
        super().reset()
        self.batch_results *= 0
        self.batch_count *= 0
