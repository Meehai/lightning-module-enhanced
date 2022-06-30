"""Wrapper than converts a Callable to a torchmetrics.Metric with compute_on_step=True that returns a number"""
from typing import Callable, Union
from torchmetrics import Metric
from overrides import overrides
import torch as tr

EpochFnType = Union[str, Callable[[tr.Tensor, tr.Tensor], tr.Tensor]]

class TorchMetricWrapper(Metric):
    """Wrapper for a regular callable torch metric"""
    def __init__(self, metric_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor], higher_is_better: bool = False,
                 epoch_fn: EpochFnType = "mean"):
        super().__init__()
        assert isinstance(metric_fn, Callable)
        epoch_fn = TorchMetricWrapper._get_epoch_fn(epoch_fn)
        self.metric_fn = metric_fn
        self.batch_results = None
        self.batch_count = tr.IntTensor([0])
        self.epoch_fn = epoch_fn
        self._higher_is_better = higher_is_better

    @staticmethod
    def _get_epoch_fn(epoch_fn: EpochFnType) -> Callable[[tr.Tensor, tr.Tensor], tr.Tensor]:
        """Get an actual callback"""
        if isinstance(epoch_fn, str):
            assert epoch_fn in ("mean", "sum")
            if epoch_fn == "mean":
                return lambda batch_results, batch_count: batch_results / batch_count
            if epoch_fn == "sum":
                return lambda batch_results, _: batch_results
        assert isinstance(epoch_fn, Callable)
        return epoch_fn

    @overrides(check_signature=False)
    def forward(self, preds: tr.Tensor, target: tr.Tensor) -> tr.Tensor:
        """This computes the pre-batch metric. This result is passed to `update` to update the state of the metric"""
        batch_res = self.metric_fn(preds, target)
        return batch_res

    @overrides(check_signature=False)
    def update(self, batch_res: tr.Tensor) -> None:
        """This is called at each batch end. It must be a number, so .item() works."""
        batch_res = batch_res.detach().cpu()
        if self.batch_results is None:
            self.batch_results = batch_res * 0
        self.batch_results += batch_res.detach().cpu()
        self.batch_count += 1

    @overrides
    def compute(self) -> tr.Tensor:
        """This is called at each epoch end. It must be a number, so it can be properly logged as a epoch metric."""
        if self.batch_count.item() == 0:
            return self.batch_count
        result = self.epoch_fn(self.batch_results, self.batch_count)
        return result

    @overrides
    def reset(self):
        """This is called at each epoch end after compute(). It resets the state for the next epoch."""
        super().reset()
        self.batch_results *= 0
        self.batch_count *= 0
