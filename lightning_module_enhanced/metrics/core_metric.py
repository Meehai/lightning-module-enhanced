"""
All metrics in LME are a subclass of this class and follow logic describe below. Only epoch results are
relevant, and batch results are somehow accumulated, such that we minimize the risk of getting invalid results from
simply averaging them during training.

For example, instead of accumulating accuracies like this:
    - epoch_accuracy = (batch_accuracy_1 + batch_accuracy_2) / 2
We do:
    - epoch_accuracy = sum([[batch_accuracy_item_b1_1, ..., batch_accuracy_item_b1_n],
                            [batch_accuracy_item_b2_1, ..., batch_accuracy_item_b2_n]]) / (b1_n + b2_n)

Methods logic:
- forward: Takes one batch of (y, gt) and returns the metric result of that batch
- batch_update: Takes the batch result from forward(y, gt) and updates the internal state of the current epoch
- epoch_result: Takes the internal state of the current epoch and returns the epoch result
- epoch_result_reduced: Takes the epoch result from epoch_result() and returns a reduced variant of the epoch metric
  that can be logged by basic loggers (MLFlowLogger or TensorBoardLogger) or None. If None, then it is not logged via
  self.log() in the LME at epoch end
- reset: Resets the internal state for the next epoch
"""
from __future__ import annotations
from typing import Callable
from abc import ABC, abstractmethod
from overrides import overrides
from torch import nn
import torch as tr

from ..logger import lme_logger as logger
from ..utils import parsed_str_type

MetricFnType = Callable[[tr.Tensor, tr.Tensor], tr.Tensor]

class CoreMetric(nn.Module, ABC):
    """Generic CoreMetric for a LME."""
    def __init__(self, higher_is_better: bool, requires_grad: bool = False):
        assert isinstance(higher_is_better, bool) and isinstance(requires_grad, bool)
        super().__init__()
        self.batch_results: tr.Tensor | None = None
        self.batch_count: tr.Tensor | None = None
        self.higher_is_better: bool = higher_is_better
        self.requires_grad = requires_grad
        # By default, all metrics do not require gradients. This is updated for loss in LME.
        self.requires_grad_(requires_grad)
        # The running model. Will be None when not training and a reference to the running LME when training
        self._running_model: Callable | None = None

    @abstractmethod
    @overrides(check_signature=False)
    def forward(self, y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor:
        """Computes the batch level metric. The result is passed to `batch_update` to update the state of the metric"""

    @abstractmethod
    def batch_update(self, batch_result: tr.Tensor) -> None:
        """Updates the internal state based on the batch result from forward(y, gt)"""

    @abstractmethod
    def epoch_result(self) -> tr.Tensor:
        """Called at each epoch end from the LME. Takes the internal state and returns the epoch result"""

    @abstractmethod
    def reset(self):
        """This is called at each epoch end after compute(). It resets the state for the next epoch."""

    @property
    def mode(self) -> str:
        """compatibility with ModelCheckpoint"""
        return "max" if self.higher_is_better else "min"

    @property
    def running_model(self) -> Callable[[], "LME"] | None:
        """returns the active running model, if available (during training/testing)"""
        return self._running_model

    @running_model.setter
    def running_model(self, running_model: Callable[[], "LME"] | None):
        assert running_model is None or (
            isinstance(running_model(), nn.Module) and hasattr(running_model(), "metadata_callback")), running_model
        self._running_model = running_model

    def epoch_result_reduced(self, epoch_result: tr.Tensor | None) -> tr.Tensor | None:
        """
        Reduces a potentially complex metric (confusion matrix or multi label accuracy) into a single number.
        This is used so that other loggers, such as mlflow logger or tensorboard logger can store these without making
        any transformation (i.e. mlflow logger will sum a confusion matrix into a single number).
        By default, does nothing. Override this if needed.
        """
        if epoch_result is None:
            return None
        assert isinstance(epoch_result, tr.Tensor), f"Got {type(epoch_result)}"
        epoch_result_reduced = epoch_result.squeeze()
        shape = epoch_result_reduced.shape
        if not (len(shape) == 0 or (len(shape) == 1 and shape[-1] == 1)):
            logger.debug2(f"Metric '{self}' has a non-number reduced value (shape: {shape}). Returning None.")
            return None
        return epoch_result_reduced

    def __str__(self):
        f_str = f"[{parsed_str_type(self)}]. Mode: {self.mode}. Grad: {self.requires_grad}. " \
                f"Count: {self.batch_count.sum() if self.batch_count is not None else '0 (None)'}"
        return f_str

    def __repr__(self):
        return str(self)

    def __call__(self, *args, **kwargs) -> tr.Tensor:
        return self.forward(*args, **kwargs)
