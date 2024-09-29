"""mean iou metric"""
from __future__ import annotations
import torch as tr
from torchmetrics.functional.classification import multiclass_stat_scores
from overrides import overrides

from ..logger import lme_logger as logger
from .core_metric import CoreMetric

# pylint: disable=not-callable
class MeanIoU(CoreMetric):
    """mean iou based on the multi class classification stats during training. Only epoch results, no batch."""
    def __init__(self, classes: list[str], class_weights: list[float] | None = None, class_axis: int = -1):
        super().__init__(higher_is_better=True)
        class_weights = [1 / len(classes) for _ in range(len(classes))] if class_weights is None else class_weights
        assert len(classes) == len(class_weights), (len(classes), classes, class_weights)
        self.classes = classes
        self.class_weights = tr.FloatTensor(class_weights)
        self.class_axis = class_axis
        assert (self.class_weights.sum() - 1).abs() < 1e-3, (self.class_weights.sum())
        self.num_classes = len(classes)
        self.batch_results = tr.zeros(4, self.num_classes).type(tr.float64)

    def _get_class_tensor(self, tensor: tr.Tensor) -> tr.Tensor:
        assert tensor.dtype in (tr.int64, tr.float32), tensor.dtype
        assert not tensor.isnan().any(), f"Tensor {tensor} has NaNs!"
        if tensor.dtype == tr.float32:
            if tensor.shape[self.class_axis] != self.num_classes:
                raise ValueError(f"Expected {self.num_classes} classes on axis {self.class_axis}, got {tensor.shape}")
            tensor = tensor.argmax(self.class_axis)
        return tensor

    def forward(self, y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor | None:
        if len(y) == 0:
            return None
        y_class = self._get_class_tensor(y)
        gt_class = self._get_class_tensor(gt)
        stats = multiclass_stat_scores(y_class, gt_class, num_classes=self.num_classes, average=None)
        return stats[:, 0:4].T # TP, FP, TN, FN

    @overrides
    def batch_update(self, batch_result: tr.Tensor | None) -> None:
        if batch_result is None:
            return
        self.batch_results = self.batch_results + batch_result.detach()

    @overrides
    def epoch_result(self) -> tr.Tensor | None:
        if (self.batch_results == 0).all():
            logger.debug(f"No batch results this epoch. Returning 0 for all {self.num_classes} classes.")
            return tr.Tensor([0] * len(self.class_weights)).to(self.device)
        tp, fp, _, fn = self.batch_results
        iou = tp / (tp + fp + fn) # (NC, )
        wmean_iou = (iou * self.class_weights).nan_to_num(0).float() # (NC, )
        return wmean_iou

    @overrides
    def epoch_result_reduced(self, epoch_result: tr.Tensor | None) -> tr.Tensor | None:
        return epoch_result.sum().float() # sum because it's guaranteed that class_weights.sum() == 1

    @overrides
    def to(self, device: tr.device | str) -> MeanIoU:
        self.device = device
        self.batch_results = self.batch_results.to(device)
        self.class_weights = self.class_weights.to(device)
        return self

    def reset(self):
        self.batch_results *= 0

    def __repr__(self):
        return f"[MeanIoU] Classes: {self.classes} Class weights: {[round(x.item(), 2) for x in self.class_weights]}"
