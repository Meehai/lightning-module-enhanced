"""mean iou metric"""
from __future__ import annotations
import torch as tr
from torchmetrics.functional.classification import multiclass_stat_scores
from overrides import overrides

from ..logger import lme_logger as logger
from .core_metric import CoreMetric

# pylint: disable=not-callable
class MeanIoU(CoreMetric):
    """
    mean iou based on the multi class classification stats during training. Only epoch results, no batch.
    Modes:
    - 'global' We store a single statistic for all the elements in a batch and compute a single IoU
    - 'samplewise' We store the statistics of each element (also inside batches) and compute an IoU for each of them
    Depending on the use case, both are valid. For the Dronescapes paper, they used samplewise which may be a gotcha.
    """
    def __init__(self, classes: list[str], class_weights: list[float] | None = None, class_axis: int = -1,
                 mode: str = "global"):
        super().__init__(higher_is_better=True)
        assert mode in ("global", "samplewise"), mode
        class_weights = [1 / len(classes) for _ in range(len(classes))] if class_weights is None else class_weights
        assert abs(sum(class_weights) - 1) < 1e-3, f"Should sum to 1, got : {sum(class_weights)}"
        assert len(classes) == len(class_weights), (len(classes), classes, class_weights)
        self.classes = classes
        self.class_weights = tr.FloatTensor(class_weights)
        self.class_axis = class_axis
        self.iou_mode = mode
        self.num_classes = len(classes)
        self.batch_results: tr.Tensor | list[tr.Tensor] | None = None
        self.reset()

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
        stats = multiclass_stat_scores(y_class, gt_class, self.num_classes, average=None, multidim_average="samplewise")
        if self.iou_mode == "global":
            return stats.sum(0)[:, 0:4] # (NC, 4)
        return stats[:, :, 0:4] # (B, NC, 4)

    @overrides
    def batch_update(self, batch_result: tr.Tensor | None) -> None:
        if batch_result is None:
            return
        if self.iou_mode == "global":
            self.batch_results = self.batch_results + batch_result.detach().cpu().T
        else:
            self.batch_results.extend(batch_result.detach().cpu())

    @overrides
    def epoch_result(self) -> tr.Tensor | None:
        if ((self.iou_mode == "global" and (self.batch_results == 0).all()) or
            (self.iou_mode == "samplewise" and len(self.batch_results) == 0)):
            logger.debug(f"No batch results this epoch. Returning 0 for all {self.num_classes} classes.")
            return tr.Tensor([0] * len(self.class_weights)).to(self.device)
        if self.iou_mode == "global":
            tp, fp, _, fn = self.batch_results # 4 x (NC, )
        else:
            tp, fp, _, fn = tr.stack(self.batch_results).permute(2, 0, 1) # 4 x (epoch_N, NC)
        iou = tp / (tp + fp + fn) # (NC, ) or (epoch_N, NC)
        wmean_iou = (iou * self.class_weights).nan_to_num(0).float() # (NC, ) or (epoch_N, NC)
        res = wmean_iou if self.iou_mode == "global" else wmean_iou.mean(dim=0) # (NC, )
        return res.to(self.device)

    @overrides
    def epoch_result_reduced(self, epoch_result: tr.Tensor | None) -> tr.Tensor | None:
        return epoch_result.sum().float() # sum because it's guaranteed that class_weights.sum() == 1

    def reset(self):
        if self.iou_mode == "global":
            self.batch_results = tr.zeros(4, self.num_classes).type(tr.int64)
        else:
            self.batch_results = []

    def __repr__(self):
        return (f"[MeanIoU] Mode {self.iou_mode}. Classes: {self.classes}. "
                f"Class weights: {[round(x.item(), 2) for x in self.class_weights]}.")
