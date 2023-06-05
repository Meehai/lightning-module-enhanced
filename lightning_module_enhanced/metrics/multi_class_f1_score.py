"""Multi class F1 Score"""
from typing import Optional
from overrides import overrides
import torch as tr
from torchmetrics.functional.classification import multiclass_stat_scores

from .core_metric import CoreMetric


class MultiClassF1Score(CoreMetric):
    """Multi class F1 Score implementation"""

    def __init__(self, num_classes: int):
        super().__init__(higher_is_better=True)
        self.num_classes = num_classes
        self.batch_results = tr.zeros(4, num_classes).type(tr.DoubleTensor)

    @overrides
    def forward(self, y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor:
        # support for both index tensors as well as float gt tensors (if one_hot in dataset)
        gt_argmax = gt.argmax(-1) if gt.dtype == tr.float else gt
        y_argmax = y.argmax(-1) if y.dtype == tr.float else y
        stats = multiclass_stat_scores(y_argmax, gt_argmax, num_classes=self.num_classes, average="none")
        # TP, FP, TN, FN
        res = stats[:, 0:4].T
        return res

    @overrides
    def batch_update(self, batch_result: tr.Tensor) -> None:
        self.batch_results.to(batch_result.device)
        self.batch_results += batch_result.detach().cpu()

    @overrides
    def epoch_result(self) -> tr.Tensor:
        tp, fp, _, fn = self.batch_results
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        f1[tr.isnan(f1)] = 0
        return f1

    @overrides
    def epoch_result_reduced(self, epoch_result: Optional[tr.Tensor]) -> Optional[tr.Tensor]:
        """One f1 score per class => average of all f1 scores"""
        return epoch_result.mean()

    @overrides
    def reset(self):
        self.batch_results *= 0

    def __str__(self):
        return f"MultiClassF1Score ({self.num_classes} classes)"
