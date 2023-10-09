"""Multi class accuracy module"""
from typing import Optional
from overrides import overrides
import torch as tr
from torch.nn import functional as F
from .core_metric import CoreMetric


class MultiClassAccuracy(CoreMetric):
    """
    Multi class accuracy. Computes the accuracy of each N classes independently, based on the .argmax() result.
    It is implemented as a global metric, so we count scores for each batch and return these final ones at the end of
    the epoch.
    """

    def __init__(self, num_classes: int):
        super().__init__(higher_is_better=True)
        self.num_classes = num_classes
        self.scores = tr.zeros(num_classes)
        self.count = tr.zeros(num_classes)

    @overrides
    def forward(self, y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor:
        if gt.shape[-1] == self.num_classes:
            gt_flat = gt.reshape(-1, self.num_classes)
            gt_argmax = gt_flat.argmax(-1)
        else:
            gt_flat = F.one_hot(gt, num_classes=self.num_classes) \
                .reshape(-1, self.num_classes) # pylint: disable=not-callable
            gt_argmax = gt.reshape(-1)

        y_flat = y.reshape(-1, self.num_classes)
        y_eq_gt = y_flat.argmax(dim=-1) == gt_argmax
        cnts_correct_per_class = (gt_flat * y_eq_gt[:, None]).sum(dim=0)
        num_per_class = gt_flat.sum(dim=0)
        return cnts_correct_per_class.to(self.scores.device), num_per_class.to(self.scores.device)

    @overrides
    def batch_update(self, batch_result) -> None:
        cnts_correct_per_class, num_per_class = batch_result
        self.scores += cnts_correct_per_class
        self.count += num_per_class

    @overrides
    def epoch_result(self) -> tr.Tensor:
        accs = self.scores / self.count
        accs[tr.isnan(accs)] = 0
        return accs

    @overrides
    def epoch_result_reduced(self, epoch_result: Optional[tr.Tensor]) -> Optional[tr.Tensor]:
        """Returns the pbar-compatible result, as the average of the per-class accuracy"""
        return epoch_result.mean()

    @overrides
    def reset(self):
        self.scores *= 0
        self.count = 0

    def __str__(self):
        f_str = f"{super().__str__()}. Num classes: {self.num_classes}."
        return f_str
