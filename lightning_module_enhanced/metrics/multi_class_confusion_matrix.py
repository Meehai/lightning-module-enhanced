"""Multi Class Confusion Matrix"""
from overrides import overrides
from torchmetrics.functional.classification import multiclass_confusion_matrix
import torch as tr

from lightning_module_enhanced.metrics import CoreMetric


class MultiClassConfusionMatrix(CoreMetric):
    """Multi Class Confusion Matrix implementation"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.batch_results = tr.zeros(num_classes, num_classes).type(tr.LongTensor)
        self.num_classes = num_classes

    def forward(self, y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor:
        res = multiclass_confusion_matrix(y.argmax(-1), gt.argmax(-1), num_classes=self.num_classes)
        return res

    @overrides
    def batch_update(self, batch_result: tr.Tensor) -> None:
        self.batch_results += batch_result.detach().cpu()

    @overrides
    def epoch_result(self) -> tr.Tensor:
        return self.batch_results

    @overrides
    def epoch_result_reduced(self, epoch_result: tr.Tensor) -> tr.Tensor:
        return None

    @overrides
    def reset(self):
        self.batch_results *= 0

    def __str__(self):
        return f"MultiClassConfusionMatrix ({self.num_classes} classes)"
