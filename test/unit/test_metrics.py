from lightning_module_enhanced.metrics import (
    CallableCoreMetric, MultiClassF1Score, MeanIoU, MultiClassAccuracy, MultiClassConfusionMatrix)
import torch as tr
import pytest
from torch.nn import functional as F
from copy import deepcopy

def test_CallableCoreMetric_forward():
    fn = CallableCoreMetric(lambda y, gt: (y - gt).pow(2).mean(), higher_is_better=False)
    y = tr.randn(5, 3)
    gt = tr.randn(5, 3)
    res = fn(y, gt).item()
    assert isinstance(res, float)

def test_CallableCoreMetric_deepcopy():
    """
    Metrics can be deepcopied properly.
    https://gitlab.com/mihaicristianpirvu/lightning-module-enhanced/-/commit/a784cc8d0fa45e2ad2f9efe4b535dc4dde542420
    """
    def metric_l1(y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor:
        return (y - gt).abs().mean()

    m1 = CallableCoreMetric(metric_l1, higher_is_better=True, requires_grad=False, epoch_fn="sum")
    m1.device = tr.device("cuda")
    m2 = deepcopy(m1)

    assert m1.metric_fn == m2.metric_fn
    assert m1.epoch_fn(10, 5) == m2.epoch_fn(10, 5)
    assert m1.device.type == m2.device.type

def test_CallableCoreMetric_None_batch_upate():
    fn = CallableCoreMetric(lambda y, gt: (y - gt).pow(2).mean(), higher_is_better=False)
    y = tr.randn(5, 3)
    gt = tr.randn(5, 3)
    fn.batch_update(None)
    assert fn.batch_count is None and fn.batch_results is None
    batch_result = [(_y - gt).pow(2).mean() for _y, _gt in zip(y, gt)]
    batch_result[4] = None
    fn.batch_update(batch_result)
    assert fn.batch_count.item() == 4 and fn.batch_results is not None

def test_MultiClassF1Score_epoch_result():
    f1 = MultiClassF1Score(num_classes=3)
    y = tr.LongTensor([[1, 2, 0]])
    gt = tr.LongTensor([[1, 0, 0]])
    assert (f1.batch_results == 0).all()
    f1.batch_update(f1.forward(y, gt))
    f1.batch_update(f1.forward(y, y))
    epoch_f1 = f1.epoch_result().numpy()
    assert ((epoch_f1 - [0.8, 1, 0.6667]) < 1e-3).all(), epoch_f1

def test_MeanIoU_epoch_result():
    iou = MeanIoU(classes=[1,2,3])
    y = tr.LongTensor([[1, 2, 0]])
    gt = tr.LongTensor([[1, 0, 0]])
    assert (iou.batch_results == 0).all()
    iou.batch_update(iou.forward(y, gt))
    epoch_iou = iou.epoch_result()
    assert ((epoch_iou.numpy() - [0.1667, 0.3333, 0]) < 1e-3).all(), epoch_iou

    iou.reset()
    iou.batch_update(iou.forward(F.one_hot(y, 3).float(), gt))
    epoch_iou = iou.epoch_result()
    assert ((epoch_iou.numpy() - [0.1667, 0.3333, 0]) < 1e-3).all(), epoch_iou

    iou = MeanIoU(classes=[1,2,3], class_axis=1)
    iou.reset()
    iou.batch_update(iou.forward(F.one_hot(y, 3).float().permute(0, 2, 1), gt))
    epoch_iou = iou.epoch_result()
    assert ((epoch_iou.numpy() - [0.1667, 0.3333, 0]) < 1e-3).all(), epoch_iou

@pytest.mark.parametrize("metric_type", [MeanIoU, MultiClassConfusionMatrix, MultiClassAccuracy, MultiClassF1Score])
def test_existing_lme_metrics_deepcopy(metric_type):
    if metric_type == MeanIoU:
        metric = metric_type(classes=[1,2,3])
    else:
        metric = metric_type(num_classes=3)
    metric.device = tr.device("cuda")
    metric_copy = deepcopy(metric)
    assert metric.device.type == metric_copy.device.type
