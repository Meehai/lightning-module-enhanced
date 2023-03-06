from lightning_module_enhanced.metrics import CallableCoreMetric
from copy import deepcopy


def metric_l1(y, gt):
    res = (y - gt).abs().mean()
    return res

def test_r_metrics_deepcopy_1():
    m1 = CallableCoreMetric(metric_l1, higher_is_better=True, requires_grad=False, epoch_fn="sum")
    m2 = deepcopy(m1)

    assert m1.metric_fn == m2.metric_fn
    assert m1.epoch_fn(10, 5) == m2.epoch_fn(10, 5)
