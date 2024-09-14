from lightning_module_enhanced.metrics import CallableCoreMetric
from lightning_module_enhanced import LME
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch as tr
from torch import nn
from copy import deepcopy

class MyMetric(CallableCoreMetric):
    def __init__(self):
        super().__init__(lambda y, gt: (y - gt).pow(2).mean(), higher_is_better=False)

    def forward(self, y, gt):
        assert self.running_model is not None
        return super().forward(y, gt)

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return tr.randn(2), tr.randn(1)

def test_core_metric_1():
    fn = CallableCoreMetric(lambda y, gt: (y - gt).pow(2).mean(), higher_is_better=False)
    y = tr.randn(5, 3)
    gt = tr.randn(5, 3)
    res = fn(y, gt).item()
    assert isinstance(res, float)

def test_core_metric_running_model_1():
    fn = MyMetric()
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"mymetric": fn}
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    assert fn.running_model is None

def test_deepcopy_metric():
    """
    Metrics can be deepcopied properly.
    https://gitlab.com/mihaicristianpirvu/lightning-module-enhanced/-/commit/a784cc8d0fa45e2ad2f9efe4b535dc4dde542420
    """
    def metric_l1(y, gt):
        res = (y - gt).abs().mean()
        return res

    m1 = CallableCoreMetric(metric_l1, higher_is_better=True, requires_grad=False, epoch_fn="sum")
    m2 = deepcopy(m1)

    assert m1.metric_fn == m2.metric_fn
    assert m1.epoch_fn(10, 5) == m2.epoch_fn(10, 5)

def test_core_metric_with_nones():
    fn = CallableCoreMetric(lambda y, gt: (y - gt).pow(2).mean(), higher_is_better=False)
    y = tr.randn(5, 3)
    gt = tr.randn(5, 3)
    fn.batch_update(None)
    assert fn.batch_count is None and fn.batch_results is None
    batch_result = [(_y - gt).pow(2).mean() for _y, _gt in zip(y, gt)]
    batch_result[4] = None
    fn.batch_update(batch_result)
    assert fn.batch_count.item() == 4 and fn.batch_results is not None

if __name__ == "__main__":
    test_core_metric_running_model_1()
