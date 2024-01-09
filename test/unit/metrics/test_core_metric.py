from lightning_module_enhanced.metrics import CallableCoreMetric
from lightning_module_enhanced import LME
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch as tr
from torch import nn

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
        return {"data": tr.randn(2), "labels": tr.randn(1)}

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
    model.metrics = {"mymetric": fn}
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    assert fn.running_model is None

if __name__ == "__main__":
    test_core_metric_running_model_1()
