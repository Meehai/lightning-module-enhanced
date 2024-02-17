import torch as tr
from torch import nn
from torch.utils.data import DataLoader
from lightning_module_enhanced import LME
from lightning_module_enhanced.metrics import CallableCoreMetric
from pytorch_lightning.trainer import Trainer

def metric_grad(y, gt):
    res = (y - gt).abs().mean()
    counters["metric_grad"] += res.requires_grad
    return res

def metric_non_grad(y, gt):
    res = (y - gt).abs().mean()
    counters["metric_non_grad"] += res.requires_grad
    return res

counters = {"metric_grad": 0, "metric_non_grad": 0}

class TrainReader:
    def __getitem__(self, ix):
        return {"data": tr.randn(3, 10), "labels": tr.randn(3, 3)}

    def __len__(self):
        return 5

def test_metrics_requires_grad():
    module = LME(nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 3)))
    module.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    module.metrics = {
        "metric1": CallableCoreMetric(metric_grad, higher_is_better=False, requires_grad=True),
        "metric2": (metric_non_grad, "min"),
    }
    module.optimizer = tr.optim.SGD(module.parameters(), lr=0.01)

    Trainer(max_epochs=10).fit(module, DataLoader(TrainReader()))
    assert counters["metric_grad"] == 50
    assert counters["metric_non_grad"] == 0

