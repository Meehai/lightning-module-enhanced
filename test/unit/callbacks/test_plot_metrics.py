from lightning_module_enhanced import LME, TrainSetup
from lightning_module_enhanced.callbacks import PlotMetrics
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn
import torch as tr

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return {"data": tr.randn(2), "labels": tr.randn(1)}


def test_plot_metrics_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "sgd", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "metrics": [{"type": "l1"}]
    }
    TrainSetup(model, cfg)
    model.callbacks = [PlotMetrics()]
    Trainer(max_epochs=3).fit(model, DataLoader(Reader()), DataLoader(Reader()))

    assert "l1" in model.callbacks[-1].history
    assert "loss" in model.callbacks[-1].history
    assert len(model.callbacks[-1].history["l1"]["train"]) == len(model.callbacks[-1].history["l1"]["val"]) == 3
    assert len(model.callbacks[-1].history["loss"]["train"]) == len(model.callbacks[-1].history["loss"]["val"]) == 3

def test_plot_metrics_2():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "sgd", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "metrics": [{"type": "l1"}]
    }
    TrainSetup(model, cfg)
    model.callbacks = [PlotMetrics()]
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    prev = model.callbacks[-1].history["l1"]["train"][0]
    assert len(model.callbacks[-1].history["l1"]["train"]) == len(model.callbacks[-1].history["l1"]["val"]) == 1
    assert len(model.callbacks[-1].history["loss"]["train"]) == len(model.callbacks[-1].history["loss"]["val"]) == 1

    Trainer(max_epochs=3).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert len(model.callbacks[-1].history["l1"]["train"]) == len(model.callbacks[-1].history["l1"]["val"]) == 3
    assert len(model.callbacks[-1].history["loss"]["train"]) == len(model.callbacks[-1].history["loss"]["val"]) == 3

    assert model.callbacks[-1].history["l1"]["train"][0] != prev
