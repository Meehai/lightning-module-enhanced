from copy import copy
from lightning_module_enhanced import LME
from lightning_module_enhanced.callbacks import MetricsHistory
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn
import torch as tr

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return tr.randn(2), tr.randn(1)

def test_metrics_history_1():
    """simple tests: at the end of training we should have 3 entries on l1/loss due to 3 epochs"""
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    pm: MetricsHistory = model.callbacks[1]
    Trainer(max_epochs=3).fit(model, DataLoader(Reader()), DataLoader(Reader()))

    assert "l1" in pm.history
    assert "loss" in pm.history
    assert len(pm.history["l1"]["train"]) == len(pm.history["l1"]["val"]) == 3
    assert len(pm.history["loss"]["train"]) == len(pm.history["loss"]["val"]) == 3

def test_metrics_history_2():
    """fine-tuning also should yield 3 epochs, even though we start from a pre-trained one"""
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    pm: MetricsHistory = model.callbacks[1]
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    prev = copy(pm.history["l1"]["train"][0])
    assert len(pm.history["l1"]["train"]) == len(pm.history["l1"]["val"]) == 1
    assert len(pm.history["loss"]["train"]) == len(pm.history["loss"]["val"]) == 1

    Trainer(max_epochs=3).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert len(pm.history["l1"]["train"]) == len(pm.history["l1"]["val"]) == 3
    assert len(pm.history["loss"]["train"]) == len(pm.history["loss"]["val"]) == 3

    assert pm.history["l1"]["train"][0] != prev


def test_metrics_history_3():
    """reload a training from first/2nd epoch. The metrics/training should continue"""
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    pm: MetricsHistory = model.callbacks[1]
    Trainer(max_epochs=2).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    prev = copy(pm.history["l1"]["train"][0])
    assert len(pm.history["l1"]["train"]) == len(pm.history["l1"]["val"]) == 2
    assert len(pm.history["loss"]["train"]) == len(pm.history["loss"]["val"]) == 2

    Trainer(max_epochs=5).fit(model, DataLoader(Reader()), DataLoader(Reader()),
                              ckpt_path=model.trainer.checkpoint_callback.best_model_path)
    assert len(pm.history["l1"]["train"]) == len(pm.history["l1"]["val"]) == 5
    assert len(pm.history["loss"]["train"]) == len(pm.history["loss"]["val"]) == 5

    assert pm.history["l1"]["train"][0] == prev

if __name__ == "__main__":
    test_metrics_history_3()
