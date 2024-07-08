from lightning_module_enhanced import LME
from lightning_module_enhanced.callbacks import MetadataCallback, PlotMetrics
from pytorch_lightning import Trainer
from torch import nn


def test_callbacks_good_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    assert len(module.callbacks) == 2
    assert isinstance(module.callbacks[0], MetadataCallback)


def test_callbacks_good_2():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.callbacks = [PlotMetrics()]
    assert len(module.callbacks) == 3
    assert isinstance(module.callbacks[0], MetadataCallback)


def test_callbacks_good_3():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.callbacks += [PlotMetrics()]
    assert len(module.callbacks) == 3
    assert isinstance(module.callbacks[0], MetadataCallback)


def test_callbacks_bad_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    try:
        module.callbacks = [PlotMetrics(), MetadataCallback()]
        pass
    except AssertionError:
        pass

def test_callbacks_model_ckpts():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min"),
                      "l2": (lambda y, gt: (y - gt).pow(2).mean(), "min")}
    module.trainer = Trainer()
    assert len(module.callbacks) == 3
    assert module.callbacks[-1].monitor == "loss"

if __name__ == "__main__":
    test_callbacks_good_1()
