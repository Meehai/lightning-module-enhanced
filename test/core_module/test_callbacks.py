from lightning_module_enhanced import LME
from lightning_module_enhanced.callbacks import MetadataCallback, PlotMetrics
from torch import nn


def test_callbacks_good_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    assert len(module.callbacks) == 1
    assert isinstance(module.callbacks[0], MetadataCallback)


def test_callbacks_good_2():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.callbacks = [PlotMetrics()]
    assert len(module.callbacks) == 2
    assert isinstance(module.callbacks[0], MetadataCallback)


def test_callbacks_bad_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    try:
        module.callbacks = [PlotMetrics(), MetadataCallback()]
        pass
    except AssertionError:
        pass
