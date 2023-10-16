import torch as tr
from lightning_module_enhanced import LME
from torch import nn

def test_contructor_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    assert module is not None

def test_device_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    assert module.device == tr.device("cpu")

def test_num_params_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    assert module.num_params == 13

def test_num_trainable_params_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    assert module.num_trainable_params == 13

def test_num_trainable_params_2():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.trainable_params = False
    assert module.num_params == 13
    assert module.num_trainable_params == 0
    module.trainable_params = True
    assert module.num_params == 13
    assert module.num_trainable_params == 13

def test_trainable_params_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.trainable_params = False
    assert module.trainable_params is False
    module.trainable_params = True
    assert module.trainable_params is True

def test_set_metrics_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.metrics = {"metric1": (lambda y, gt: (y - gt).mean(), "min")}
    assert len(module.metrics) == 2 # criterion_fn is always set to some default

def test_set_metrics_before_criterion():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.criterion_fn = lambda y, gt: (y - gt).mean()
    module.metrics = {"metric1": (lambda y, gt: (y - gt).mean(), "min")}
    assert len(module.metrics) == 2

def test_set_metrics_after_criterion():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.metrics = {"metric1": (lambda y, gt: (y - gt).mean(), "min")}
    module.criterion_fn = lambda y, gt: (y - gt).mean()
    assert len(module.metrics) == 2

def test_set_criterion_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.criterion_fn = lambda y, gt: (y - gt).mean()
    assert module.criterion_fn is not None
    assert len(module.metrics) == 1


if __name__ == "__main__":
    test_set_criterion_1()