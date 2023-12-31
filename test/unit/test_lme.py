import torch as tr
from lightning_module_enhanced import LME
from torch import nn
from copy import deepcopy

def test_constructor_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    assert module is not None

def test_constructor_2():
    try:
        _ = LME(LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))))
        raise Exception
    except ValueError as e:
        assert str(e).find("nested") != -1
        pass

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

def test_reset_parameters_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    params = deepcopy(tuple(module.parameters()))
    module.reset_parameters()
    new_params = deepcopy(tuple(module.parameters()))
    for p1, p2 in zip(params, new_params):
        assert not tr.allclose(p1, p2)

def test_reset_parameters_2():
    """
    Same as above, but deeper. Fails for bigger values tho. Seems torch related somehow.
    """
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    for _ in range(10):
        model = nn.Sequential(model)
    module = LME(model)
    params = deepcopy(tuple(module.parameters()))
    module.reset_parameters()
    new_params = deepcopy(tuple(module.parameters()))
    for p1, p2 in zip(params, new_params):
        assert not tr.allclose(p1, p2)


if __name__ == "__main__":
    test_set_criterion_1()