# pylint: disable=all
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
    assert module.is_trainable_model is False
    module.trainable_params = True
    assert module.is_trainable_model is True

def test_is_parametric_model_1():
    class NonParametricModule(nn.Module):
        def forward(self, x: tr.Tensor) -> tr.Tensor:
            return x ** 2
    module = LME(NonParametricModule())
    assert not module.is_parametric_model
    assert not module.is_trainable_model
    assert module.num_params == 0

def test_set_metrics_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.metrics = {"metric1": (lambda y, gt: (y - gt).mean(), "min")}
    assert len(module.metrics) == 1

def test_set_metrics_before_criterion():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.criterion_fn = lambda y, gt: (y - gt).mean()
    module.metrics = {"metric1": (lambda y, gt: (y - gt).mean(), "min")}
    assert len(module.metrics) == 1

def test_set_metrics_after_criterion():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.metrics = {"metric1": (lambda y, gt: (y - gt).mean(), "min")}
    module.criterion_fn = lambda y, gt: (y - gt).mean()
    assert len(module.metrics) == 1

def test_set_criterion_1():
    module = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    module.criterion_fn = lambda y, gt: (y - gt).mean()
    assert module.criterion_fn is not None
    assert len(module.metrics) == 0

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

def test_reset_parameters_3():
    """
    Regression test.
    https://gitlab.com/mihaicristianpirvu/lightning-module-enhanced/-/commit/0448daa18cc4414dc7377c3dfa8cb58b0e83e747
    This tests that if we have parameters(), but no reset_parameters(), then we'll try to recursively call
    reset_parameters(), by first converting the model to a LME.
    """
    module = LME(nn.Sequential(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))))
    params = deepcopy(tuple(module.parameters()))
    module.reset_parameters()
    new_params = deepcopy(tuple(module.parameters()))
    for p1, p2 in zip(params, new_params):
        assert not tr.allclose(p1, p2)


def test_model_algorithm_no_trainer_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    x = tr.randn(10, 2), tr.randn(10, 1)
    y, metrics, _, _ = model.model_algorithm(model, x)
    assert y.shape == (10, 1)
    assert metrics.keys() == {"l1", "loss"}
    assert metrics["l1"].grad_fn is None and isinstance(metrics["l1"].item(), float)
    assert metrics["loss"].grad_fn is not None and isinstance(metrics["loss"].item(), float)

if __name__ == "__main__":
    test_trainable_params_1()
