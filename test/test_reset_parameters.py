import pytest
import torch as tr
from copy import deepcopy
from lightning_module_enhanced import LightningModuleEnhanced
from torch import nn

def test_reset_parameters_1():
    module = LightningModuleEnhanced(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    params = deepcopy(tuple(module.parameters()))
    module.reset_parameters()
    new_params = deepcopy(tuple(module.parameters()))
    for p1, p2 in zip(params, new_params):
        assert not tr.allclose(p1, p2)

def test_reset_parameters_2():
    module = LightningModuleEnhanced(nn.Sequential(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))))
    params = deepcopy(tuple(module.parameters()))
    module.reset_parameters()
    new_params = deepcopy(tuple(module.parameters()))
    for p1, p2 in zip(params, new_params):
        assert not tr.allclose(p1, p2)
