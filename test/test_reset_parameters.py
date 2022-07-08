import pytest
import torch as tr
from copy import deepcopy
from lightning_module_enhanced import LightningModuleEnhanced
from torch import nn

class TestProperties:
    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        self.module = LightningModuleEnhanced(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
        yield

    def test_reset_parameters(self):
        params = deepcopy(tuple(self.module.parameters()))
        self.module.reset_parameters()
        new_params = deepcopy(tuple(self.module.parameters()))
        for p1, p2 in zip(params, new_params):
            assert not tr.allclose(p1, p2)
    