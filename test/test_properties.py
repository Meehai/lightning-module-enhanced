import pytest
import torch as tr
from lightning_module_enhanced import LightningModuleEnhanced
from torch import nn

class TestProperties:
    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        self.module = LightningModuleEnhanced(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
        yield

    def test_contructor_1(self):
        assert self.module is not None

    def test_device_1(self):
        assert self.module.device == tr.device("cpu")

    def test_num_params_1(self):
        assert self.module.num_params == 13

    def test_num_trainable_params_1(self):
        assert self.module.num_trainable_params == 13

    def test_num_trainable_params_2(self):
        self.module.trainable_params = False
        assert self.module.num_params == 13
        assert self.module.num_trainable_params == 0
        self.module.trainable_params = True
        assert self.module.num_params == 13
        assert self.module.num_trainable_params == 13

    def test_trainable_params_1(self):
        self.module.trainable_params = False
        assert self.module.trainable_params == False
        self.module.trainable_params = True
        assert self.module.trainable_params == True
