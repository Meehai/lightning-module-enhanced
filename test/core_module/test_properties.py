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

    def test_set_metrics_1(self):
        self.module.metrics = {
            "metric1": lambda y, gt: (y - gt).mean()
        }
        assert len(self.module.metrics) == 1

    def test_set_metrics_before_criterion(self):
        self.module.criterion_fn = lambda y, gt: (y - gt).mean()
        self.module.metrics = {
            "metric1": lambda y, gt: (y - gt).mean()
        }
        assert len(self.module.metrics) == 1

    def test_set_metrics_after_criterion(self):
        self.module.metrics = {
            "metric1": lambda y, gt: (y - gt).mean()
        }
        self.module.criterion_fn = lambda y, gt: (y - gt).mean()
        assert len(self.module.metrics) == 1

    def test_set_criterion_1(self):
        self.module.criterion_fn = lambda y, gt: (y - gt).mean()
        assert self.module.criterion_fn is not None
        assert len(self.module.metrics) == 0

if __name__ == "__main__":
    p =TestProperties()
    p.module = LightningModuleEnhanced(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    p.test_set_criterion_1()
