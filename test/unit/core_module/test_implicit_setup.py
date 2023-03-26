from lightning_module_enhanced import LME
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn, optim
import torch as tr


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

    def forward(self, x):
        return self.sequential(x)


class BaseModelPlusCriterion(BaseModel):
    @property
    def criterion_fn(self):
        return lambda y, gt: (y - gt).pow(2).mean()


class BaseModelPlusCriterionOptimizer(BaseModelPlusCriterion):
    @property
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.01)


class BaseModelPlusCriterionOptimizerAndBadScheduler(BaseModelPlusCriterionOptimizer):
    @property
    def scheduler_dict(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)


class BaseModelPlusCriterionOptimizerAndScheduler(BaseModelPlusCriterionOptimizer):
    @property
    def scheduler_dict(self):
        return {"scheduler": optim.lr_scheduler.ReduceLROnPlateau(self.optimizer), "monitor": "val_loss"}


def test_implicit_nothing():
    model = LME(BaseModel())
    assert len(model.metrics) == 0
    assert model.criterion_fn is None
    assert model.optimizer is None
    assert model.scheduler_dict is None


def test_implicit_criterion_fn():
    model = LME(BaseModelPlusCriterion())
    assert model.optimizer is None
    assert model.scheduler_dict is None
    assert model.criterion_fn is not None
    assert len(model.metrics) == 1 and "loss" in model.metrics.keys()


def test_implicit_optimizer():
    model = LME(BaseModelPlusCriterionOptimizer())
    assert model.scheduler_dict is None
    assert model.criterion_fn is not None
    assert len(model.metrics) == 1 and "loss" in model.metrics.keys()
    assert model.optimizer is not None


def test_implicit_scheduler():
    model = LME(BaseModelPlusCriterionOptimizerAndScheduler())
    assert model.scheduler_dict is not None
    assert model.criterion_fn is not None
    assert len(model.metrics) == 1 and "loss" in model.metrics.keys()
    assert model.optimizer is not None


def test_implicit_scheduler_bad():
    try:
        model = LME(BaseModelPlusCriterionOptimizerAndBadScheduler())
    except AssertionError:
        pass
