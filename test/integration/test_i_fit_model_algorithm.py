from __future__ import annotations
import pytest
from functools import partial
import torch as tr
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.metrics import CallableCoreMetric

class Reader(Dataset):
    def __init__(self, d_in: int, d_out: int, n: int = 100):
        self.x = tr.randn(n, d_in)
        self.gt = tr.randn(n, d_out)
    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]
    def __len__(self):
        return len(self.x)

def test_fit_model_algorithm_1():
    cnt = {"cnt": 0}

    def my_model_algo(model, batch, cnt) -> ModelAlgorithmOutput:
        cnt["cnt"] += 1
        return ((y := model(batch[0])), model.lme_metrics(y, batch[1]), *batch)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = partial(my_model_algo, cnt=cnt)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))
    assert cnt["cnt"] == 10

def test_fit_model_algorithm_not_include_loss():
    def my_model_algo(model, batch) -> ModelAlgorithmOutput:
        x, gt = batch
        y = model.forward(x)
        res = model.lme_metrics(y, gt, include_loss=False)
        assert "loss" not in res
        res["loss"] = model.criterion_fn(y, gt)
        return y, res, x, gt

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = my_model_algo
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_fit_model_algorithm_implicit_metrics(): # (!30) implicit metrics are no longer supported because very buggy
    def my_model_algo(model, batch) -> ModelAlgorithmOutput:
        x, gt = batch
        y = model.forward(x)
        metrics = {"loss": model.criterion_fn(y, gt), "lala": (y - gt).mean()}
        return y, metrics, x, gt

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = my_model_algo
    with pytest.raises(ValueError) as exc:
        Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))
    assert f"{exc.value}" == "Expected metrics: ['loss'] vs. this batch: ['lala', 'loss']"

def test_fit_model_algorithm_with_Nones():
    i, j = 0, 0
    class MyMetric(CallableCoreMetric):
        def __init__(self):
            super().__init__(lambda y, gt: (y - gt).pow(2).mean(), higher_is_better=False)
        def forward(self, y: tr.Tensor, gt: tr.Tensor):
            nonlocal j
            j += 1
            return super().forward(y, gt)

    def my_model_algo(model: LME, batch: dict) -> ModelAlgorithmOutput:
        nonlocal i
        x, gt = batch
        y = model.forward(x)
        i += 1
        if i % 2 == 0:
            return None
        metrics = model.lme_metrics(y, gt)
        return y, metrics, x, gt
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.metrics = {"my_metric": MyMetric()}
    model.model_algorithm = my_model_algo
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))
    assert i == 10 and j == 5, (i, j)

