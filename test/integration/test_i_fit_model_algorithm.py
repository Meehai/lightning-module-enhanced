from __future__ import annotations
from functools import partial
import torch as tr
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn, optim
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.utils import to_device, to_tensor

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return tr.randn(2), tr.randn(1)

def test_fit_model_algorithm_1():
    cnt = {"cnt": 0}

    def my_model_algo(model, batch, cnt) -> ModelAlgorithmOutput:
        cnt["cnt"] += 1
        return ((y := model(batch[0])), model.lme_metrics(y, batch[1]), *batch)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = partial(my_model_algo, cnt=cnt)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    assert cnt["cnt"] == 10

def test_fit_model_algorithm_not_include_loss():
    def my_model_algo(model, batch) -> ModelAlgorithmOutput:
        x, gt = batch[0], to_device(to_tensor(batch[1]), model.device)
        y = model.forward(x)
        res = model.lme_metrics(y, gt, include_loss=False)
        assert "loss" not in res
        res["loss"] = model.criterion_fn(y, gt)
        return y, res, x, gt

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = my_model_algo
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))

def test_fit_model_algorithm_implicit_metrics():
    def my_model_algo(model, batch) -> ModelAlgorithmOutput:
        x, gt = batch[0], to_device(to_tensor(batch[1]), model.device)
        y = model.forward(x)
        metrics = {"loss": model.criterion_fn(y, gt), "lala": (y - gt).mean()}
        return y, metrics, x, gt

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = my_model_algo
    assert set(model.metrics) == set()
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    assert set(model.metrics) == {"lala"}
    assert model.metrics_history.history.keys() == {"loss", "lala"}
    assert len(model.metrics_history["loss"]["train"]) == 1 and len(model.metrics_history["loss"]["val"]) == 0
    assert len(model.metrics_history["lala"]["train"]) == 1 and len(model.metrics_history["lala"]["val"]) == 0

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = my_model_algo
    assert set(model.metrics) == set()
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert set(model.metrics) == {"lala"}
    assert len(model.metrics_history["loss"]["train"]) == 1 and len(model.metrics_history["loss"]["val"]) == 1
    assert len(model.metrics_history["lala"]["train"]) == 1 and len(model.metrics_history["lala"]["val"]) == 1
