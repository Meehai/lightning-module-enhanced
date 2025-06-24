# pylint: disable=all
# flake8: noqa
import pytest
from lightning_module_enhanced import LME
from lightning_module_enhanced.metrics import CallableCoreMetric
from lightning_module_enhanced.callbacks import PlotMetrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch as tr

class Reader(Dataset):
    def __init__(self, d_in: int, d_out: int, n: int = 100):
        self.x = tr.randn(n, d_in)
        self.gt = tr.randn(n, d_out)
    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]
    def __len__(self):
        return len(self.x)

class EpikMetric(CallableCoreMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
    def forward(self, y, gt):
        return None
    def batch_update(self, batch_result) -> None:
        pass
    def epoch_result(self) -> tr.Tensor:
        if self.epoch <= 5:
            self.epoch += 1
            res = self.epoch
        else:
            res = 1
        return tr.FloatTensor([res]).to(self.device)
    def reset(self):
        pass

def test_checkpoint_callback_basic():
    model = LME(nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 3)))
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.checkpoint_monitors = ["loss"]

    with pytest.raises(RuntimeError) as e:
        _ = model.checkpoint_callback
    assert "LightningModuleEnhanced is not attached to a `Trainer`." in str(e)

    Trainer(max_epochs=2).fit(model, train_dataloaders=DataLoader(Reader(10, 3, n=10)))
    assert model.checkpoint_callback.monitor == "loss"

    Trainer(max_epochs=2).fit(model, train_dataloaders=DataLoader(Reader(10, 3, n=10)),
                              val_dataloaders=DataLoader(Reader(10, 3, n=10)))
    assert model.checkpoint_callback.monitor == "val_loss"

def test_checkpoint_callback_checkpoint_monitors():
    model = LME(nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 3)))
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.metrics = {"epik_metric": EpikMetric(lambda y, gt: (y - gt) ** 2, higher_is_better=True)}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.callbacks = [PlotMetrics()]
    model.checkpoint_monitors = ["epik_metric", "loss"]

    with pytest.raises(RuntimeError) as e:
        _ = model.checkpoint_callback
    assert "LightningModuleEnhanced is not attached to a `Trainer`." in str(e)

    Trainer(max_epochs=2).fit(model, train_dataloaders=DataLoader(Reader(10, 3, n=10)))
    assert model.checkpoint_callback.monitor == "epik_metric"

    Trainer(max_epochs=2).fit(model, train_dataloaders=DataLoader(Reader(10, 3, n=10)),
                              val_dataloaders=DataLoader(Reader(10, 3, n=10)))
    assert model.checkpoint_callback.monitor == "val_epik_metric"

def test_checkpoint_callback_checkpoint_monitors_and_explicit_non_val():
    model = LME(nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 3)))
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.metrics = {"epik_metric": EpikMetric(lambda y, gt: (y - gt) ** 2, higher_is_better=True)}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.callbacks = [PlotMetrics(), ModelCheckpoint(monitor="epik_metric")] # note: non_val is explicit!
    model.checkpoint_monitors = ["epik_metric", "loss"]

    with pytest.raises(RuntimeError) as e:
        _ = model.checkpoint_callback
    assert "LightningModuleEnhanced is not attached to a `Trainer`." in str(e)

    Trainer(max_epochs=2).fit(model, train_dataloaders=DataLoader(Reader(10, 3, n=10)))
    assert model.checkpoint_callback.monitor == "epik_metric"

    Trainer(max_epochs=2).fit(model, train_dataloaders=DataLoader(Reader(10, 3, n=10)),
                              val_dataloaders=DataLoader(Reader(10, 3, n=10)))
    assert model.checkpoint_callback.monitor == "epik_metric" # overwrites the default behaviour
