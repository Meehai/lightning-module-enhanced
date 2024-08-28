from functools import partial
from pathlib import Path
import shutil
from torch import nn, optim
from torch.nn import functional as F
import pytest
import torch as tr
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.metrics import CallableCoreMetric
from lightning_module_enhanced.utils import get_project_root

class Reader(Dataset):
    def __init__(self):
        self.x = tr.randn(100, 2)
        self.gt = tr.randn(100, 1)

    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]

    def __len__(self):
        return len(self.x)

def test_load_metrics_metadata():
    """Fixes this: https://gitlab.com/meehai/lightning-module-enhanced/-/issues/11"""
    train_loader = DataLoader(Reader(), batch_size=10)
    val_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (log_dir_name := "load_metrics_metadata"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.criterion_fn = F.mse_loss
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)

    pl_logger = CSVLogger(get_project_root() / "test/logs", name=log_dir_name, version=0)
    t1 = Trainer(max_epochs=3, logger=pl_logger)
    t1.fit(model, train_loader, val_loader)

    pl_logger2 = CSVLogger(get_project_root() / "test/logs", name=log_dir_name, version=1)
    ckpt_path = get_project_root() / "test/logs" / log_dir_name / "version_0" / "checkpoints" / "last.ckpt"
    t2 = Trainer(max_epochs=6, logger=pl_logger2)
    t2.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 6, \
        model.metadata_callback.metadata["epoch_metrics"]
    assert (get_project_root() / "test/logs" / log_dir_name / "version_1" / "checkpoints" / "loaded.ckpt").exists()
    assert (get_project_root() / "test/logs" / log_dir_name / "version_1" / "checkpoints" / ckpt_path.name).exists()
    shutil.rmtree(get_project_root() / "test/logs" / log_dir_name, ignore_errors=True)

def test_load_implicit_metrics():
    def _model_algorithm(model, batch) -> ModelAlgorithmOutput:
        x, gt = batch
        y = model(x)
        metrics = {"loss": F.mse_loss(y, gt), "my_metric": (y - gt).abs().mean()}
        return y, metrics, x, gt

    train_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_load_implicit_metrics"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.model_algorithm = _model_algorithm

    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)
    t1 = Trainer(max_epochs=3, logger=pl_logger)
    t1.fit(model, train_loader)

    model2 = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model2.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model2.model_algorithm = _model_algorithm
    pl_logger2 = CSVLogger(get_project_root() / "test/logs", name=logdir, version=1)
    ckpt_path = get_project_root() / "test/logs" / logdir / "version_0" / "checkpoints" / "last.ckpt"
    t2 = Trainer(max_epochs=6, logger=pl_logger2)
    t2.fit(model2, train_loader, ckpt_path=ckpt_path)

def test_implicit_core_metrics():
    def _model_algorithm(model, batch, epik_metric) -> ModelAlgorithmOutput:
        x, gt = batch
        y = model(x)
        metrics = {"loss": F.mse_loss(y, gt), "epik_metric": epik_metric(y, gt)}
        return y, metrics, x, gt

    class EpikMetric(CallableCoreMetric):
        def forward(self, y, gt):
            return self
        def batch_update(self, batch_result) -> None:
            pass

    train_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_load_implicit_metrics"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    epik_metric = EpikMetric(lambda y, gt: (y - gt) ** 2, higher_is_better=True)
    model.model_algorithm = partial(_model_algorithm, epik_metric=epik_metric)
    model.checkpoint_monitors = ["epik_metric", "loss"]

    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)
    t1 = Trainer(max_epochs=3, logger=pl_logger)
    t1.fit(model, train_loader)
    assert id(epik_metric) == id(model.metrics["epik_metric"])
    assert t1.callbacks[-1].monitor == "loss" and t1.callbacks[-1].mode == "min"
    assert t1.callbacks[-2].monitor == "epik_metric" and t1.callbacks[-2].mode == "max"

def test_implicit_metrics_and_checkpoint_monitors():
    def _model_algorithm(model, batch) -> ModelAlgorithmOutput:
        x, gt = batch
        y = model(x)
        metrics = {"loss": F.mse_loss(y, gt), "my_metric": (y - gt).abs().mean()}
        return y, metrics, x, gt

    train_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs/" / (logdir := "test_implicit_metrics_and_checkpoint_monitors"),
                  ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.model_algorithm = _model_algorithm
    model.checkpoint_monitors = ["my_metric_wrong", "loss"]
    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)
    with pytest.raises(ValueError):
        Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader)
    model.checkpoint_monitors = ["my_metric", "loss"]
    (t1 := Trainer(max_epochs=3, logger=pl_logger)).fit(model, train_loader)
    assert t1.callbacks[-2].monitor == "my_metric" and Path(t1.callbacks[-2].best_model_path).exists()

if __name__ == "__main__":
    test_implicit_metrics_and_checkpoint_monitors()
