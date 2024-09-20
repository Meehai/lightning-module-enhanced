from __future__ import annotations
from functools import partial
import shutil
from torch import nn, optim
from torch.nn import functional as F
import pytest
import torch as tr
import csv
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.metrics import CallableCoreMetric, CoreMetric
from lightning_module_enhanced.utils import get_project_root
from lightning_module_enhanced.callbacks import PlotMetrics

class Reader(Dataset):
    def __init__(self, d_out: int = 1, length: int = 100):
        self.x = tr.randn(length, 2)
        self.gt = tr.randn(length, d_out)

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

def test_load_implicit_metrics(): # (!30) implicit metrics are no longer supported because very buggy
    def _model_algorithm(model, batch: tuple[tr.Tensor, tr.Tensor]) -> ModelAlgorithmOutput:
        x, gt = batch
        y: tr.Tensor = model(x)
        metrics = {"loss": F.mse_loss(y, gt), "my_metric": (y - gt).abs().mean()}
        return y, metrics, x, gt

    train_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_load_implicit_metrics"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.model_algorithm = _model_algorithm
    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)

    with pytest.raises(ValueError) as exc:
        Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader)
    assert f"{exc.value}" == "Expected metrics: ['loss'] vs. this batch: ['loss', 'my_metric']"

def test_implicit_core_metrics(): # (!30) implicit metrics are no longer supported because very buggy
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
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_implicit_core_metrics"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    epik_metric = EpikMetric(lambda y, gt: (y - gt) ** 2, higher_is_better=True)
    model.model_algorithm = partial(_model_algorithm, epik_metric=epik_metric)

    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)
    with pytest.raises(ValueError) as exc:
        Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader)
    assert f"{exc.value}" == "Expected metrics: ['loss'] vs. this batch: ['epik_metric', 'loss']"

def test_metrics_from_algorithm(): # Pasing metrics from model_algorithm is fine as long as they were defined before
    def _model_algorithm(model, batch: tuple[tr.Tensor, tr.Tensor]) -> ModelAlgorithmOutput:
        x, gt = batch
        y: tr.Tensor = model(x)
        metrics = {"loss": F.mse_loss(y, gt), "my_metric": (y - gt).abs().mean()}
        return y, metrics, x, gt

    train_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_metrics_from_algorithm"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.model_algorithm = _model_algorithm
    with pytest.raises(AssertionError):
        model.metrics = {"my_metric": None}
    model.metrics = {"my_metric": (None, "min")}
    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)

    Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader)
    metrics_csv = list(csv.DictReader(open(f"{pl_logger.log_dir}/metrics.csv", "r")))
    assert len(metrics_csv) == 3 and "my_metric" in metrics_csv[0] and "loss" in metrics_csv[0]

def test_epoch_metric():
    class MyEpochMetric(CoreMetric):
        def __init__(self):
            super().__init__(higher_is_better=False)
            self.reset()
        def forward(self, y: tr.Tensor, gt: tr.Tensor):
            self.batch_results.extend((y - gt).abs().mean(dim=1))
        def batch_update(self, batch_result: F.Tensor) -> None:
            pass
        def epoch_result(self) -> F.Tensor:
            return sum(self.batch_results) / len(self.batch_results)
        def reset(self):
            self.batch_results = []

    train_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_epoch_metric"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min"), "my_epoch_metric": MyEpochMetric()}
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)

    Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader)
    metrics_csv = list(csv.DictReader(open(f"{pl_logger.log_dir}/metrics.csv", "r")))
    assert len(metrics_csv) == 3 and "my_epoch_metric" in metrics_csv[0] and "loss" in metrics_csv[0]

def test_epoch_metric_reduced():
    class MyEpochMetric(CoreMetric):
        def __init__(self):
            super().__init__(higher_is_better=False)
            self.reset()
        def forward(self, y: tr.Tensor, gt: tr.Tensor):
            return (y - gt).abs()
        def batch_update(self, batch_result: F.Tensor) -> None:
            self.batch_results.extend(batch_result)
        def epoch_result(self) -> F.Tensor:
            return sum(self.batch_results) / len(self.batch_results)
        def epoch_result_reduced(self, epoch_result: F.Tensor | None) -> F.Tensor | None:
            return sum(epoch_result) / len(epoch_result)
        def reset(self):
            self.batch_results = []

    train_loader = DataLoader(Reader(d_out=5), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_epoch_metric_reduced"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 5)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min"), "my_epoch_metric": MyEpochMetric()}
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)

    Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader)
    metrics_csv = list(csv.DictReader(open(f"{pl_logger.log_dir}/metrics.csv", "r")))
    assert len(metrics_csv) == 3 and "my_epoch_metric" in metrics_csv[0] and "loss" in metrics_csv[0]

def test_epoch_metric_reduced_val():
    class MyEpochMetric(CoreMetric):
        def __init__(self):
            super().__init__(higher_is_better=False)
            self.batch_count = []
            self.reset()
        def forward(self, y: tr.Tensor, gt: tr.Tensor):
            self.batch_count[-1] += len(y)
        def batch_update(self, batch_result: F.Tensor) -> None:
            pass
        def epoch_result(self) -> tr.Tensor:
            return tr.Tensor([0]).to(self.running_model().device)
        def reset(self):
            self.batch_count.append(0)

    train_loader = DataLoader(Reader(d_out=5), batch_size=10)
    val_loader = DataLoader(Reader(d_out=5, length=30), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_epoch_metric_reduced"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 5)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min"), "my_epoch_metric": MyEpochMetric()}
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)

    Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader, val_loader)
    assert model.metrics["my_epoch_metric"].batch_count == [100, 100, 100, 0] # used to use same object for val too!!!

def test_CoreMetric_higher_is_better():
    def _model_algorithm(model, batch) -> ModelAlgorithmOutput:
        x, gt = batch
        y = model(x)
        metrics = {**model.lme_metrics(y, gt, include_loss=False), "loss": F.mse_loss(y, gt)}
        return y, metrics, x, gt

    class EpikMetric(CallableCoreMetric):
        def forward(self, y, gt):
            return None
        def batch_update(self, batch_result) -> None:
            pass
        def epoch_result(self) -> F.Tensor:
            return tr.FloatTensor([self.running_model().trainer.current_epoch]).to(self.running_model().device)

    train_loader = DataLoader(Reader(), batch_size=10)
    shutil.rmtree(get_project_root() / "test/logs" / (logdir := "test_implicit_core_metrics"), ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.metrics = {"epik_metric": EpikMetric(lambda y, gt: (y - gt) ** 2, higher_is_better=True)}
    model.model_algorithm = _model_algorithm
    model.callbacks = [PlotMetrics()]
    model.checkpoint_monitors = ["epik_metric", "loss"]

    pl_logger = CSVLogger(get_project_root() / "test/logs", name=logdir, version=0)
    Trainer(max_epochs=3, logger=pl_logger).fit(model, train_loader)
    cb = [cb for cb in model.trainer.callbacks if isinstance(cb, ModelCheckpoint) and cb.monitor == "epik_metric"]
    assert len(cb) == 1
    cb = cb[0]
    assert cb.mode == "max", cb.mode

def test_metrics_history_1():
    """simple tests: at the end of training we should have 3 entries on l1/loss due to 3 epochs"""
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    pl_logger = CSVLogger(get_project_root() / "test/logs", name="test_metrics_history_1", version=0)
    Trainer(max_epochs=3, logger=pl_logger).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    metrics_csv = list(csv.DictReader(open(f"{pl_logger.log_dir}/metrics.csv", "r")))
    assert len(metrics_csv) == 3
    assert "l1" in metrics_csv[0] and "loss" in metrics_csv[0]
    assert "val_l1" in metrics_csv[0] and "val_loss" in metrics_csv[0]

def test_metrics_history_2():
    """fine-tuning also should yield 3 epochs, even though we start from a pre-trained one"""
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}

    pl_logger = CSVLogger(get_project_root() / "test/logs", name="test_metrics_history_2", version=0)
    Trainer(max_epochs=1, logger=pl_logger).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert len(list(csv.DictReader(open(f"{pl_logger.log_dir}/metrics.csv", "r")))) == 1

    pl_logger2 = CSVLogger(get_project_root() / "test/logs", name="test_metrics_history_2", version=1)
    Trainer(max_epochs=3, logger=pl_logger2).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert len(list(csv.DictReader(open(f"{pl_logger2.log_dir}/metrics.csv", "r")))) == 3

def test_metrics_history_3():
    """reload a training from first/2nd epoch. The metrics/training should continue"""
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}

    pl_logger = CSVLogger(get_project_root() / "test/logs", name="test_metrics_history_3", version=0)
    Trainer(max_epochs=2, logger=pl_logger).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert len(data := list(csv.DictReader(open(f"{pl_logger.log_dir}/metrics.csv", "r")))) == 2
    assert all(x["loss"] != 0 for x in data) and all(x["val_loss"] != 0 for x in data), data

    pl_logger2 = CSVLogger(get_project_root() / "test/logs", name="test_metrics_history_3", version=1)
    Trainer(max_epochs=5, logger=pl_logger2).fit(model, DataLoader(Reader()), DataLoader(Reader()),
                                                 ckpt_path=model.trainer.checkpoint_callback.last_model_path)
    assert len(data2 := list(csv.DictReader(open(f"{pl_logger2.log_dir}/metrics.csv", "r")))) == 5
    assert data[0:2] == data2[0:2]

if __name__ == "__main__":
    test_metrics_history_3()
