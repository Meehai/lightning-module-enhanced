from functools import partial
import shutil
from torch import nn, optim
from torch.nn import functional as F
import pytest
import torch as tr
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.metrics import CallableCoreMetric, CoreMetric
from lightning_module_enhanced.utils import get_project_root

class Reader(Dataset):
    def __init__(self, d_out: int = 1):
        self.x = tr.randn(100, 2)
        self.gt = tr.randn(100, d_out)

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
    assert f"{exc.value}" == "Expected metrics: set() vs. this batch: dict_keys(['loss', 'my_metric'])"

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
    assert f"{exc.value}" == "Expected metrics: set() vs. this batch: dict_keys(['loss', 'epik_metric'])"

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
    assert len(model.metrics_history.history["my_metric"]["train"]) == 3
    assert len(model.metrics_history.history["loss"]["train"]) == 3

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
    assert len(model.metrics_history.history["my_epoch_metric"]["train"]) == 3
    assert len(model.metrics_history.history["loss"]["train"]) == 3

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
    assert len(model.metrics_history.history["my_epoch_metric"]["train"]) == 3
    assert len(model.metrics_history.history["loss"]["train"]) == 3
