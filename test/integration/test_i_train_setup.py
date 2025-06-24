from functools import partial
import pytest
from lightning_module_enhanced import LME
from lightning_module_enhanced.trainable_module import TrainableModuleMixin
from lightning_fabric.utilities.exceptions import MisconfigurationException
from torchmetrics.functional.classification import multiclass_accuracy
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch as tr

class Reader(Dataset):
    def __init__(self, d_in: int, d_out: int, n: int = 100):
        self.x = tr.randn(n, d_in)
        self.gt = tr.randn(n, d_out)
    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]
    def __len__(self):
        return len(self.x)

class ReaderClassification:
    def __init__(self, n_dims: int = 1):
        self.n_dims = n_dims
    def __len__(self):
        return 10
    def __getitem__(self, ix):
        res = tr.zeros(self.n_dims)
        res[tr.randn(self.n_dims).argmax()] = 1
        return tr.randn(2), res

def test_all_are_none():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    assert len(model.metrics) == 0
    assert model.criterion_fn.metric_fn == TrainableModuleMixin._default_criterion_fn
    assert model.optimizer is None
    assert model.scheduler is None
    with pytest.raises(ValueError):
        Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_train_setup_minimal():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_train_setup_scheduler_bad_2():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.scheduler = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "val_loss"}
    # fails because no val_loss is available. We cannot make a pre-flight check because this is set up at .fit() time
    # and we have no val dataloader.
    with pytest.raises(MisconfigurationException):
        Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_train_setup_scheduler_good():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.scheduler = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "loss"}
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_train_setup_metrics_good_accuracy():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 5), nn.ReLU()))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"accuracy": (partial(multiclass_accuracy, num_classes=5, average="none"), "max")}
    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=7).fit(model, DataLoader(ReaderClassification(n_dims=5)))
    assert len(model.metadata_callback.metadata["epoch_metrics"]["accuracy"]) == 7
    assert len(model.metadata_callback.metadata["epoch_metrics"]["accuracy"][0]) == 5

def test_train_setup_metrics_l1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_train_setup_metrics_mse():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"mse": (lambda y, gt: (y - gt).pow(2).mean(), "min")}
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

if __name__ == "__main__":
    test_train_setup_metrics_good_accuracy()
