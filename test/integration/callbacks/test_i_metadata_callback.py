from lightning_module_enhanced import LME
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch import nn
import torch as tr

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return {"data": tr.randn(2), "labels": tr.randn(1)}


def test_metadata_callback_train_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=2).fit(model, DataLoader(Reader()))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert "epoch_timestamps" in meta and len(meta["epoch_timestamps"]) == 2
    assert "epoch_average_duration" in meta
    assert len(meta["epoch_metrics"]["l1"]) == 2
    assert "optimizer" in meta
    assert "best_model" in meta

def test_metadata_callback_test_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}

    assert model.metadata_callback.metadata is None
    Trainer().test(model, DataLoader(Reader()))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert len(meta["epoch_metrics"]["l1"]) == 1
    assert "optimizer" not in meta
    assert "best_model" not in meta
    assert "epoch_timestamps" not in meta
    assert "epoch_average_duration" not in meta

def test_metadata_callback_no_checkpoint():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))

def test_metadata_callback_two_ModelCheckpoints():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    model.callbacks = [ModelCheckpoint(save_last=True, save_top_k=1, monitor="loss")]

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=2).fit(model, DataLoader(Reader()))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert "epoch_timestamps" in meta and len(meta["epoch_timestamps"]) == 2
    assert "epoch_average_duration" in meta
    assert len(meta["epoch_metrics"]["l1"]) == 2
    assert "optimizer" in meta
    assert "best_model" in meta

def test_metadata_callback_two_monitors():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    model.checkpoint_monitors = ["loss", "l1"]

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=2).fit(model, DataLoader(Reader()))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert "epoch_timestamps" in meta and len(meta["epoch_timestamps"]) == 2
    assert "epoch_average_duration" in meta
    assert len(meta["epoch_metrics"]["l1"]) == 2
    assert "optimizer" in meta
    assert "best_model" in meta
