# pylint: disable=all
# flake8: noqa
from lightning_module_enhanced import LME
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import torch as tr
import json

class Reader(Dataset):
    def __init__(self, d_in: int, d_out: int, n: int = 100):
        self.x = tr.randn(n, d_in)
        self.gt = tr.randn(n, d_out)
    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]
    def __len__(self):
        return len(self.x)

class CustomScheduler(ReduceLROnPlateau):
    def step(self, metrics, epoch=None):
        print("!!!Applied!!!")
        self._reduce_lr(epoch)

def test_MetadataCallback_train_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=2).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert "epoch_timestamps" in meta and len(meta["epoch_timestamps"]) == 2
    assert "epoch_average_duration" in meta
    assert len(meta["epoch_metrics"]["l1"]) == 2
    assert "optimizer" in meta and isinstance(meta["optimizer"]["starting_lr"], float)
    assert "best_model" in meta

def test_MetadataCallback_test_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}

    assert model.metadata_callback.metadata is None
    Trainer().test(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert len(meta["epoch_metrics"]["l1"]) == 1
    assert "optimizer" not in meta
    assert "best_model" not in meta
    assert "epoch_timestamps" not in meta
    assert "epoch_average_duration" not in meta

def test_MetadataCallback_no_checkpoint():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_MetadataCallback_two_ModelCheckpoints():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.callbacks = [ModelCheckpoint(save_last=True, save_top_k=1, monitor="loss")]

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=2).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert "epoch_timestamps" in meta and len(meta["epoch_timestamps"]) == 2
    assert "epoch_average_duration" in meta
    assert len(meta["epoch_metrics"]["l1"]) == 2
    assert "optimizer" in meta
    assert "best_model" in meta

def test_MetadataCallback_two_monitors():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.checkpoint_monitors = ["loss", "l1"]

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=2).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "model_parameters" in meta
    assert "epoch_metrics" in meta and isinstance(meta["epoch_metrics"], dict) and "l1" in meta["epoch_metrics"]
    assert "epoch_timestamps" in meta and len(meta["epoch_timestamps"]) == 2
    assert "epoch_average_duration" in meta
    assert len(meta["epoch_metrics"]["l1"]) == 2
    assert "optimizer" in meta
    assert "best_model" in meta
    assert "scheduler" not in meta
    assert "early_stopping" not in meta

def test_MetadataCallback_scheduler_ReduceLROnPlateau():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.scheduler = {"scheduler": tr.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, "min"), "monitor": "loss"}
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=2).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata is not None
    meta = model.metadata_callback.metadata
    assert "scheduler" in meta
    assert meta["scheduler"]["type"] == "ReduceLROnPlateau"
    assert meta["scheduler"]["monitor"] == "loss"
    assert "early_stopping" not in meta

def test_MetadataCallback_scheduler_CustomScheduler():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.scheduler = {"scheduler": CustomScheduler(model.optimizer, "min", factor=0.5), "monitor": "loss"}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=3).fit(model, DataLoader(Reader(2, 1, 10)))
    best_epoch = model.metadata_callback.metadata["best_model"]["epoch"]
    assert model.metadata_callback.metadata["best_model"]["scheduler_num_lr_reduced"] == best_epoch
    assert model.metadata_callback.metadata["best_model"]["optimizer_lr"] == 0.01 * (0.5 ** best_epoch)

def test_MetadataCallback_early_stopping():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.scheduler = {"scheduler": tr.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, "min"), "monitor": "loss"}
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.metrics = {"l1": (lambda y, gt: (y - gt).abs().mean(), "min")}
    model.callbacks = [EarlyStopping("loss", min_delta=0.1, patience=1, mode="min")]
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)

    Trainer(max_epochs=2).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata is not None
    meta = model.metadata_callback.metadata
    assert "early_stopping" in meta
    assert meta["early_stopping"]["monitor"] == "loss"
    assert meta["early_stopping"]["patience"] == 1
    assert meta["early_stopping"]["mode"] == "min"
    assert meta["early_stopping"]["min_delta"] == -0.1

def test_MetadataCallback_metadata_hparams():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.scheduler = {"scheduler": tr.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, "min"), "monitor": "loss"}
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=2).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata is not None
    meta = json.load(open(model.metadata_callback.log_file_path, "r"))
    assert meta["fit_hparams"] is None

    model.hparams.metadata_hparams = {"my_hparam": 1, "other": [2, 3, 4, None, 5]}
    Trainer(max_epochs=5).fit(model, DataLoader(Reader(2, 1, 10)))
    meta2 = json.load(open((pth2 := model.metadata_callback.log_file_path), "r"))
    assert meta2["fit_hparams"] == {'my_hparam': 1, 'other': [2, 3, 4, None, 5]}

    Trainer().test(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.log_file_path != pth2
    meta3 = json.load(open(model.metadata_callback.log_file_path, "r"))
    assert meta3["test_hparams"] == {'my_hparam': 1, 'other': [2, 3, 4, None, 5]}

if __name__ == "__main__":
    test_MetadataCallback_scheduler_ReduceLROnPlateau()
