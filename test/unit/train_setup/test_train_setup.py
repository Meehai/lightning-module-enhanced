from lightning_module_enhanced import LME, TrainSetup
from lightning_module_enhanced.trainable_module import TrainableModuleMixin
from lightning_module_enhanced.schedulers import ReduceLROnPlateauWithBurnIn
from lightning_fabric.utilities.exceptions import MisconfigurationException
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch as tr

def test_train_setup_basic():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "SGD", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"}
    }
    TrainSetup(model, cfg)
    assert isinstance(model.optimizer, optim.SGD)
    assert model.optimizer.state_dict()["param_groups"][0]["lr"] == 0.01
    assert model.criterion_fn.metric_fn == F.mse_loss

def test_train_setup_scheduler_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "SGD", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "scheduler": {"type": "ReduceLROnPlateau",
                      "args": {"factor": 0.9, "patience": 5},
                      "optimizer_args": {"monitor": "val_loss"}}
    }
    TrainSetup(model, cfg)
    assert isinstance(model.scheduler_dict["scheduler"], optim.lr_scheduler.ReduceLROnPlateau)
    assert model.scheduler_dict["scheduler"].factor == 0.9
    assert model.scheduler_dict["scheduler"].patience == 5
    assert model.scheduler_dict["monitor"] == "val_loss"

def test_train_setup_scheduler_2():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "SGD", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "scheduler": {"type": "ReduceLROnPlateauWithBurnIn",
                      "args": {"factor": 0.9, "patience": 5, "burn_in_epochs": 20},
                      "optimizer_args": {"monitor": "val_loss"}}
    }
    TrainSetup(model, cfg)
    assert isinstance(model.scheduler_dict["scheduler"], ReduceLROnPlateauWithBurnIn)
    assert model.scheduler_dict["scheduler"].factor == 0.9
    assert model.scheduler_dict["scheduler"].patience == 5
    assert model.scheduler_dict["scheduler"].burn_in_epochs == 20
    assert model.scheduler_dict["monitor"] == "val_loss"

def test_train_setup_scheduler_3():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "SGD", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "scheduler": {"type": "ReduceLROnPlateau", "args": {"factor": 0.9, "patience": 5}}
    }
    try:
        TrainSetup(model, cfg)
        raise Exception
    except AssertionError:
        pass

def test_train_setup_metrics_accuracy_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 5), nn.ReLU()))
    cfg = {
        "optimizer": {"type": "SGD", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "metrics": [{"type": "accuracy", "args": {"task": "multiclass", "num_classes": 5, "average": "none"}}]
    }
    TrainSetup(model, cfg)
    assert sorted(model.metrics.keys()) == ["accuracy", "loss"]

def test_train_setup_metrics_accuracy_2():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "SGD", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "metrics": [{"type": "accuracy"}]
    }
    try:
        TrainSetup(model, cfg)
        raise Exception
    except AssertionError:
        pass

def test_train_setup_metrics_multi():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    cfg = {
        "optimizer": {"type": "SGD", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "metrics": [{"type": "accuracy", "args": {"task": "multiclass", "num_classes": 5, "average": "none"}}]
    }
    TrainSetup(model, cfg)
    model.metrics = {**model.metrics, "mse": (lambda y, gt: (y - gt).pow(2).mean(), "min")}
    assert sorted(model.metrics.keys()) == ["accuracy", "loss", "mse"]
