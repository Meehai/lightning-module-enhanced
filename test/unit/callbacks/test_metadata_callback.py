from lightning_module_enhanced import LME, TrainSetup
from pytorch_lightning import Trainer
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
    cfg = {
        "optimizer": {"type": "sgd", "args": {"lr": 0.01}},
        "criterion": {"type": "mse"},
        "metrics": [{"type": "l1"}]
    }
    TrainSetup(model, cfg)

    assert model.metadata_callback.metadata is None
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    assert model.metadata_callback.metadata is not None

    meta = model.metadata_callback.metadata
    assert "epoch_metrics" in meta
    assert "optimizer" in meta
    assert "best_model" in meta
