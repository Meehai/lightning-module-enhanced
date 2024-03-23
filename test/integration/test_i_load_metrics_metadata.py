# Fixes this: https://gitlab.com/meehai/lightning-module-enhanced/-/issues/11
from pathlib import Path
import shutil
from torch import nn, optim
from torch.nn import functional as F
import torch as tr
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from lightning_module_enhanced import LME

class Reader(Dataset):
    def __init__(self):
        self.x = tr.randn(100, 2)
        self.gt = tr.randn(100, 1)

    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]

    def __len__(self):
        return len(self.x)

def test_load_metrics_metadata():
    train_loader = DataLoader(Reader(), batch_size=10)
    val_loader = DataLoader(Reader(), batch_size=10)
    log_dir_name = "load_metrics_metadata"
    shutil.rmtree(Path(__file__).parent / log_dir_name, ignore_errors=True)

    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.criterion_fn = F.mse_loss
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)

    pl_logger = CSVLogger(Path(__file__).parent, name=log_dir_name, version=0)
    t1 = Trainer(max_epochs=3, logger=pl_logger)
    t1.fit(model, train_loader, val_loader)

    pl_logger2 = CSVLogger(Path(__file__).parent, name=log_dir_name, version=1)
    ckpt_path = Path(__file__).parent / log_dir_name / "version_0" / "checkpoints" / "last.ckpt"
    t2 = Trainer(max_epochs=6, logger=pl_logger2)
    t2.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 6, \
        model.metadata_callback.metadata["epoch_metrics"]
    assert (Path(__file__).parent / log_dir_name / "version_1" / "checkpoints" / "loaded.ckpt").exists()
    assert (Path(__file__).parent / log_dir_name / "version_1" / "checkpoints" / ckpt_path.name).exists()
    shutil.rmtree(Path(__file__).parent / log_dir_name, ignore_errors=True)

if __name__ == "__main__":
    test_load_metrics_metadata()
