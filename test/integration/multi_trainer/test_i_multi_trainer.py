from tempfile import TemporaryDirectory
import shutil
import torch as tr
import pytest
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from lightning_module_enhanced import LME
from lightning_module_enhanced.multi_trainer import MultiTrainer
from pathlib import Path

class Reader(Dataset):
    def __init__(self, d_in: int, d_out: int, n: int = 100):
        self.d_in = d_in
        self.d_out = d_out
        self.x = tr.randn(n, d_in)
        self.gt = tr.randn(n, d_out)
    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]
    def __len__(self):
        return len(self.x)

class Model(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x: tr.Tensor):
        return self.fc(x)

def test_multi_trainer_ctor():
    trainer = Trainer()
    e = MultiTrainer(trainer, num_trains=5)
    assert e is not None
    assert e.done_so_far == 0

def test_multi_trainer_fit():
    train_data = Reader(3, 3, 100)
    validation_data = Reader(3, 3, 100)
    model = LME(Model(train_data.d_in, train_data.d_out))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    train_dataloader = DataLoader(train_data)
    val_dataloader = DataLoader(validation_data)
    save_dir = "/tmp/save_dir_2" if __name__ == "__main__" else TemporaryDirectory().name
    shutil.rmtree(save_dir, ignore_errors=True)

    trainer = Trainer(max_epochs=3, logger=CSVLogger(save_dir=save_dir, name="", version=0))
    mt1 = MultiTrainer(trainer, num_trains=3)
    mt1.fit(model, train_dataloader, val_dataloader)
    out_path = Path(save_dir) / "version_0/MultiTrainer"
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 3
    assert mt1.done_so_far == 3

    trainer = Trainer(max_epochs=3, logger=CSVLogger(save_dir=save_dir, name="", version=0))
    mt2 = MultiTrainer(trainer, num_trains=5)
    assert mt2.done_so_far == 3
    mt2.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert mt2.done_so_far == 5

    trainer = Trainer(max_epochs=3, logger=CSVLogger(save_dir=save_dir, name="", version=0))
    mt3 = MultiTrainer(trainer, num_trains=5)
    assert mt3.done_so_far == 5
    mt3.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert mt3.done_so_far == 5

    trainer = Trainer(max_epochs=3, logger=CSVLogger(save_dir=save_dir, name="", version=1))
    mt4 = MultiTrainer(trainer, num_trains=5)
    assert mt4.done_so_far == 0
    mt4.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert mt4.done_so_far == 5

def test_multi_trainer_3():
    train_data = Reader(3, 3, 100)
    model = LME(Model(train_data.d_in, train_data.d_out))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    train_dataloader = DataLoader(train_data)
    save_dir = "/tmp/save_dir_3" if __name__ == "__main__" else TemporaryDirectory().name
    shutil.rmtree(save_dir, ignore_errors=True)
    trainer = Trainer(max_epochs=3, logger=CSVLogger(save_dir=save_dir, name="", version="12"))
    mt2 = MultiTrainer(trainer, num_trains=5)
    mt2.fit(model, train_dataloader)
    assert (Path(save_dir) / "12").exists()
    assert "checkpoints" in [x.name for x in (Path(save_dir) / "12").iterdir()]
    assert "fit_metadata.json" in [x.name for x in (Path(save_dir) / "12").iterdir()]

def test_multi_trainer_parallel_cpu():
    train_data = Reader(3, 3, 100)
    model = LME(Model(train_data.d_in, train_data.d_out))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    train_dataloader = DataLoader(train_data)
    save_dir = "/tmp/save_dir_parallel" if __name__ == "__main__" else TemporaryDirectory().name
    shutil.rmtree(save_dir, ignore_errors=True)

    trainer = Trainer(max_epochs=3, logger=CSVLogger(save_dir=save_dir, name="", version="12"), accelerator="cpu")
    with pytest.raises(AssertionError, match=f"Expected {1 << 20}, got"):
        mt2 = MultiTrainer(trainer, num_trains=5, n_devices=1 << 20)

    mt2 = MultiTrainer(trainer, num_trains=5, n_devices=-1)
    mt2.fit(model, train_dataloader, ckpt_path=None)
    assert (Path(save_dir) / "12").exists()
    assert "checkpoints" in [x.name for x in (Path(save_dir) / "12").iterdir()]
    assert "fit_metadata.json" in [x.name for x in (Path(save_dir) / "12").iterdir()]

if __name__ == "__main__":
    test_multi_trainer_parallel_cpu()
