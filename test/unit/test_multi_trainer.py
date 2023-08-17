from tempfile import TemporaryDirectory
import shutil
import torch as tr
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from lightning_module_enhanced import LME
from lightning_module_enhanced.multi_trainer import MultiTrainer
from pathlib import Path

class Reader:
    def __init__(self, n_data: int, n_dims: int):
        self.n_data = n_data
        self.n_dims = n_dims
        self.data = tr.randn(n_data, n_dims)
        self.labels = tr.randn(n_data, n_dims)

    def __getitem__(self, ix):
        return {"data": self.data[ix], "labels": self.labels[ix]}

    def __len__(self):
        return self.n_data


class Model(nn.Module):
    def __init__(self, n_dims: int):
        super().__init__()
        self.fc = nn.Linear(n_dims, n_dims)

    def forward(self, x: tr.Tensor):
        return self.fc(x)

def test_multi_trainer_1():
    trainer = Trainer()
    e = MultiTrainer(trainer, num_trains=5)
    assert e is not None
    assert e.done_so_far == 0

def test_multi_trainer_2():
    train_data = Reader(n_data=100, n_dims=3)
    validation_data = Reader(n_data=100, n_dims=3)
    model = LME(Model(n_dims=train_data.n_dims))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_dataloader = DataLoader(train_data)
    val_dataloader = DataLoader(validation_data)
    save_dir = "save_dir_exp_2" if __name__ == "__main__" else TemporaryDirectory().name
    shutil.rmtree(save_dir, ignore_errors=True)

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    mt1 = MultiTrainer(trainer, num_trains=3)
    mt1.fit(model, train_dataloader, val_dataloader)
    out_path = Path(save_dir) / "version_0/MultiTrainer"
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 3
    assert mt1.done_so_far == 3

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    mt2 = MultiTrainer(trainer, num_trains=5)
    assert mt2.done_so_far == 3
    mt2.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert mt2.done_so_far == 5

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    mt3 = MultiTrainer(trainer, num_trains=5)
    assert mt3.done_so_far == 5
    mt3.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert mt3.done_so_far == 5

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=1))
    mt4 = MultiTrainer(trainer, num_trains=5)
    assert mt4.done_so_far == 0
    mt4.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert mt4.done_so_far == 5

def test_multi_trainer_3():
    train_data = Reader(n_data=100, n_dims=3)
    model = LME(Model(n_dims=train_data.n_dims))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_dataloader = DataLoader(train_data)
    save_dir = "save_dir_exp_3" if __name__ == "__main__" else TemporaryDirectory().name
    shutil.rmtree(save_dir, ignore_errors=True)
    trainer = Trainer(max_epochs=3, logger=CSVLogger(save_dir=save_dir, name="", version="12"))
    mt2 = MultiTrainer(trainer, num_trains=5)
    mt2.fit(model, train_dataloader)
    assert (Path(save_dir) / "12").exists()
    assert "checkpoints" in [x.name for x in (Path(save_dir) / "12").iterdir()]
    assert "fit_metadata.json" in [x.name for x in (Path(save_dir) / "12").iterdir()]

if __name__ == "__main__":
    test_multi_trainer_2()
