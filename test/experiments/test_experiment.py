from tempfile import TemporaryDirectory
import torch as tr
from torch import nn, optim
from torch.utils.data import DataLoader
from lightning_module_enhanced.experiments import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_module_enhanced import LightningModuleEnhanced as LME
from pathlib import Path

class MyExperiment(Experiment):
    def __init__(self, trainer, n_experiments: int):
        super().__init__(trainer)
        self.n_experiments = n_experiments
        self.cnt = 0

    def __len__(self):
        return self.n_experiments

    def on_iteration_start(self, ix: int):
        self.cnt += 1

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

    def criterion_fn(self, y: tr.Tensor, gt: tr.Tensor):
        return (y - gt).pow(2).mean()

    @property
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.01)

def test_experiment_1():
    trainer = Trainer()
    e = MyExperiment(trainer, 5)
    assert e is not None
    assert e.cnt == 0

def test_experiment_2():
    train_data = Reader(n_data=100, n_dims=3)
    validation_data = Reader(n_data=100, n_dims=3)
    model = LME(Model(n_dims=train_data.n_dims))
    train_dataloader = DataLoader(train_data)
    val_dataloader = DataLoader(validation_data)
    save_dir = "save_dir_exp_2" if __name__ == "__main__" else TemporaryDirectory().name

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    e = MyExperiment(trainer, 3)
    e.fit(model, train_dataloader, val_dataloader)
    out_path = Path(save_dir) / "version_0"
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 3
    assert e.cnt == 3

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    e = MyExperiment(trainer, 5)
    e.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert e.cnt == 2

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    e = MyExperiment(trainer, 5)
    e.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert e.cnt == 0

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=1))
    e = MyExperiment(trainer, 5)
    e.fit(model, train_dataloader, val_dataloader)
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert e.cnt == 5

def test_experiment_3():
    train_data = Reader(n_data=100, n_dims=3)
    validation_data = Reader(n_data=100, n_dims=3)
    model = LME(Model(n_dims=train_data.n_dims))
    train_dataloader = DataLoader(train_data)
    val_dataloader = DataLoader(validation_data)
    save_dir = "save_dir_exp_3" if __name__ == "__main__" else TemporaryDirectory().name

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    e = MyExperiment(MyExperiment(trainer, 3), 5)
    e.fit(model, train_dataloader, val_dataloader)
    out_path = Path(save_dir) / "version_0"
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 5
    assert e.cnt == 5

if __name__ == "__main__":
    test_experiment_2()
