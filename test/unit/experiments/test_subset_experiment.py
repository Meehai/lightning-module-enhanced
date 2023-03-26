from tempfile import TemporaryDirectory
import torch as tr
from torch import nn, optim
from torch.utils.data import DataLoader
from lightning_module_enhanced.experiments import SubsetExperiment
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_module_enhanced import LME
from pathlib import Path

lens = []


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

    @property
    def callbacks(self):
        return [MyCallback()]


class MyCallback(Callback):
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        global lens
        train_reader_len = len(trainer.train_dataloader.dataset.datasets)
        lens.append(train_reader_len)


def test_subset_experiment_1():
    train_data = Reader(n_data=100, n_dims=3)
    validation_data = Reader(n_data=100, n_dims=3)
    model = LME(Model(n_dims=train_data.n_dims))
    train_dataloader = DataLoader(train_data)
    val_dataloader = DataLoader(validation_data)
    save_dir = "save_dir" if __name__ == "__main__" else TemporaryDirectory().name

    trainer = Trainer(max_epochs=3, logger=TensorBoardLogger(save_dir=save_dir, name="", version=0))
    e = SubsetExperiment(trainer, 3)
    e.fit(model, train_dataloader, val_dataloader)
    out_path = Path(save_dir) / "version_0"
    assert len([x for x in out_path.iterdir() if x.is_dir()]) == 3
    assert lens == [33, 66, 100]


if __name__ == "__main__":
    test_subset_experiment_1()
