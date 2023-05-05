from lightning_module_enhanced import LME
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch as tr


class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return {"data": tr.randn(2), "labels": tr.randn(1)}

def test_fit_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))

def test_fit_no_criterion():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
        assert False
    except ValueError:
        pass

def test_fit_no_optimizer():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean() 
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
        assert False
    except ValueError:
        pass

### Test fit twice ###

def test_fit_twice():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 1


def test_fit_twice_with_validation():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 1


def test_fit_twice_with_validation_only_once_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 1


def test_fit_twice_with_validation_only_once_2():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert model.trainer.current_epoch == 10
    assert len(model.metrics) == 1
    # This should start from epoch 0 towards epoch 10, basically from scratch, but with pretrained weights
    assert list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0] == 0
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 10


def test_fit_twice_with_validation_only_once_3():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.metrics = {"metric1": (lambda y, gt: (y - gt).pow(2).mean(), "min")}
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert model.trainer.current_epoch == 10
    assert len(model.metrics) == 2
    # This should start from epoch 0 towards epoch 10, basically from scratch, but with pretrained weights
    assert list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0] == 0
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 10


def test_fit_twice_from_ckpt():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    trainer1 = Trainer(max_epochs=5)
    trainer1.fit(model, DataLoader(Reader()))
    Trainer(max_epochs=10).fit(
        model, DataLoader(Reader()), DataLoader(Reader()), ckpt_path=trainer1.checkpoint_callbacks[0].last_model_path
    )
    # This should start from epoch 5 towards epoch 10
    assert list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0] == 5
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 5
    assert model.trainer.current_epoch == 10
    assert len(model.metrics) == 1

def test_fit_and_test_good():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.metrics = {
        "metric1": (lambda y, gt: (y - gt).abs().mean(), "min"),
        "metric2": (lambda y, gt: (y - gt) * 0, "min"),
    }

    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    res = Trainer().test(model, DataLoader(Reader()))
    assert len(res) == 1
    assert sorted(res[0].keys()) == ["loss", "metric1", "metric2"], res[0].keys()

def test_fit_with_scheduler():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.scheduler_dict = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "loss"}

    Trainer(max_epochs=2).fit(model, DataLoader(Reader()))
    assert model.scheduler_dict["scheduler"].last_epoch == 2

if __name__ == "__main__":
    test_fit_twice_from_ckpt()
