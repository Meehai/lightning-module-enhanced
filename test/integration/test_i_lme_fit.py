from __future__ import annotations
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.utils import to_device, to_tensor
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch as tr

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return tr.randn(2), tr.randn(1)

class MultiArgsLME(LME):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_algorithm = self.model_algorithm_multi_args

    @staticmethod
    def model_algorithm_multi_args(self, train_batch: dict) -> ModelAlgorithmOutput:
        x, gt = train_batch[0], to_device(to_tensor(train_batch[1]), self.device)
        y = self.forward(**x) if isinstance(x, dict) else self.forward(x)
        return y, self.lme_metrics(y, gt), x, gt

def test_fit_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))

def test_fit_no_criterion():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
        assert False
    except NotImplementedError:
        pass

def test_fit_no_optimizer():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
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
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 0


def test_fit_twice_with_validation():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 0


def test_fit_twice_with_validation_only_once_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 0


def test_fit_twice_with_validation_only_once_2():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert model.trainer.current_epoch == 10
    assert len(model.metrics) == 0
    # This should start from epoch 0 towards epoch 10, basically from scratch, but with pretrained weights
    assert list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0] == 0
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 10


def test_fit_twice_with_validation_only_once_3():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.metrics = {"metric1": (lambda y, gt: (y - gt).pow(2).mean(), "min")}
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=20).fit(model, DataLoader(Reader()))
    Trainer(max_epochs=10).fit(model, DataLoader(Reader()), DataLoader(Reader()))
    assert model.trainer.current_epoch == 10
    assert len(model.metrics) == 1
    # This should start from epoch 0 towards epoch 10, basically from scratch, but with pretrained weights
    assert list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0] == 0
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 10


def test_fit_twice_from_ckpt():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    trainer1 = Trainer(max_epochs=5)
    trainer1.fit(model, DataLoader(Reader()))
    Trainer(max_epochs=10).fit(
        model, DataLoader(Reader()), DataLoader(Reader()), ckpt_path=trainer1.checkpoint_callbacks[0].last_model_path
    )
    # This should start from epoch 5 towards epoch 10
    assert list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0] == "0", \
        list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0]
    assert len(model.metadata_callback.metadata["epoch_metrics"]["loss"]) == 10
    assert model.trainer.current_epoch == 10
    assert len(model.metrics) == 0

def test_fit_and_test_good():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
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
    model.scheduler = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "loss"}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=3).fit(model, DataLoader(Reader()))
    assert model.scheduler["scheduler"].last_epoch == 2

def test_fit_different_forward_params_1():
    class MyReader:
        def __len__(self):
            return 10

        def __getitem__(self, ix):
            # data contains a dict with key 'input' which maps to nn.Linear's forward function arg
            return {"input": tr.randn(2)}, tr.randn(1)

    model = MultiArgsLME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(MyReader()))

def test_fit_different_forward_params_2():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

        def forward(self, x):
            return self.fc(x)

    class MyReader:
        def __len__(self):
            return 10

        def __getitem__(self, ix):
            # data contains a dict with key 'x' which maps to MyModel's forward function arg
            return {"x": tr.randn(2)}, tr.randn(1)

    model = MultiArgsLME(MyModel())
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(MyReader()))

def test_fit_different_forward_params_3():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

        def forward(self, x):
            return self.fc(x)

    class MyReader:
        def __len__(self):
            return 10

        def __getitem__(self, ix):
            # data contains a dict with key 'blabla' which doesn't map to MyModel's forward function arg (x)
            return {"blabla": tr.randn(2)}, tr.randn(1)

    model = MultiArgsLME(MyModel())
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(MyReader()))
    except TypeError:
        pass

def test_fit_different_forward_params_4():
    class MyModel2Args(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

        def forward(self, x, y):
            return self.fc(x) + self.fc(y)

    class MyReader:
        def __len__(self):
            return 10

        def __getitem__(self, ix):
            # data contains a dict with key 'x' which doesn't map to MyModel2Args's forward function arg (2 args)
            return {"blabla": tr.randn(2)}, tr.randn(1)

    model = MultiArgsLME(MyModel2Args())
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(MyReader()))
    except TypeError:
        pass

def test_fit_different_forward_params_5():
    class MyModel2Args(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

        def forward(self, x, y):
            return self.fc(x) + self.fc(y)

    class MyReader:
        def __len__(self):
            return 10

        def __getitem__(self, ix):
            # data contains a dict with no key, which doesn't map to MyModel2Args's forward fn (2 args)
            return tr.randn(2), tr.randn(1)

    model = MultiArgsLME(MyModel2Args())
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(MyReader()))
    except TypeError:
        pass

def test_fit_different_forward_params_6():
    class MyModel2Args(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

        def forward(self, x, y):
            return self.fc(x) + self.fc(y)

    class MyReader:
        def __len__(self):
            return 10

        def __getitem__(self, ix):
            # data contains a dict with 2 keys, mapping the name of the arguments of MyModel2Args' forward function
            return {"x": tr.randn(2), "y": tr.randn(2)}, tr.randn(1)

    model = MultiArgsLME(MyModel2Args())
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(MyReader()))

def test_i_load_from_checkpoint():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.scheduler = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "loss"}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.checkpoint_monitors = ["loss"]
    model.hparams.hello = "world"
    (t1 := Trainer(max_epochs=3)).fit(model, DataLoader(Reader()))

    model2 = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model2.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model2.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model2.scheduler = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "loss"}
    model2.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)

    model2.load_from_checkpoint(t1.checkpoint_callback.last_model_path)
    assert model2.hparams.hello == "world"
    assert model2.optimizer.state_dict() == model.optimizer.state_dict()
    assert model2.scheduler["scheduler"].state_dict() == model.scheduler["scheduler"].state_dict()

if __name__ == "__main__":
    test_i_load_from_checkpoint()
