from __future__ import annotations
import pytest
from pathlib import Path
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.callbacks import PlotCallbackGeneric, PlotMetrics
from lightning_module_enhanced.utils import get_project_root
from lightning_module_enhanced.schedulers import MinMaxLR
from lightning_module_enhanced.metrics import CallableCoreMetric
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch as tr

class Reader(Dataset):
    def __init__(self, d_in: int, d_out: int, n: int = 100):
        self.x = tr.randn(n, d_in)
        self.gt = tr.randn(n, d_out)
    def __getitem__(self, ix):
        return self.x[ix], self.gt[ix]
    def __len__(self):
        return len(self.x)

class MultiArgsLME(LME):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_algorithm = self.model_algorithm_multi_args

    @staticmethod
    def model_algorithm_multi_args(self, train_batch: dict) -> ModelAlgorithmOutput:
        x, gt = train_batch
        y = self.forward(**x) if isinstance(x, dict) else self.forward(x)
        return y, self.lme_metrics(y, gt), x, gt

def test_fit_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_fit_no_criterion():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))
        assert False
    except NotImplementedError:
        pass

def test_fit_no_optimizer():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))
        assert False
    except ValueError:
        pass

### Test fit twice ###

def test_fit_twice():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=10).fit(model, DataLoader(Reader(2, 1, 10)))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 0


def test_fit_twice_with_validation():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=10).fit(model, DataLoader(Reader(2, 1, 10)), DataLoader(Reader(2, 1, 10)))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader(2, 1, 10)), DataLoader(Reader(2, 1, 10)))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 0


def test_fit_twice_with_validation_only_once_1():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=10).fit(model, DataLoader(Reader(2, 1, 10)), DataLoader(Reader(2, 1, 10)))
    Trainer(max_epochs=20).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.trainer.current_epoch == 20
    assert len(model.metrics) == 0


def test_fit_twice_with_validation_only_once_2():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=20).fit(model, DataLoader(Reader(2, 1, 10)))
    Trainer(max_epochs=10).fit(model, DataLoader(Reader(2, 1, 10)), DataLoader(Reader(2, 1, 10)))
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
    Trainer(max_epochs=20).fit(model, DataLoader(Reader(2, 1, 10)))
    Trainer(max_epochs=10).fit(model, DataLoader(Reader(2, 1, 10)), DataLoader(Reader(2, 1, 10)))
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
    trainer1.fit(model, DataLoader(Reader(2, 1, 10)))
    Trainer(max_epochs=10).fit(
        model, DataLoader(Reader(2, 1, 10)), DataLoader(Reader(2, 1, 10)),
        ckpt_path=trainer1.checkpoint_callbacks[-1].last_model_path
    )
    # This should start from epoch 5 towards epoch 10
    assert list(model.metadata_callback.metadata["epoch_metrics"]["loss"].keys())[0] == 0, \
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

    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))
    res = Trainer().test(model, DataLoader(Reader(2, 1, 10)))
    assert len(res) == 1
    assert sorted(res[0].keys()) == ["loss", "metric1", "metric2"], res[0].keys()

def test_fit_with_ReduceLROnPlateau_scheduler():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.scheduler = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "loss"}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=3).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.scheduler["scheduler"].last_epoch == 2

def test_fit_with_DeltaLR_scheduler():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.scheduler = {"scheduler": MinMaxLR(model.optimizer, min_lr=0.001, max_lr=0.5,
                                             n_steps=2, warmup_steps=2), "monitor": "loss"}
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    Trainer(max_epochs=10).fit(model, DataLoader(Reader(2, 1, 10)))
    assert model.metadata_callback.metadata["optimizer"]["lr_history"] == \
        [0.01, 0.01, 0.0055, 0.001, 0.0055, 0.01, 0.255, 0.5, 0.255, 0.01]

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
    with pytest.raises(AssertionError) as exc:
        model.checkpoint_monitors = ["loss", "some_metric"]
    assert f"{exc.value}" == "Not in metrics: {'some_metric'} (metrics: [])"
    model.checkpoint_monitors = ["loss"]
    model.hparams.hello = "world"
    (t1 := Trainer(max_epochs=3)).fit(model, DataLoader(Reader(2, 1, 10)))

    model2 = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model2.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model2.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model2.scheduler = {"scheduler": ReduceLROnPlateau(model.optimizer, factor=0.9, patience=5), "monitor": "loss"}
    model2.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)

    model2.load_from_checkpoint(t1.checkpoint_callbacks[-1].last_model_path)
    assert model2.hparams.hello == "world"
    assert model2.optimizer.state_dict() == model.optimizer.state_dict()
    assert model2.scheduler["scheduler"].state_dict() == model.scheduler["scheduler"].state_dict()

    model3 = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model3.load_from_checkpoint(t1.checkpoint_callbacks[-1].last_model_path)
    assert model3.optimizer is None and model3.scheduler is None
    assert model3.hparams.hello == "world"

@pytest.mark.parametrize("mode", ["first", "random"])
def test_fit_with_PlotCallbacks(mode: str):
    last_epoch_seen = -1
    def plot_fn(model: LME, batch: tuple, y: tr.Tensor, out_dir):
        nonlocal last_epoch_seen
        assert model.trainer.current_epoch == last_epoch_seen + 1
        last_epoch_seen += 1
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.callbacks = [PlotMetrics(), PlotCallbackGeneric(plot_fn, mode=mode)]
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    # with pytest.raises(AssertionError): # TODO: doesn't raise in gitlab O_o
    #     Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))
    pl_logger = CSVLogger(get_project_root() / "test/logs", name="test_fit_with_PlotCallbacks", version=0)
    Trainer(max_epochs=3, logger=pl_logger).fit(model, DataLoader(Reader(2, 1, 10)))

def test_fit_with_two_optimizers():
    model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    all_params = list(model.parameters())
    model.optimizer = [
        optim.SGD(all_params[0:len(all_params)//2], lr=0.01),
        optim.SGD(all_params[len(all_params)//2:], lr=0.05)
    ]
    Trainer(max_epochs=1).fit(model, DataLoader(Reader(2, 1, 10)))

def test_fit_twice_and_proper_resume_metrics_state(tmpdir: str, monkeypatch: pytest.MonkeyPatch):
    class EpikMetric(CallableCoreMetric):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.epoch = 0
        def forward(self, y, gt):
            return None
        def batch_update(self, batch_result) -> None:
            pass
        def epoch_result(self) -> tr.Tensor:
            self.epoch += 1
            # return "1" after 3 epochs so we can test after a load from epoch 3 it maintains it as 'best epoch'
            return tr.FloatTensor([self.epoch if self.epoch <= 3 else 1]).to(self.device)
        def reset(self):
            pass

    def build_model(d_in: int, d_out: int) -> LME:
        model = LME(tr.nn.Linear(d_in, d_out))
        model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
        model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
        model.metrics = {"epik_metric": EpikMetric(lambda y, gt: (y - gt) ** 2, higher_is_better=True)}
        model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)
        model.callbacks = [PlotMetrics()]
        model.checkpoint_monitors = ["epik_metric", "loss"]
        model._save_weights_only_monitor_ckpts = False # so the intermediate ckpts also save all states for reload
        return model

    monkeypatch.setenv("LME_LOAD_MODEL_CHECKPOINT_BEST_SCORES", "1")
    d_in, d_out = 5, 10
    model = build_model(d_in, d_out)
    print(model.summary)
    train_loader = tr.utils.data.DataLoader(Reader(n=30, d_in=d_in, d_out=d_out), batch_size=10)
    val_loader = tr.utils.data.DataLoader(Reader(n=30, d_in=d_in, d_out=d_out), batch_size=10)
    trainer = Trainer(max_epochs=5, logger=[CSVLogger(tmpdir)])
    trainer.fit(model, train_loader, val_dataloaders=val_loader)

    ckpt_path = model.checkpoint_callback.best_model_path
    model = build_model(d_in, d_out)
    trainer = Trainer(max_epochs=5, logger=[CSVLogger(tmpdir)])
    trainer.fit(model, train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    all_ckpts = (Path(tmpdir) / "lightning_logs/version_1/checkpoints").iterdir()
    ckpts = [k for k in all_ckpts if k.name.find("val_epik_metric") != -1]
    assert len(ckpts) == 1, ckpts

if __name__ == "__main__":
    test_fit_twice_from_ckpt()
