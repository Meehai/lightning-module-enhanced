from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn, optim
import torch as tr

from lightning_module_enhanced import LME, TrainSetup

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return {"data": tr.randn(2), "labels": tr.randn(1)}

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


if __name__ == "__main__":
    test_fit_and_test_good()
