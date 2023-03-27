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


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

    def forward(self, x):
        return self.sequential(x)

class BaseModelPlusSetup(BaseModel):
    @property
    def criterion_fn(self):
        return lambda y, gt: (y - gt).pow(2).mean()

    @property
    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.01)

def test_fit_good_implicit():
    model = LME(BaseModelPlusSetup())
    TrainSetup(model, {})
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
