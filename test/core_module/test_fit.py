from lightning_module_enhanced import LightningModuleEnhanced
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch import nn, optim
import torch as tr


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

def test_fit_no_criterion():
    model = LightningModuleEnhanced(BaseModel())
    try:
        Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
        assert False
    except ValueError:
        pass

def test_fit_good_implicit():
    model = LightningModuleEnhanced(BaseModelPlusSetup())
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))

def test_fit_good_explicit():
    model = LightningModuleEnhanced(BaseModel())
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))

def test_fit_explicit_over_implicit_1():
    model = LightningModuleEnhanced(BaseModelPlusSetup())
    def f(y, gt):
        return (y - gt).abs().mean()
    model.criterion_fn = f
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    assert model.criterion_fn.metric_fn == f

def test_fit_explicit_over_implicit_2():
    model = LightningModuleEnhanced(BaseModelPlusSetup())
    model.optimizer = optim.Adam(model.parameters(), lr=0.01)
    Trainer(max_epochs=1).fit(model, DataLoader(Reader()))
    assert isinstance(model.optimizer, optim.Adam)

if __name__ == "__main__":
    test_fit_good_implicit()
