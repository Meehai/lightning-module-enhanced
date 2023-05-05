from lightning_module_enhanced import LME
from lightning_module_enhanced.utils import accelerator_params_from_module
from torch import nn, optim
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch as tr

class Reader:
    def __len__(self):
        return 10

    def __getitem__(self, ix):
        return {"data": tr.randn(2), "labels": tr.randn(1)}


def test_accelerator_params_from_module_1():
    module = LME(nn.Linear(2, 1)).to("cpu")
    module.optimizer = optim.SGD(module.parameters(), lr=0.01)
    module.criterion_fn = lambda y, gt: ((y - gt)**2).mean()
    params = accelerator_params_from_module(module)
    Trainer(**params, max_epochs=1).fit(module, DataLoader(Reader()), DataLoader(Reader()))

if __name__ == "__main__":
    test_accelerator_params_from_module_1()
