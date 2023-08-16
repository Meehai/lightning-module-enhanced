#!/usr/bin/env python3
"""simple usage of model_algorithm callback"""
from pytorch_lightning import Trainer
import torch as tr
from lightning_module_enhanced import LME
from lightning_module_enhanced.utils import to_device, to_tensor

class MyReader:
    def __init__(self, n: int, in_c: int, out_c: int):
        self.in_c = in_c
        self.out_c = out_c
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, ix):
        return {"data": tr.randn(self.in_c), "labels": tr.randn(self.out_c)}

def my_model_algo(model: LME, batch: dict, prefix="") -> dict[str, tr.Tensor]:
    x = batch["data"]
    y = model.forward(x)
    gt = to_device(to_tensor(batch["labels"]), model.device)
    res = model.lme_metrics(y, gt, prefix, include_loss=False) # if set to True, remove next line
    res["loss"] = model.criterion_fn(y, gt)
    return res

if __name__ == "__main__":
    in_c, out_c = 5, 10
    model = LME(tr.nn.Linear(in_c, out_c))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = my_model_algo
    Trainer(max_epochs=10).fit(model, tr.utils.data.DataLoader(MyReader(100, in_c, out_c), batch_size=10))
