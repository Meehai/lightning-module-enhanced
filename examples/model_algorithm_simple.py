#!/usr/bin/env python3
"""simple usage of model_algorithm callback"""
from __future__ import annotations
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
        return tr.randn(self.in_c), tr.randn(self.out_c)

def my_model_algo(model: LME, batch: dict) -> tuple[tr.Tensor, dict[str, tr.Tensor]]:
    x, gt = batch[0], to_device(to_tensor(batch[1]), model.device)
    y = model.forward(x)
    res = model.lme_metrics(y, gt, include_loss=False) # if set to True, remove next line
    res["loss"] = model.criterion_fn(y, gt)
    return y, res, x, gt

if __name__ == "__main__":
    in_c, out_c = 5, 10
    model = LME(tr.nn.Linear(in_c, out_c))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = my_model_algo
    Trainer(max_epochs=10).fit(model, tr.utils.data.DataLoader(MyReader(100, in_c, out_c), batch_size=10))
