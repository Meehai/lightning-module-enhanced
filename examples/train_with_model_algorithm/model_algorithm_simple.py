#!/usr/bin/env python3
"""simple usage of model_algorithm callback"""
from __future__ import annotations
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch as tr
from lightning_module_enhanced import LME, ModelAlgorithmOutput
from lightning_module_enhanced.callbacks import PlotMetrics

class MyReader:
    def __init__(self, n: int, in_c: int, out_c: int):
        self.in_c = in_c
        self.out_c = out_c
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, ix):
        return tr.randn(self.in_c), tr.randn(self.out_c)

def my_model_algo(model: LME, batch: dict) -> ModelAlgorithmOutput:
    x, gt = batch
    y = model.forward(x)
    res = model.lme_metrics(y, gt, include_loss=False) # if set to True, remove next line
    res["loss"] = model.criterion_fn(y, gt)
    return ModelAlgorithmOutput(y=y, metrics=res, x=x, gt=gt)

if __name__ == "__main__":
    in_c, out_c = 5, 10
    model = LME(tr.nn.Linear(in_c, out_c))
    model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.metrics = {
        "l1": (lambda y, gt: (y - gt).abs().mean(), "min"),
        "accuracy_like": (lambda y, gt: ((y > 0.5) == (gt > 0.5)).type(tr.float32).mean(), "max")
    }
    model.model_algorithm = my_model_algo
    model.callbacks = [PlotMetrics()]
    print(model.summary)
    train_loader = tr.utils.data.DataLoader(MyReader(n=100, in_c=in_c, out_c=out_c), batch_size=10)
    val_loader = tr.utils.data.DataLoader(MyReader(n=100, in_c=in_c, out_c=out_c), batch_size=10)
    Trainer(max_epochs=10, logger=CSVLogger("")).fit(model, train_loader, val_dataloaders=val_loader)
