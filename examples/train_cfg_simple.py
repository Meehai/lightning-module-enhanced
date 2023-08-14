#!/usr/bin/env python3
"""simple usage of a train cfg"""
import yaml
from lightning_module_enhanced import LME, TrainSetup
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch as tr

train_cfg_str = """
optimizer:
  type: adamw
  args:
    lr: 0.01
scheduler:
  type: ReduceLROnPlateau
  args:
    mode: min
    patience: 10
    factor: 0.5
  optimizer_args:
    monitor: val_loss
criterion:
    type: mse
"""

class MyReader:
    def __init__(self, n: int, in_c: int, out_c: int):
        self.in_c = in_c
        self.out_c = out_c
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, ix):
        return {"data": tr.randn(self.in_c), "labels": tr.randn(self.out_c)}

if __name__ == "__main__":
    train_cfg = yaml.safe_load(train_cfg_str)
    in_c, out_c = 5, 10
    model = LME(tr.nn.Linear(in_c, out_c))
    TrainSetup(model, train_cfg)
    Trainer().fit(model, DataLoader(MyReader(100, in_c, out_c), batch_size=10))
