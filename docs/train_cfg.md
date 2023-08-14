# Train CFG

Implementation at: [here](../lightning_module_enhanced/train_setup/train_setup.py). Documentation may be incomplete,
but code never lies.

`Lightning Module Enhanced` has basic support for a train yaml config file. It should follow
[this pattern](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers)
from standard Lightning Module.

Note: it's most likely better to use the standard setters to LME instead of a yaml file, however, this can be useful
if you have some baseline config and you don't want all the boilerplate in the code.

Keys:
- `optimizer`, the `args` are sent to the constructor of the optimizer type.
- `scheduler`, the `args` are sent to the constructor of the scheduler type, while `optimizer_args` are sent to the
dictionary from the link above (i.e. `monitor`, `interval`, `frequency` etc.).
- `criterion`. Only a few standard criterions are supported. These are better provided manually via
`model.criterion = criterion_fn`.
- `metrics`, only a few standard metrics are supported. Same as criterion, better provide them yourselves.
- `callbacks` not supported yet.

See runnable example: [copy pasta from here](../examples/train_cfg_simple.py).

```python
#!/usr/bin/env python3
"""simple usage of a train cfg"""
import yaml
from lightning_module_enhanced import LME, TrainSetup
from pytorch_lightning import Trainer
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
    monitor: loss
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
    Trainer(max_epochs=10).fit(model, tr.utils.data.DataLoader(MyReader(100, in_c, out_c), batch_size=10))
````
