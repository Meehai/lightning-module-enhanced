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

See runnable example: [here](../examples/train_cfg_simple.py).
