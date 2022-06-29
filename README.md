# Lightning Module Enhanced


## Description

Wrapper on top of [Lightning Module](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html), by adding some common patterns, such as creating attributes for optimizer, scheduler, metric, criterion, device. Adding `model.summary()` via [torchinfo](https://github.com/TylerYep/torchinfo). Adding `np_forward` to pass numpy data and relatively automatically getting the correct device for a forward pass.


## Train CFG

`Lightning Module Enhanced` has basic support for a train yaml config file. It should follow [this pattern](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers) from standard Lightning Module.

For `optimizer`, the `args` are sent to the constructor of the optimizer type.

For `scheduler`, the `args` are sent to the constructor of the scheduler type, while `optimizer_args` are sent to the dictionary from the link above (i.e. `monitor`, `interval`, `frequency` etc.).


```
    import yaml
    from lightning_module_enhanced import LightningModuleEnhanced
    train_cfg = yaml.safe_load(open("/path/to/train_cfg.yaml", "r"))
    model = LightningModuleEnhanced(get_pytorch_nn_module())
    model.setup_for_train(train_cfg)
```

and

`train_cfg.yaml`
```
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
```
