# Lightning Module Enhanced documentation

Wrapper on top of [Lightning Module](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).

- Adds some common patterns, such as creating attributes for optimizer, scheduler, metric, criterion, device.
- Adds `model.summary()` via [torchinfo](https://github.com/TylerYep/torchinfo).
- Adding `np_forward` to pass numpy data and relatively automatically getting the correct device for a forward pass.
- Adds `model.model_algorithm` method for manual low level callback for a forward/loss/metrics pass. Defaults to
`LME.feed_forward_algorithm(model:LME, batch: dict={"data": ..., "labels": ...}, prefix: str)`.


Examples:
- See [train cfg](train_cfg.md) example for using a yaml train config that automatizes the setup for
optimizer, scheduler, loss, metrics.
- See [model algorithm](model_algorithm.md) for how to use a model algorithm callback, for more complicated train
or test semantics, that aren't a simple feed forward.
