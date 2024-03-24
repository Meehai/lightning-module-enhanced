# Lightning Module Enhanced documentation

Wrapper on top of [Lightning Module](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).

- Adds some common patterns, such as creating attributes for optimizer, scheduler, metric, criterion, device.
- Adds `model.summary()` via [torchinfo](https://github.com/TylerYep/torchinfo).
- Adding `np_forward` to pass numpy data and relatively automatically getting the correct device for a forward pass.
- Adds `model.model_algorithm` method for manual low level callback for a forward/loss/metrics pass.
- Adds [MultiTrainer](../lightning_module_enhanced/multi_trainer.py) that wraps lightning's Trainer to train the same
model N times and then seamlessly pick the best varaints. Used to minimize variance of training.


Examples:
- See [model algorithm](model_algorithm.md) for how to use a model algorithm callback, for more complicated train
or test semantics, that aren't a simple feed forward.
- See [multi trainer test](../test/integration/multi_trainer/test_i_multi_trainer.py) for `MultiTrainer` code.
