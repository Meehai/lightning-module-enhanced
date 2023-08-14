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

## Model algorithm

We may want some low level control over the model algorithm, that is more complicated than a feed forward network.
We support this via a callback. The default is `LME.feed_forward_algorithm` as explained earlier. See `test_fit.py`
for more examples.

Example:

```python

from pytorch_lightning import 

def my_model_algo(model, batch, prefix=""):
    x = batch["data"]
    assert isinstance(x, (dict, tr.Tensor)), type(x)
    # This allows {"data": {"a": ..., "b": ...}} to be mapped to forward(a, b)
    y = model.forward(x)
    gt = to_device(to_tensor(batch["labels"]), model.device)
    res = model.lme_metrics(y, gt, prefix, include_loss=False)
    assert "loss" not in res
    res["loss"] = model.criterion_fn(y, gt)
    return res

model = LME(nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)))
model.criterion_fn = lambda y, gt: (y - gt).pow(2).mean()
model.optimizer = optim.SGD(model.parameters(), lr=0.01)
model.model_algorithm = my_model_algo
Trainer(max_epochs=1).fit(model, DataLoader(Reader()))


```