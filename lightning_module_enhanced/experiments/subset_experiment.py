"""
Subset experiment module. Wrapper on top of a regular trainer to train the model n times with increasing sizes
of the original dataset
"""
from copy import deepcopy
from overrides import overrides
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from .experiment import Experiment

class SubsetExperiment(Experiment):
    """Subset experiment implementation"""
    def __init__(self, trainer: Trainer, num_subsets: int):
        super().__init__(trainer)
        self.num_subsets = num_subsets

    def _do_plot(self, res, n_total: int):
        ls = np.linspace(1 / self.num_subsets, 1, self.num_subsets)[0: len(res)]
        x = np.arange(len(res))
        for metric in res[0].keys():
            losses = [_res[metric] for _res in res]
            plt.figure()
            plt.scatter(x, losses)
            plt.plot(x, losses)
            plt.xticks(x, [f"{x*100:.2f}%" for x in ls])
            plt.xlabel("Percent used")
            plt.ylabel(f"Validation {metric}")
            plt.title(f"Subset experiment for {n_total} train size")
            out_file = f"{self.trainer.logger.log_dir}/subset_val_{metric}.png"
            plt.savefig(out_file)
            plt.close()

    def fit(self, model, train_dataloaders, val_dataloaders, *args, **kwargs):
        """The main function, uses same args as a regular pl.Trainer"""
        assert self.done is False, "Cannot fit twice"
        dataset = train_dataloaders.dataset
        dataloader_params = {
            "collate_fn": train_dataloaders.collate_fn,
            "num_workers": train_dataloaders.num_workers,
            "batch_size": train_dataloaders.batch_size,
        }

        ls = np.linspace(1 / self.num_subsets, 1, self.num_subsets)
        subset_lens = [int(len(dataset) * x) for x in ls]
        indices = [np.random.choice(len(dataset), x, replace=False) for x in subset_lens]
        subsets = [Subset(dataset, ind) for ind in indices]
        dataloaders = [DataLoader(subset, **dataloader_params) for subset in subsets]

        for cb in model.configure_callbacks():
            assert not isinstance(cb, ModelCheckpoint), "Subset experiment cannot have another ModelCheckpoint"

        res = []
        for i in range(self.num_subsets):
            iter_res = self.do_one_iteration(i, model, dataloaders[i], val_dataloaders, *args, **kwargs)
            res.append(iter_res)
            self._do_plot(res, len(dataset))
        self.done = True
        self.df_fit_metrics = pd.DataFrame(self.fit_metrics)
        self.ix = self.df_fit_metrics["loss"].argmin()

    @property
    def checkpoint_callback(self):
        return self.fit_trainers[self.ix].checkpoint_callback

    def __len__(self):
        return self.num_subsets
