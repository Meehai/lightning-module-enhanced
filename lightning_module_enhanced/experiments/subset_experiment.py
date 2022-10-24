"""
Subset experiment module. Wrapper on top of a regular trainer to train the model n times with increasing sizes
of the original dataset
"""
from overrides import overrides
from typing import Dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .experiment import Experiment

class SubsetExperiment(Experiment):
    """Subset experiment implementation"""
    def __init__(self, trainer: Trainer, num_subsets: int):
        super().__init__(trainer)
        self.num_subsets = num_subsets

    def _do_plot(self):
        breakpoint()
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
            plt.title(f"Subset experiment for {len(self._train_dataset)} train size")
            out_file = f"{self.trainer.logger.log_dir}/subset_val_{metric}.png"
            plt.savefig(out_file)
            plt.close()

    @overrides
    def ix_to_id(self, ix: int) -> Dict[int, str]:
        return f""

    @overrides
    def on_fit_start(self):
        ls = np.linspace(1 / self.num_subsets, 1, self.num_subsets)
        subset_lens = [int(len(self._dataset) * x) for x in ls]
        indices = [np.random.choice(len(self._dataset), x, replace=False) for x in subset_lens]
        subsets = [Subset(self._dataset, ind) for ind in indices]
        self.dataloaders = [DataLoader(subset, **self.dataloader_params) for subset in subsets]

    @overrides
    def on_fit_end(self):
        pass

    @overrides
    def on_before_iteration(self, ix: int):
        pass

    @overrides
    def on_after_iteration(self, ix: int):
        return self._do_plot()

    # @overrides
    # def fit(self, model, train_dataloaders, val_dataloaders, *args, **kwargs):
    #     """The main function, uses same args as a regular pl.Trainer"""
    #     assert self.done is False, "Cannot fit twice"
    #     self.fit_setup(model, train_dataloaders, val_dataloaders)

    #     res = []
    #     for i in range(self.num_subsets):
    #         iter_res = self.do_one_iteration(i, self._model, self.dataloaders[i],
    #                                          self._val_dataloaders, *args, **kwargs)
    #         res.append(iter_res)
    #         self._do_plot()
    #     self.done = True
    #     self.df_fit_metrics = pd.DataFrame(self.fit_metrics)
    #     self.best_id = self.df_fit_metrics.iloc[self.df_fit_metrics["loss"].argmin()].index

    def __len__(self):
        return self.num_subsets
