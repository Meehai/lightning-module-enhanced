"""
Subset experiment module. Wrapper on top of a regular trainer to train the model n times with increasing sizes
of the original dataset
"""
from typing import Union, List
from overrides import overrides
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from .experiment import Experiment

class SubsetExperiment(Experiment):
    """Subset experiment implementation"""
    def __init__(self, trainer: Trainer, subsets: Union[int, List[int]]):
        super().__init__(trainer)
        assert isinstance(subsets, (int, list))
        self.subsets = subsets
        self.num_subsets = subsets if isinstance(subsets, int) else len(subsets)
        self.dataloaders = None
        self.tmp_train_dataloader = None

    def _do_plot(self):
        ls = np.linspace(1 / self.subsets, 1, self.subsets) if isinstance(self.subsets, int) else self.subsets
        x = ls[0: len(self.fit_metrics)]
        xticks = [f"{_x*100:.2f}%" for _x in ls[0: len(x)]]
        metrics = self.df_fit_metrics.columns
        for metric in metrics:
            ys = self.df_fit_metrics[metric]
            plt.figure()
            plt.scatter(x, ys)
            plt.plot(x, ys)
            plt.xlabel("Percent used")
            plt.ylabel(f"Validation {metric}")
            plt.xticks(x, xticks)
            plt.title(f"Subset experiment for {len(self._train_dataset)} train size")
            out_file = f"{self.trainer.logger.log_dir}/subset_val_{metric}.png"
            plt.savefig(out_file)
            plt.close()

    @overrides
    def on_experiment_start(self):
        ls = np.linspace(1 / self.subsets, 1, self.subsets) if isinstance(self.subsets, int) else self.subsets
        subset_lens = [int(len(self._train_dataset) * x) for x in ls]
        indices = [np.random.choice(len(self._train_dataset), x, replace=False) for x in subset_lens]
        subsets = [Subset(self._train_dataset, ind) for ind in indices]
        self.dataloaders = [DataLoader(subset, **self._dataloader_params) for subset in subsets]
        self.tmp_train_dataloader = self._train_dataloaders

    @overrides
    def on_experiment_end(self):
        self._train_dataloaders = self.tmp_train_dataloader

    @overrides
    def on_iteration_start(self, ix: int):
        self._train_dataloaders = self.dataloaders[ix]

    @overrides
    def on_iteration_end(self, ix: int):
        return self._do_plot()

    def __len__(self):
        return self.num_subsets
