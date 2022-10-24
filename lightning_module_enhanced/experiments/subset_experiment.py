"""
Subset experiment module. Wrapper on top of a regular trainer to train the model n times with increasing sizes
of the original dataset
"""
from overrides import overrides
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from .experiment import Experiment

class SubsetExperiment(Experiment):
    """Subset experiment implementation"""
    def __init__(self, trainer: Trainer, num_subsets: int):
        super().__init__(trainer)
        self.num_subsets = num_subsets
        self.dataloaders = None
        self.tmp_train_dataloader = None

    def _do_plot(self):
        ls = np.linspace(1 / self.num_subsets, 1, self.num_subsets)[0: len(self.df_fit_metrics)]
        x = np.arange(len(self.df_fit_metrics))
        metrics = self.df_fit_metrics.columns
        for metric in metrics:
            ys = self.df_fit_metrics[metric]
            plt.figure()
            plt.scatter(x, ys)
            plt.plot(x, ys)
            plt.xticks(x, [f"{x*100:.2f}%" for x in ls])
            plt.xlabel("Percent used")
            plt.ylabel(f"Validation {metric}")
            plt.title(f"Subset experiment for {len(self._train_dataset)} train size")
            out_file = f"{self.trainer.logger.log_dir}/subset_val_{metric}.png"
            plt.savefig(out_file)
            plt.close()

    @overrides
    def on_fit_start(self):
        ls = np.linspace(1 / self.num_subsets, 1, self.num_subsets)
        subset_lens = [int(len(self._train_dataset) * x) for x in ls]
        indices = [np.random.choice(len(self._train_dataset), x, replace=False) for x in subset_lens]
        subsets = [Subset(self._train_dataset, ind) for ind in indices]
        self.dataloaders = [DataLoader(subset, **self._dataloader_params) for subset in subsets]
        self.tmp_train_dataloader = self._train_dataloaders

    @overrides
    def on_fit_end(self):
        self._train_dataloaders = self.tmp_train_dataloader

    @overrides
    def on_iteration_start(self, ix: int):
        self._train_dataloaders = self.dataloaders[ix]

    @overrides
    def on_iteration_end(self, ix: int):
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
