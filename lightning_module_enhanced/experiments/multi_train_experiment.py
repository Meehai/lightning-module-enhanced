"""
MultiTrain experiment module. Wrapper on top of a regular trainer to train the model n times and pick the best result
plus statistics about them
"""
from overrides import overrides
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .experiment import Experiment

class MultiTrainExperiment(Experiment):
    """MultiTrain experiment implementation"""
    def __init__(self, trainer: Trainer, num_experiments: int):
        super().__init__(trainer)
        self.num_experiments = num_experiments

    def on_before_iteration(self, ix: int):
        breakpoint()

    def on_after_iteration(self, ix: int):
        breakpoint()

    # @overrides
    # def fit(self, model, train_dataloaders, val_dataloaders, *args, **kwargs):
    #     """The main function, uses same args as a regular pl.Trainer"""
    #     assert self.done is False, "Cannot fit twice"
    #     super().fit_setup(model, train_dataloaders, val_dataloaders)

    #     for i in range(self.num_experiments):
    #         self.do_one_iteration(i, model, train_dataloaders, val_dataloaders, *args, **kwargs)
    #         pd.DataFrame(self.fit_metrics).to_csv(f"{self.trainer.log_dir}/results.csv")
    #     self.done = True
    #     self.df_fit_metrics = pd.DataFrame(self.fit_metrics)
    #     self.ix = self.df_fit_metrics["loss"].argmin()

    def __len__(self):
        return self.num_experiments
