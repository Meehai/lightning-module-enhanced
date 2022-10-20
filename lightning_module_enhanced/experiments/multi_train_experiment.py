"""
MultiTrain experiment module. Wrapper on top of a regular trainer to train the model n times and pick the best result
plus statistics about them
"""
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .experiment import Experiment

class MultiTrainExperiment(Experiment):
    """MultiTrain experiment implementation"""
    def __init__(self, trainer: Trainer, num_experiments: int):
        super().__init__(trainer)
        self.num_experiments = num_experiments

        # post fit stuff
        self.df_res = None
        self.ix = None

    def fit(self, model, train_dataloaders, val_dataloaders, *args, **kwargs):
        """The main function, uses same args as a regular pl.Trainer"""
        assert self.done is False, "Cannot fit twice"

        for cb in model.configure_callbacks():
            assert not isinstance(cb, ModelCheckpoint), "Subset experiment cannot have another ModelCheckpoint"

        for i in range(self.num_experiments):
            self.do_one_iteration(i, model, train_dataloaders, val_dataloaders, *args, **kwargs)
            pd.DataFrame(self.fit_metrics).to_csv(f"{self.trainer.log_dir}/results.csv")
        self.done = True
        self.df_fit_metrics = pd.DataFrame(self.fit_metrics)
        self.ix = self.df_fit_metrics["loss"].argmin()

    @property
    def checkpoint_callback(self):
        assert self.done is True
        return self.fit_trainers[self.ix].checkpoint_callback

    def __len__(self):
        return self.num_experiments
