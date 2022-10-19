"""
MultiTrain experiment module. Wrapper on top of a regular trainer to train the model n times and pick the best result
plus statistics about them
"""
from copy import deepcopy
from typing import List
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

class MultiTrainExperiment:
    """MultiTrain experiment implementation"""
    def __init__(self, trainer: Trainer, num_experiments: int):
        self.trainer = trainer
        self.num_experiments = num_experiments
        self.done = False

        # compatibility purposes
        self.logger = self.trainer.logger
        # set after experiment is done to the best internal trainer
        # self.checkpoint_callback = None

    def _clone_trainer(self):
        trainers = []
        for i in range(self.num_experiments):
            new_trainer = deepcopy(self.trainer)
            version_prefix = f"{self.trainer.logger.version}/" if self.trainer.logger.version != "" else ""
            version = f"{version_prefix}experiment_{i}_{self.num_experiments}"
            new_logger = TensorBoardLogger(save_dir=self.trainer.logger.save_dir,
                                           name=self.trainer.logger.name, version=version)
            new_trainer.logger = new_logger
            trainers.append(new_trainer)
        return trainers

    def fit(self, model, train_dataloaders, val_dataloaders, *args, **kwargs):
        """The main function, uses same args as a regular pl.Trainer"""
        assert self.done is False

        for cb in model.configure_callbacks():
            assert not isinstance(cb, ModelCheckpoint), "Subset experiment cannot have another ModelCheckpoint"

        trainers: List[Trainer] = self._clone_trainer()
        res = []
        for i in range(self.num_experiments):
            seed_everything(i)
            # reset parameters, train, test on best ckpt and save the current plot at each iteration
            iter_model = deepcopy(model)
            iter_model.reset_parameters()
            trainers[i].fit(iter_model, train_dataloaders, val_dataloaders, *args, **kwargs)
            ckpt_path = trainers[i].checkpoint_callback.best_model_path
            res.append(self.trainer.test(iter_model, val_dataloaders, ckpt_path=ckpt_path)[0])
            del iter_model
        self.done = True

        self.df_res = pd.DataFrame(res)
        self.trainers = trainers
        # ix = self.df_res["loss"].argmin()
        # self.checkpoint_callback = self.trainers[ix].checkpoint_callback
        # out_file = f"{self.trainer.logger.log_dir}/results.csv"
        # self.df_res.to_csv(out_file)

    def test(self, *args, **kwargs):
        assert self.done is True
        ix = self.df_res["loss"].argmin()
        return self.trainers[ix].test(*args, **kwargs)

    @property
    def checkpoint_callback(self):
        assert self.done is True
        ix = self.df_res["loss"].argmin()
        return self.trainers[ix].checkpoint_callback
