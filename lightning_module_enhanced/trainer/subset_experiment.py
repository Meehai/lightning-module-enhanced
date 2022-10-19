"""
Subset experiment module. Wrapper on top of a regular trainer to train the model n times with increasing sizes
of the original dataset
"""
from copy import deepcopy
from typing import List
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

class SubsetExperiment:
    """Subset experiment implementation"""
    def __init__(self, trainer: Trainer, num_subsets: int):
        self.trainer = trainer
        self.num_subsets = num_subsets
        self.trainers = None
        # compatibility purposes
        self.logger = self.trainer.logger

    def _clone_trainer(self):
        trainers = []
        for i in range(self.num_subsets):
            new_trainer = deepcopy(self.trainer)
            version_prefix = f"{self.trainer.logger.version}/" if self.trainer.logger.version != "" else ""
            version = f"{version_prefix}subset_{i}_{self.num_subsets}"
            new_logger = TensorBoardLogger(save_dir=self.trainer.logger.save_dir,
                                           name=self.trainer.logger.name, version=version)
            new_trainer.logger = new_logger
            trainers.append(new_trainer)
        return trainers

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

        self.trainers = self._clone_trainer()
        res = []
        for i in range(self.num_subsets):
            seed_everything(i)
            # reset parameters, train, test on best ckpt and save the current plot at each iteration
            iter_model = deepcopy(model)
            iter_model.reset_parameters()

            self.trainers[i].fit(iter_model, dataloaders[i], val_dataloaders, *args, **kwargs)
            ckpt_path = self.trainers[i].checkpoint_callback.best_model_path
            res.append(self.trainers[i].test(iter_model, val_dataloaders, ckpt_path=ckpt_path)[0])
            self._do_plot(res, len(dataset))
            del iter_model
