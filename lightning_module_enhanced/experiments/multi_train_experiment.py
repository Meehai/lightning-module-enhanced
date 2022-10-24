"""
MultiTrain experiment module. Wrapper on top of a regular trainer to train the model n times and pick the best result
plus statistics about them
"""
from pytorch_lightning import Trainer

from .experiment import Experiment

class MultiTrainExperiment(Experiment):
    """MultiTrain experiment implementation"""
    def __init__(self, trainer: Trainer, num_experiments: int):
        super().__init__(trainer)
        self.num_experiments = num_experiments

    def __len__(self):
        return self.num_experiments
