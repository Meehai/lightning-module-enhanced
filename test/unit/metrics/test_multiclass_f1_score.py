import torch as tr
from torch import nn
from torch.utils.data import DataLoader
from lightning_module_enhanced import LME
from lightning_module_enhanced.metrics import MultiClassF1Score
from pytorch_lightning.trainer import Trainer

counters = {"metric_grad": 0, "metric_non_grad": 0}

class TrainReader:
    def __getitem__(self, ix):
        return tr.randn(3, 10), tr.randn(3, 3)

    def __len__(self):
        return 5

def test_multi_class_f1_score_1():
    model = LME(nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 3)))
    model.criterion_fn = lambda y, gt: (y - gt).abs().mean()
    model.metrics = {"f1_score": MultiClassF1Score(num_classes=3)}
    model.optimizer = tr.optim.SGD(model.parameters(), lr=0.01)
    model.model_algorithm = lambda model, batch: (y := model(batch[0]), model.lme_metrics(y, batch[1]), *batch)

    Trainer(max_epochs=10).fit(model, DataLoader(TrainReader()))

if __name__ == "__main__":
    test_multi_class_f1_score_1()
