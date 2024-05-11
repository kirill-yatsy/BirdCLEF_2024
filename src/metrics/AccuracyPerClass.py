from torchmetrics import Metric
import torch
from typing import Optional, Union, Sequence, Tuple
import wandb
from wandb import plot


class AccuracyPerClass(Metric):
    def __init__(self, targetToLabelMapper: dict, **kwargs):
        super().__init__(**kwargs)
        self.add_state("predicted", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([]), dist_reduce_fx="cat")
        self.targetToLabelMapper = targetToLabelMapper

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.predicted = torch.cat([self.predicted, preds])
        self.target = torch.cat([self.target, target])

    def compute(self) -> list[list[Union[str, bool]]]:
        target = self.target.cpu().numpy()
        predicted = self.predicted.cpu().numpy()
        output = []
        # output
        for t, p in zip(target, predicted):
            output.append([self.targetToLabelMapper[t], t == p])

        return output

    def log(self, title: str):
        data = self.compute()
        table = wandb.Table(data=data, columns=["label", "value"])
        bar_chart = plot.bar(table, "label", "value", title)
        wandb.log({"accuracy_per_class_id": bar_chart})
