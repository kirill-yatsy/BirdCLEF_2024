from typing import List, Mapping, Any, Dict
from lightning import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics
from src.config import BirdConfig, ConfigHolder
import time
import lightning as L
import torch
from enum import Enum
from lightning.pytorch.utilities.types import STEP_OUTPUT

from dataclasses import dataclass


@dataclass
class BirdCleffModelConfig(L.LightningModule):
    num_classes: int

    validation_epoch_outputs: Dict[str, torch.Tensor]
    training_epoch_outputs: Dict[str, torch.Tensor]


class IterationTimeCallback(Callback):
    def on_train_start(self, trainer: L.Trainer, pl_module):
        self.train_time = time.time()

    def on_train_end(self, trainer: L.Trainer, pl_module):
        minutes = (time.time() - self.train_time) / 60
        trainer.logger.log_metrics({"training_time_minutes": minutes})

    def on_validation_start(self, trainer, pl_module):
        self.val_time = time.time()

    def on_validation_end(self, trainer, pl_module):
        minutes = (time.time() - self.val_time) / 60
        trainer.logger.log_metrics({"validation_time_minutes": minutes})


class F1Callback(Callback):
    def setup(
        self, trainer: L.Trainer, pl_module: BirdCleffModelConfig, stage: str
    ) -> None:
        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=pl_module.num_classes
        )

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: BirdCleffModelConfig
    ) -> None:
        if pl_module.config.train.fast_dev_run:
            return

        f1 = self.f1(
            pl_module.training_epoch_outputs["y_hat"],
            pl_module.training_epoch_outputs["y"],
        )
        trainer.logger.log_metrics({"train_f1": f1.item()})

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: BirdCleffModelConfig
    ) -> None:
        f1 = self.f1(
            pl_module.validation_epoch_outputs["y_hat"],
            pl_module.validation_epoch_outputs["y"],
        )
        trainer.logger.log_metrics({"val_f1": f1.item()})


class AccuracyCallback(Callback):
    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        self.accuracy = torchmetrics.Accuracy(
            num_classes=pl_module.num_classes, task="multiclass"
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.accuracy.update(y_hat, y)
        trainer.logger.log_metrics(
            {"train_batch_accuracy": self.accuracy(y_hat, y).item()},
            step=trainer.global_step,
        )
        # pl_module.log(
        #     "pl_module.log", self.accuracy(y_hat, y).item()
        # )
        # trainer.logger.log_metrics(
        #     {"trainer.logger.log_metrics": self.accuracy(y_hat, y).item()}
        # )
        # wandb.log({"train_batch_accuracy": self.accuracy(y_hat, y).item()})
        # pl_module.log(
        #     {"train_batch_accuracy": self.accuracy(y_hat, y).item()}
        # )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.accuracy.update(
            pl_module.validation_epoch_outputs["y_hat"],
            pl_module.validation_epoch_outputs["y"],
        )

        trainer.logger.log_metrics(
            {
                "validation_epoch_accuracy": self.accuracy(
                    pl_module.validation_epoch_outputs["y_hat"],
                    pl_module.validation_epoch_outputs["y"],
                ).item()
            }
        )

        self.accuracy.reset()


# class ConfusionMatrixCallback(Callback):
#     def setup(
#         self, trainer: L.Trainer, pl_module: BirdCleffModelConfig, stage: str
#     ) -> None:
#         self.confusion_matrix = torchmetrics.ConfusionMatrix(
#             task="multiclass", num_classes=pl_module.num_classes
#         )

#     def on_train_epoch_end(
#         self, trainer: L.Trainer, pl_module: BirdCleffModelConfig
#     ) -> None:
#         cm = self.confusion_matrix(
#             pl_module.training_epoch_outputs["y_hat"],
#             pl_module.training_epoch_outputs["y"],
#         )
#         trainer.logger.log_metrics({"train_confusion_matrix": cm.item()})
#         self.confusion_matrix.reset()

#     def on_validation_epoch_end(
#         self, trainer: L.Trainer, pl_module: BirdCleffModelConfig
#     ) -> None:
#         cm = self.confusion_matrix(
#             pl_module.validation_epoch_outputs["y_hat"],
#             pl_module.validation_epoch_outputs["y"],
#         )
#         # trainer.logger.log_metrics({"val_confusion_matrix": cm.item()})
#         trainer.logger.log_metrics({"val_confusion_matrix": cm.item()})
#         self.confusion_matrix.reset()


def get_callbacks(config: BirdConfig) -> List[L.Callback] | L.Callback:
    return [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=config.train.save_model_every_epoch_keep_last,
            mode="min",
            dirpath=config.train.save_model_path,
            filename="model-{epoch:02d}-{val_loss:.2f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=True,
            mode="min",
        ),
        IterationTimeCallback(),
        F1Callback(),
        AccuracyCallback(),
        # ConfusionMatrixCallback(),
    ]
