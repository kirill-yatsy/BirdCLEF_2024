from typing import List, Mapping, Any, Dict
from lightning import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.accelerators import CUDAAccelerator
import torchmetrics 
import time
import lightning as L
import torch
from enum import Enum
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pl_bolts.callbacks import PrintTableMetricsCallback
from pl_bolts.callbacks import TrainingDataMonitor

from dataclasses import dataclass

from src.config import CONFIG
from src.configs.base_config import BirdConfig


@dataclass
class BirdCleffModelConfig(L.LightningModule):
    num_classes: int

    validation_epoch_outputs: Dict[str, torch.Tensor]
    training_epoch_outputs: Dict[str, torch.Tensor]


 

class F1Callback(Callback):
    def setup(
        self, trainer: L.Trainer, pl_module: BirdCleffModelConfig, stage: str
    ) -> None:
        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=pl_module.num_classes
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        f1 = self.f1(
            pl_module.training_epoch_outputs["y_hat"],
            pl_module.training_epoch_outputs["y"],
        )
        trainer.logger.log_metrics({"train_batch_f1": f1.item()})

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: BirdCleffModelConfig
    ) -> None: 
        f1 = self.f1(
            pl_module.training_epoch_outputs["y_hat"],
            pl_module.training_epoch_outputs["y"],
        )
        trainer.logger.log_metrics({"train_epoch_f1": f1.item()})

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: BirdCleffModelConfig
    ) -> None:
        f1 = self.f1(
            pl_module.validation_epoch_outputs["y_hat"],
            pl_module.validation_epoch_outputs["y"],
        )
        trainer.logger.log_metrics({"val_epoch_f1": f1.item()})


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



def get_callbacks() -> List[L.Callback] | L.Callback:
    return [
        # PrintTableMetricsCallback(),
        # CUDAAccelerator(),
        # TrainingDataMonitor(),
        Timer(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=CONFIG.train.save_model_every_epoch_keep_last,
            mode="min",
            dirpath=CONFIG.train.save_model_path,
            filename="model-fine-tune1-{epoch:02d}-{val_loss:.2f}" if CONFIG.train.fine_tune else "model-{epoch:02d}-{train_loss:.2f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=True,
            mode="min",
        ), 
        F1Callback(),
        AccuracyCallback(),
        # ConfusionMatrixCallback(),
    ]
