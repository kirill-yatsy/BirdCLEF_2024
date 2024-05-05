from typing import List
from lightning import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from src.config import BirdConfig, ConfigHolder
import time
import lightning as L

from src.model.custom_model_lightning import CustomModelLightning

class IterationTimeCallback(Callback):
    def on_train_start(self, trainer: L.Trainer, pl_module: CustomModelLightning):

        self.train_time = time.time()
    def on_train_end(self, trainer: L.Trainer, pl_module):
        minutes = (time.time() - self.train_time) / 60
        trainer.logger.log_metrics({
            "training_time": minutes
        })

    def on_validation_start(self, trainer, pl_module):
        self.val_time = time.time()
    
    def on_validation_end(self, trainer, pl_module):
        minutes = (time.time() - self.val_time) / 60
        trainer.logger.log_metrics({
            "validation_time": minutes
        })
        
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
        IterationTimeCallback()
    ]