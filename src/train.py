import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from src.data.StratifiedSampler import StratifiedSampler
from src.data.dataset import BirdClefDataset
from src.data.get_classified_df import get_classified_df
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
import hydra

from src.config import ConfigHolder


def train():
    df = get_classified_df()
    print(ConfigHolder.config)
    # print(df)
    # print(len(df))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    targets = df["y"]
    train_index, val_index = next(sss.split(X=np.zeros(len(targets)), y=targets))
    dataset = BirdClefDataset(df)

    train_sampler = StratifiedSampler(train_index)
    val_sampler = StratifiedSampler(val_index)

    train_loader = DataLoader(
        dataset,
        batch_size=ConfigHolder.config.train.batch_size,
        sampler=train_sampler,
        num_workers=4,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=ConfigHolder.config.train.batch_size,
        sampler=val_sampler,
        num_workers=4,
        prefetch_factor=2,
    )

    iterD = iter(train_loader)
    waveforms, targets, _ = next(iterD)

    epoch_num = ConfigHolder.config.train.epoch_number
    for epoch in range(epoch_num):
        with tqdm(train_loader, unit="batch") as tepoch:
            for waveforms, targets, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(loss=0.0)
                tepoch.update()

                # model.train()
                # optimizer.zero_grad()
                # outputs = model(waveforms)
                # loss = criterion(outputs, targets)
                # loss.backward()
                # optimizer.step()
                # tepoch.set_postfix(loss=loss.item())

                # model.eval()
                # outputs = model(waveforms)
                # loss = criterion(outputs, targets)
                # tepoch.set_postfix(loss=loss.item())

    print("Training the model...")
