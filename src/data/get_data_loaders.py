import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from src.config import ConfigHolder
from src.data.StratifiedSampler import StratifiedSampler
from src.data.dataset import BirdClefDataset
from src.data.get_classified_df import get_classified_df
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
import hydra


def get_data_loaders():
    df = get_classified_df()
    # print(ConfigHolder.config)

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
        num_workers=ConfigHolder.config.train.num_workers,
        prefetch_factor=4, 
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=ConfigHolder.config.train.batch_size,
        sampler=val_sampler,
        num_workers=ConfigHolder.config.train.num_workers,
        prefetch_factor=4, 
        # multiprocessing_context="spawn",
    )

    return df, train_loader, val_loader
