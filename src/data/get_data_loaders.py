import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm 
from src.configs.base_config import BirdConfig
from src.data.dataset import BirdClefDataset
from src.data.get_classified_df import get_classified_df
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass
import hydra

import torch.utils.data 

class StratifiedSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

def get_data_loaders(config: BirdConfig):
    df = get_classified_df(config)  
    
    # Counting the instances per class
    class_counts = df['y'].value_counts()

    # Filtering out classes with only one instance
    single_instance_classes = class_counts[class_counts == 1].index
    single_instance_indices = df[df['y'].isin(single_instance_classes)].index

    # Data excluding single-instance classes
    filtered_df = df[~df.index.isin(single_instance_indices)]

    # Preparing the data for StratifiedShuffleSplit
    targets = filtered_df["y"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, val_index = next(sss.split(X=np.zeros(len(targets)), y=targets))

    # Converting these indices to match the original dataframe indices
    train_index = filtered_df.iloc[train_index].index
    val_index = filtered_df.iloc[val_index].index

    # Adding the single-instance classes to the training and validation set
    # train_index = train_index.union(single_instance_indices)
    # val_index = val_index.union(single_instance_indices)
 
    dataset = BirdClefDataset( df, config)

    train_sampler = StratifiedSampler(train_index)
    val_sampler = StratifiedSampler(val_index)

 
    train_loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        num_workers=config.train.num_workers,
        prefetch_factor=2, 
        # multiprocessing_context=None if config.train.fast_dev_run else "spawn",
        # persistent_workers=True 
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        sampler=val_sampler,
        num_workers=config.train.num_workers,
        prefetch_factor=2, 
        # multiprocessing_context=None if config.train.fast_dev_run else "spawn",
        # persistent_workers=True
    )

    return df, train_loader, val_loader
