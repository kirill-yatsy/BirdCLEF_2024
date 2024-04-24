

# import numpy as np
# from sklearn.model_selection import StratifiedShuffleSplit
# from src.data.StratifiedSampler import StratifiedSampler
# from src.data.dataset import BirdClefDataset
# from src.data.get_classified_df import get_classified_df
# from torch.utils.data import DataLoader
# from hydra.core.config_store import ConfigStore
# from hydra.core.hydra_config import HydraConfig
# from dataclasses import dataclass
import hydra

from src.utils.get_config import ConfigHolder


def train(): 
    # df = get_classified_df() 
    print(ConfigHolder.get_instance())
    # print(df)
    # print(len(df))
    
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
    # targets = df['y']
    # train_index, val_index = next(sss.split(X=np.zeros(len(targets)), y=targets)) 
    # dataset = BirdClefDataset(df)

    # train_sampler = StratifiedSampler(train_index)
    # val_sampler = StratifiedSampler(val_index)

    # train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    # val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

    # iterD = iter(train_loader)
    # iaaa = next(iterD)
    # # print(iaaa)
    # print('Training the model...')