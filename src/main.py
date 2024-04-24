import argparse
from src.config import init_config
from src.train import train
from omegaconf import DictConfig, OmegaConf
import hydra

from src.utils.get_config import ConfigHolder

@hydra.main(version_base=None, config_path="../configs", config_name="exp1")   
def main(cfg: DictConfig) -> None: 
    ConfigHolder.set_instance(OmegaConf.to_yaml(cfg))
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', choices=['train', 'test'], default='train')

    args = parser.parse_args()

    if args.operation == 'train':
        train()

 
if __name__ == '__main__':
    main()