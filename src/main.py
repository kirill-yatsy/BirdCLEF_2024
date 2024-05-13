import argparse 
from src.config import CONFIG
from src.train import train
from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.pytorch import Trainer, seed_everything


def main() -> None:
    seed_everything(CONFIG.seed)  
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["train", "test"], default="train")

    args = parser.parse_args()

    if args.operation == "train":
        train()


if __name__ == "__main__":
    main()
