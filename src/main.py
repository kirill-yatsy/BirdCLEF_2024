import argparse
from src.config import ConfigHolder
from src.train import train
from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.pytorch import Trainer, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="exp1")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ConfigHolder.set_instance(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["train", "test"], default="train")

    args = parser.parse_args()

    if args.operation == "train":
        train()


if __name__ == "__main__":
    main()
