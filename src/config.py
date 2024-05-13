import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass
import enum

from src.configs.tf_efficientnet_b0_ns import EFFICIENTNET_CONFIG


CONFIG = EFFICIENTNET_CONFIG