import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass
import enum

from src.configs.tf_efficientnet_b0_ns import EFFICIENTNET_CONFIG
from src.configs.tf_efficientnet_b3_ns import EFFICIENTNET_CONFIG_B3_NS


CONFIG = EFFICIENTNET_CONFIG 