import yaml
 
from dataclasses import dataclass
import enum

from src.configs.dpn68b import DPN68B_CONFIG
from src.configs.efficientnet_b2 import EFFICIENTNET_CONFIG_B2
from src.configs.seresnext50d_32x4d import SERESNEXT26T_32X4D_CONFIG
from src.configs.tf_efficientnet_b0_ns import EFFICIENTNET_CONFIG
from src.configs.efficientnet_b3 import EFFICIENTNET_CONFIG_B3


CONFIG = EFFICIENTNET_CONFIG_B2