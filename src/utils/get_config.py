import yaml 
from omegaconf import DictConfig, OmegaConf
import hydra


class ConfigHolder:
    __instance = None
    
    @staticmethod
    def set_instance(cfg):
        ConfigHolder.__instance = cfg

    @staticmethod
    def get_instance():
        return ConfigHolder.__instance
        