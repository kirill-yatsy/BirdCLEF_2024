import yaml 
from omegaconf import DictConfig, OmegaConf
import hydra
 
@hydra.main(version_base=None, config_path="../configs", config_name="exp1")   
def init_config(cfg: DictConfig):
    print("Config loaded")