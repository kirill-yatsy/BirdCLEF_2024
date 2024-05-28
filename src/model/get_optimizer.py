from torch import optim, nn, utils, Tensor

from src.config import CONFIG


def get_optimizer(model) -> nn.Module:
    return optim.Adam(model.parameters(), lr=CONFIG.train.lr)