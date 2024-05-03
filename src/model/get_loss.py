from torch import optim, nn, utils, Tensor


def get_loss() -> nn.Module:
    return nn.CrossEntropyLoss()