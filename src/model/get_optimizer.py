from torch import optim, nn, utils, Tensor


def get_optimizer(model) -> nn.Module:
    return optim.Adam(model.parameters(), lr=0.0001)