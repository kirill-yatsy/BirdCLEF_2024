import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F

from src.config import CONFIG

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # if CONFIG.augmentations.useMixup:
        #     inputs = inputs.argmax(dim=1)
        
        # convert inputs to float
        inputs = inputs.float()
        targets = targets.float()

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def get_loss() -> nn.Module:
    # return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()
    # return FocalLoss()