import numpy as np
import random
import os
import torch
import torch
import torch.nn as nn

def set_seed(seed = 43):
    '''sets the seed of the entire notebook so results are the same every time we run'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# https://arxiv.org/abs/1708.02002
# To be implemented in future
class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # inputs..cpu(),targets.cpu()
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss
