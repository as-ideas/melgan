import torch
import torch.nn as nn
import torch.nn.functional as F

from model.discriminator import STFTDiscriminator
from utils.stft import STFT
from .discriminator import Discriminator
from .identity import Identity


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)] + [STFTDiscriminator()]
        )
        
        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

        self.stft = STFT()


    def forward(self, x):
        ret = list()

        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))


        return ret # [(feat, score), (feat, score), (feat, score)]
