import pathmagic

from torch import nn
from modules.block import CBA

assert pathmagic


class DecoderHead(nn.Module):
    def __init__(self, in_ch, out_ch, drop=None):
        super().__init__()

        self.head = CBA(in_ch, out_ch)

        if drop is not None:
            self.drop = nn.Dropout(p=drop)
        else:
            self.drop = nn.Identity()

    def forward(self, x):
        x = self.head(x)
        x = self.drop(x)

        return x
