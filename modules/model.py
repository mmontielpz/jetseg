import pathmagic

import torch.nn as nn

from modules.encoder import JetNet
from modules.decoder import RegSeg

from torch.nn import functional as F

assert pathmagic

DEBUG = False


class JetSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = JetNet(cfg)
        self.decoder = RegSeg(cfg)

    def forward(self, x):

        # Get encoded feature maps
        encode_fts = self.encoder(x)

        # Get decoded feature maps
        decode_fts = self.decoder(encode_fts)

        # Interpolate decoded output as original img size
        x = F.interpolate(decode_fts, size=x.shape[-2:],
                          mode='bilinear', align_corners=False)

        return x


# Testing Blocks #
if DEBUG:

    import sys
    import torch
    from torchinfo import summary

    from modules.utils import ModelConfigurator

    # Create model configuration
    dataset_name = "camvid"
    cfg = ModelConfigurator(comb=0, mode=3,
                            dataset_name=dataset_name, debug=False)

    # Build model
    model = JetSeg(cfg)
    model = model.to("cuda")

    # Get torch summary
    summary(model, input_size=(1, 3, cfg.img_sizes[0], cfg.img_sizes[1]))

    # Create dummy tensor
    dummy_x = torch.rand(size=(3, cfg.img_sizes[0], cfg.img_sizes[1]))
    dummy_x = dummy_x.unsqueeze(0)

    # Send data to device
    dummy_x = dummy_x.to("cuda")
    print(f'[INFO] Input Tensor Shape: {dummy_x.shape}')

    # Evaluation
    model.eval()
    out = model(dummy_x)
    print(f'[INFO] Output Tensor Shape: {out.shape}')
