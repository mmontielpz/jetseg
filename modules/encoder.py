import pathmagic

import sys
from torch import nn
from modules.block import JetBlock

assert pathmagic


class JetNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.debug = cfg.debug
        self.arch = cfg.encoder_arch
        self.jsc = cfg.jetseg_comb
        self.num_classes = cfg.num_classes
        self.decoder_fmaps = self.get_decoder_fmaps(cfg.decoder_arch)
        self.cbam_fmaps = cfg.cbam_fmaps
        self.sam_fmaps = cfg.sam_fmaps
        self.ecam_fmaps = cfg.ecam_fmaps
        self.last_fmap = cfg.last_fmap
        self.encoder = self.create_encoder()

    def get_decoder_fmaps(self, decoder_arch):
        return [block_cfg[0] for block_cfg in decoder_arch]

    def create_encoder(self):
        enc = nn.ModuleList()

        # Block counter
        self.n_block = 0

        # Building encoder/feature extractor
        for arch_cfg in self.arch:

            if self.debug:
                print(f'[DEBUG] Encoder Block Configuration: {arch_cfg}')
                print()

            # Getting Block Configuration
            fmap_h, fmap_w, stage, residual, \
                self.in_ch, exp_ch, out_ch = arch_cfg

            # Create feature map size
            fmap = [fmap_h, fmap_w]

            # Create block configuration
            block_cfg = {
                "stage": stage,
                "n_block": self.n_block,
                "jetseg_comb": self.jsc,
                "in_ch": self.in_ch,
                "exp_ch": exp_ch,
                "out_ch": out_ch,
                "residual": residual,
                "fmap_size": fmap,
                "last_fmap": self.last_fmap,
                "cbam_fmaps": self.cbam_fmaps,
                "ecam_fmaps": self.ecam_fmaps,
                "sam_fmaps": self.sam_fmaps,
            }

            # Adding block
            enc += [
                JetBlock(block_cfg)
            ]

            # Update input features
            self.in_ch = out_ch

            # Counting num of blocks
            self.n_block += 1

        return enc

    def feature_encoding(self, x):
        fmap_outs = {}

        for i, layer in enumerate(self.encoder):

            # print(f'[DEBUG] Layer: {layer}')
            # print(f'[DEBUG] Layer Input: {x.shape}')
            # print()
            x = layer(x)
            # print(f'[DEBUG] Layer Output: {x.shape}')
            # print()

            # Getting the feature map (max size)
            fmap_max = max(x.shape[-1], x.shape[-2])

            if fmap_max in self.decoder_fmaps:

                # Adding feature map outputs
                fmap = str(x.shape[-2]) + "x" + str(x.shape[-1])
                fmap_outs[fmap] = x

        # Get feature map outputs
        fmap_outs = fmap_outs.values()
        fmap_outs = list(fmap_outs)
        fmap_outs.reverse()

        return fmap_outs

    def forward(self, x):

        x = self.feature_encoding(x)

        return x
