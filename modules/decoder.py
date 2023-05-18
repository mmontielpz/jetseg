import pathmagic

import sys

import torch
from torch import nn
from torch.nn import functional as F
from modules.heads import DecoderHead
from modules.block import CBA


assert pathmagic


class RegSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.debug = cfg.debug
        self.num_classes = cfg.num_classes
        self.max_fts = cfg.max_features
        self.last_fts = cfg.last_features
        self.decoder_blocks = cfg.decoder_arch
        self.decoder_heads_out = cfg.decoder_heads_output_features

        # Create encoder blocks
        self.heads = self.create_heads()

        # Set conv fts out
        # first_features = self.decoder_heads_out // 2
        # self.decoder_heads_out //= 2
        self.conv = CBA(self.decoder_heads_out, self.decoder_heads_out)

        # TODO: Check and validate
        self.add_last_heads = False

        self.last_conv = CBA(self.decoder_heads_out * 2, self.last_fts)

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.4),
            nn.Conv2d(self.last_fts, self.num_classes, 1),
        )

    def create_heads(self):
        heads = nn.ModuleList()

        # For each encode block create a decoder head
        for decoder_block in self.decoder_blocks:

            if self.debug:
                print(f'[DEBUG] Decoder Block Configuration: {decoder_block}')
                print()

            # Getting in channels from decoder structure block
            # in_ch = decoder_block[-3]
            in_ch = decoder_block[-3]

            # Adding a decoder head
            heads += [
                DecoderHead(in_ch, self.decoder_heads_out)
            ]

        return heads

    def process_encoded_ch(self, x):

        if self.debug:
            for encode_block in x:
                print(f'[DEBUG] Encode Features: {encode_block.shape}')
                print()

        # pass input tensors through heads
        heads_out = []
        for i, head in enumerate(self.heads):
            x_i = x[i]

            if self.debug:
                print(f'[DEBUG] Decoder Head {i} Structure: {head}')
                print()

            if self.debug:
                print(f'[DEBUG] Decoder Head {i} Input size: {x_i.shape}')
                print()

            x_i = head(x_i)

            if self.debug:
                print(f'[DEBUG] Decoder Head {i} Output size: {x_i.shape}')
                print()

            heads_out.append(x_i)

        # interpolate and add feature maps
        for i in range(1, len(heads_out)):

            # Getting head out
            head_out_prev = heads_out[i-1]
            head_out = heads_out[i]

            # Interpolate and add feature maps
            head_out_prev = F.interpolate(head_out_prev,
                                          size=head_out.shape[-2:],
                                          mode='bilinear',
                                          align_corners=False)
            if self.debug:
                print(f'[DEBUG] Decoder Head {i} Up size: {head_out_prev.shape}')
                print()

            # Addition
            head_out = head_out + head_out_prev

        # Apply Convolution to previous head output
        if self.debug:
            print(f'[DEBUG] Conv for previous Head {len(heads_out)-1} (input): {head_out.shape}')
            print()

        head_out = self.conv(head_out)

        if self.debug:
            print(f'[DEBUG] Conv for previous Head {len(heads_out)-1} (output): {head_out.shape}')
            print()

        # Get last head output
        last_out = heads_out[-1]

        if self.debug:
            print(f'[DEBUG] Decoder Head {len(heads_out)} (last_output): {last_out.shape}')
            print()

        # Interpolate
        head_out = F.interpolate(head_out, size=last_out.shape[-2:],
                                 mode="bilinear", align_corners=False)

        if self.debug:
                print(f'[DEBUG] Decoder Head {len(heads_out)} Up size: {head_out.shape}')
                print()

        # Addition heads output and last head output ???
        if self.add_last_heads:
            last_out += head_out

        # if self.debug:
        #         print(f'[DEBUG] Last Output of Heads (after addition): {last_out.shape}')
        #         print()

        # Concatenate
        last_out = torch.cat((head_out, last_out), dim=1)

        if self.debug:
            print(f'[DEBUG] Concatenate size (cba input): {last_out.shape}')
            print()

        # Apply last convolution
        last_out = self.last_conv(last_out)

        if self.debug:
            print(f'[DEBUG] Last CBA (output/classifier input): {last_out.shape}')
            print()

        # Applying classifier
        output = self.classifier(last_out)

        if self.debug:
            print(f'[DEBUG] Conv Classifier (output): {output.shape}')
            print()

        return output

    def forward(self, x):

        res = self.process_encoded_ch(x)

        return res
