import torch
import torch.nn as nn

import sys
import re
import math

from modules.afunc import REU, TanhExp


def validate_group_conv_sizes(in_ch, out_ch):
    group_sizes = []

    for i in range(1, in_ch + 1):
        if in_ch % i == 0 and out_ch % i == 0:
            group_sizes.append(i)

    return group_sizes


def get_groups(in_ch, out_ch):

    # Getting valid group convolution sizes
    group_sizes = validate_group_conv_sizes(in_ch, out_ch)

    # Select the mean value of groups
    mean = sum(group_sizes) / len(group_sizes)
    idx = min(
        range(len(group_sizes)), key=lambda i: abs(group_sizes[i] - mean))

    # Getting the number of groups (the closest to the average)
    return group_sizes[idx]


def map_jet_combination(jetseg_combination, stage, residual):

    if stage == 0 and residual:
        return jetseg_combination["stage0"]

    elif stage == 0 and not residual:
        return jetseg_combination["stage0"]
        # return jetseg_combination["stage0_residual"]

    elif stage == 1 and residual:
        return jetseg_combination["stage1_residual"]

    elif stage == 1 and not residual:
        return jetseg_combination["stage1_non_residual"]

    elif stage == 2 and residual:
        return jetseg_combination["stage2_residual"]

    elif stage == 2 and not residual:
        return jetseg_combination["stage2_non_residual"]

    elif stage == 3 and residual:
        return jetseg_combination["stage3_residual"]

    elif stage == 3 and not residual:
        return jetseg_combination["stage3_non_residual"]

    else:
        raise ValueError("[ERROR] Invalid JetSeg Combination")


def map_jet_conv(fmap_size, stage, n_block, residual,
                 in_channels, out_channels, jsc):

    # Get jet combination based on stage and residual
    jsc = map_jet_combination(jsc, stage, residual)

    stride = 1 if n_block != 0 else 2
    # residual = False if n_block == 0 else residual

    return JetConv(fmap_size, in_channels, out_channels,
                   residual=residual, jsc=jsc, s=stride)


def map_act_func(act_func):

    if act_func.lower() == "reu":
        ActFunc = REU()
    elif act_func.lower() == "tanhe":
        ActFunc = TanhExp()
    else:
        raise ValueError("The Activation Function doesn't exists")

    return ActFunc


def map_fmaps(fmap_size, sa_fmaps, eca_fmaps, cbam_fmaps):

    if fmap_size in cbam_fmaps:
        return "cbam", cbam_fmaps
    elif fmap_size in sa_fmaps:
        return "sa", sa_fmaps
    elif fmap_size in eca_fmaps:
        return "eca", eca_fmaps
    else:
        return None, None


def map_module(block_type, valid_maps, fmap_size, channels):

    if not block_type:
        return nn.Identity()

    if block_type.lower() == "cbam":

        if fmap_size in valid_maps:
            block = CBAM(channels)
        else:
            return nn.Identity()

    elif block_type.lower() == "eca":

        if fmap_size in valid_maps:
            block = ECA(channels)
        else:
            return nn.Identity()

    elif block_type.lower() == "sa":

        if fmap_size in valid_maps:
            block = SAM(channels)
        else:
            return nn.Identity()

    else:
        raise ValueError("Invalid block type")

    return block


class CBA(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, s=1, pad=0, dil=1, g=1,
                 bias=False, act=True):
        super(CBA, self).__init__()

        self.name = "CBA"
        self.ks = ks
        self.s = s
        self.dil = dil

        # Get the number of group channels
        self.g = get_groups(in_ch, out_ch)

        self.conv = nn.Conv2d(in_ch, out_ch, ks, s, pad, dil, g, bias)
        self.bn = nn.BatchNorm2d(out_ch)

        if act:
            self.act = REU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class CS(nn.Module):
    def __init__(self, g):
        super(CS, self).__init__()

        self.name = "cs"
        self.g = g

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.g

        # Reshape into (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.g, channels_per_group, height, width)

        # Transpose (swap) dimensions 1 and 2
        x = x.transpose(1, 2).contiguous()

        # Flatten back into (batch_size, num_channels, height, width)
        x = x.view(batch_size, num_channels, height, width)

        # Return the shuffled tensor
        return x


class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.name = "ECA"

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma

        # Calculate kernel size for 1D convolution
        kernel_size = int(abs((math.log2(channels) / gamma) + b / gamma))
        self.ks = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.ks,
            padding=(self.ks - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # 1D convolution to model channel-wise dependencies
        y = self.conv(
            y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Sigmoid activation for channel-wise attention weights
        y = self.sigmoid(y)

        # Multi-scale information fusion
        out = x * y.expand_as(x)

        return out


class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        self.name = "SAM"

        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Channel-wise attention
        channel_out = self.conv1(x)
        channel_attention = self.sigmoid(channel_out)

        # Spatial-wise attention
        avg_spatial_attention = self.avg_pool(channel_attention)
        max_spatial_attention = self.max_pool(channel_attention)
        spatial_attention = avg_spatial_attention + max_spatial_attention

        # Element-wise multiplication
        x = x * channel_attention * spatial_attention

        return x


class CA(nn.Module):
    def __init__(self, in_planes, reduction_ratio=8):
        super(CA, self).__init__()
        self.name = "CA"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes,
                             in_planes // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio,
                             in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        self.name = "SA"
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.name = "CBAM"
        self.channel_gate = CA(in_planes, reduction_ratio)
        self.spatial_gate = SA(kernel_size)

    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x


class JetConv(nn.Module):
    def __init__(self, input, in_ch, out_ch, s=1, residual=False, jsc=None):
        super().__init__()

        self.name = "JetConv"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.s = s
        self.jsc = jsc
        self.residual = residual
        self.lvl = 0

        # Getting input width and height sizes
        self.width, self.height = input

        # Asymmetric sides
        self.asym_sides = ["l", "r"]

        # Many convolutions ? (lvls ???)
        self.residuals = [] if jsc is not None else None

        # JetSeg Combination Validation
        if self.jsc is None:
            raise ValueError("[ERROR] The JetSeg Combination can't be NONE")

        # Build jet convs
        ops = nn.ModuleDict()
        self.ops = self.build_ops(ops)

        # Getting the number of features based on the lvl and residual
        lvl_fts = self.get_lvl_fts()

        # Get new channels
        new_channels = in_ch * lvl_fts

        if in_ch == out_ch:
            group_channels = get_groups(new_channels, in_ch)
        else:
            group_channels = get_groups(new_channels, out_ch)

        # Set the number of group convolutions
        self.g = group_channels

        # Point-wise Group Conv
        self.pw = nn.Conv2d(new_channels, out_ch, kernel_size=1,
                            stride=self.s, groups=group_channels, bias=False)

    def check_asym_side(self, op_name):

        match = re.search(r"lvl[0-" + str(self.lvl) + "]_side_r", op_name)

        if match:
            return True
        else:
            return False

    def get_residual(self):
        print(self.residuals)

        if len(self.residuals) <= 1:
            return self.residuals.pop(0)

        else:

            residuals = []
            results = []

            for i in range(len(self.residuals)-1):
                if i == 0:
                    res = self.residuals[i]
                else:
                    res = self.residuals[i] + residuals[-1]

                residuals.append(res)
                result = self.residuals[i+1] + res
                results.append(result)

            # Clear memory
            self.residuals = []

            # Concatenate the last residual and the last result
            return torch.cat([residuals[-1], results[-1]], dim=1)

    def get_lvl_fts(self):

        if self.residuals and self.lvl > 0:
            return 2
        else:
            return 1

    def get_dil_size(self):

        if self.ks <= 3:
            return 1
        elif self.ks == 5:
            return 2
        elif self.ks == 7:
            return 3

    def get_pad_size(self):
        # Get padding configuration sizes (left=height and right=width)
        pad_l = int(((self.height - 1) * (1 - 1) + self.dil * (self.ks - 1)) / 2)
        pad_r = int(((self.width - 1) * (1 - 1) + self.dil * (self.ks - 1)) / 2)

        return pad_l, pad_r

    def get_asym_cfg(self, asym_side):
        # pad_left, pad_right, ks_left, ks_right, dil_left, dil_right

        # Right side (1xN)
        if asym_side.lower() == "r":

            if self.ks == 3:
                return 0, 0, \
                    1, self.ks, \
                    1, 0
            else:
                return 0, self.pad_r, \
                    1, self.ks, \
                    0, self.dil

        # Left side (Nx1)
        elif asym_side.lower() == "l":

            if self.ks == 3:
                return 0, 0, \
                    self.ks, 1, \
                    0, 1
            else:
                return self.pad_l, 0, \
                    self.ks, 1, \
                    self.dil, 0
        else:
            raise ValueError("[ERROR] Incorrect asymmetric side")

    def get_ks(self, ks_comb):
        return max(max(ks) for ks in ks_comb)

    def get_lvl(self):

        if self.residual:
            return 0
        else:
            if self.ks <= 3:
                return 1
            elif self.ks == 5:
                return 2
            elif self.ks == 7:
                return 3

    def is_asymmetric(self, conv_comb):

        # Asymmetric convolution (more than one tuples) [(Nx1), (1xN)]
        if len(conv_comb) == 2 and all(isinstance(item, tuple) for item in conv_comb):
            return True
        # non-Asymmetric convolution (one tuple) [(NxN)]
        elif len(conv_comb) == 1 and isinstance(conv_comb[0], tuple):
            return False
        else:
            raise ValueError("[ERROR] Not asymmetric not standard convolution ???")

    def build_conv(self, ks_comb):
        ops = {}

        # Check if is asymmetric or non-asymmetric convolution
        if self.is_asymmetric(ks_comb):

            # Getting the kernel size
            self.ks = self.get_ks(ks_comb)

            # Get the dilation size
            self.dil = self.get_dil_size()

            # Get actual lvl
            self.lvl = self.get_lvl()

            # Get padding configuration sizes (left=height and right=width)
            self.pad_l, self.pad_r = self.get_pad_size()

            for side in self.asym_sides:

                # Get asymmetric configuration side
                pad_l, pad_r, ks_l, ks_r, \
                    dil_l, dil_r = self.get_asym_cfg(side)

                # Get asymmetric name
                asym_name = "lvl_" + str(self.lvl) + "_ks" + \
                    str(self.ks) + "_asymmetric_" + "side_" + side

                # Build Asymmetric Depth-wise Dilated Separable Convolution
                ops[asym_name] = nn.Conv2d(self.in_ch, self.in_ch,
                                           groups=self.in_ch,
                                           kernel_size=(ks_l, ks_r),
                                           stride=(1, 1),
                                           padding=(pad_l, pad_r), bias=False,
                                           dilation=(dil_l, dil_r))
        else:

            # Getting the kernel size
            self.ks = self.get_ks(ks_comb)

            # Get the dilation size
            self.dil = 1

            # Get actual lvl
            self.lvl = self.get_lvl()

            # Get padding configuration sizes (left=height and right=width)
            self.pad_l, self.pad_r = self.get_pad_size()

            # Get op name
            op_name = "lvl_" + str(self.lvl) + "_ks" + \
                str(self.ks) + "_standard"

            # Build Standard Convolution
            ops[op_name] = nn.Conv2d(self.in_ch, self.in_ch,
                                     groups=self.in_ch,
                                     padding=(self.pad_l, self.pad_r),
                                     kernel_size=self.ks, bias=False,
                                     dilation=self.dil)

        return ops

    def build_ops(self, ops):

        # Many convolutions ? (lvls ???)
        if len(self.jsc) == 1:

            # Flatten list or not in order to validate asymmetric or
            # non-asymmetric
            self.jsc = [elem for sublist in self.jsc for elem in sublist] if any(isinstance(elem, list) for elem in self.jsc) else self.jsc

            # Adding list type or not in order to validate asymmetric or
            # non-asymmetric
            self.jsc = [self.jsc] if not isinstance(self.jsc, list) else self.jsc

            # Get Jet Convolution
            jet_conv = self.build_conv(self.jsc)

            # Non-level Jet Convolution
            ops.update(jet_conv)

        else:

            # Iterate each jet convolution
            for conv in self.jsc:

                # Adding list type or not in order to validate asymmetric or
                # non-asymmetric
                conv = [conv] if not isinstance(conv, list) else conv

                # Get Jet Convolution
                jet_conv = self.build_conv(conv)

                # Level Jet Convolution
                ops.update(jet_conv)

        return ops

    def forward(self, x):

        # Residual block ?
        if self.residual:
            res = x.clone()

        for op_name, op in self.ops.items():

            # Process jet operations
            x = op(x)

            # Get the left asymmetric side
            asym_left = self.check_asym_side(op_name)

            # Add lvl residual ?
            if self.residuals and asym_left:
                self.residuals.append(x.clone())

        # If residual mode get the residual of all available lvls
        if self.residuals:
            x = self.get_residual()

        # Apply point-wise convolution
        x = self.pw(x)

        # Residual block ?
        if self.residual:
            x += res

        return x


class JetBlock(nn.Module):
    def __init__(self, block_cfg):
        super().__init__()

        stage = block_cfg["stage"]
        n_block = block_cfg["n_block"]
        jetseg_comb = block_cfg["jetseg_comb"]
        residual = block_cfg["residual"]
        in_ch = block_cfg["in_ch"]
        exp_ch = block_cfg["exp_ch"]
        out_ch = block_cfg["out_ch"]
        fmap_size = block_cfg["fmap_size"]
        last_fmap = block_cfg["last_fmap"]
        cbam_fmaps = block_cfg["cbam_fmaps"]
        ecam_fmaps = block_cfg["ecam_fmaps"]
        sam_fmaps = block_cfg["sam_fmaps"]

        # Setting residual
        self.residual = residual

        # Mean number of groups convolutions
        exp_conv_g = get_groups(in_ch, exp_ch)

        # Check if residual
        stride = 1 if self.residual else 2

        # Estimate point-wise group convolutions
        if residual:
            point_wise_g = get_groups(exp_ch, out_ch)
        else:
            point_wise_g = get_groups(exp_ch, out_ch)

        # Validate avoid downsampling the last block
        stride = 1 if fmap_size[0] <= last_fmap else stride

        # Getting spatial attention or efficient channel attention fmaps
        module_type, valid_fmaps = map_fmaps(fmap_size[0], sam_fmaps,
                                             ecam_fmaps, cbam_fmaps)

        # Create list of block operations
        self.ops = nn.ModuleList()

        # Residual block ?
        # if n_block == 0 or self.residual:
        if n_block == 0:

            # Add encoder block operations
            self.ops += [
                # Jet Convolution
                map_jet_conv(fmap_size, stage, n_block,
                             self.residual, in_ch, out_ch, jsc=jetseg_comb),
                # Batch Normalization
                nn.BatchNorm2d(out_ch),
                # Activation function
                REU(),
            ]

        else:

            # Add encoder block operations
            self.ops += [
                # 1x1 Expansion Conv
                nn.Conv2d(in_channels=in_ch, out_channels=exp_ch,
                          kernel_size=1, stride=stride, groups=exp_conv_g,
                          bias=False),
                # Batch Normalization
                nn.BatchNorm2d(exp_ch),
                # Activation function
                REU(),
                # Channel Shuffle
                CS(exp_conv_g),
                # Jet Convolution
                map_jet_conv(fmap_size, stage, n_block,
                             self.residual, exp_ch, exp_ch, jsc=jetseg_comb),
                # Efficient Channel Attention , Spatial Attention Module or
                # Convolutional Block Attention Module
                map_module(module_type, valid_fmaps, fmap_size[0], exp_ch),
                # Activation function
                REU(),
                # 1x1 Point-wise Reduce Conv
                nn.Conv2d(in_channels=exp_ch, out_channels=out_ch,
                          kernel_size=1, groups=point_wise_g, bias=False),
                # Dropout
                # nn.Dropout(p=0.20),
            ]

    def forward(self, x):

        for op in self.ops:

            if self.residual:
                res = x.clone()

            x = op(x)

            if self.residual:
                x += res

        return x
