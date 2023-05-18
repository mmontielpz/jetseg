import pathmagic

import sys
import math
import torch
import torch.nn as nn

from functools import reduce

from modules.block import JetConv, CBA, CS, ECA, SAM, CBAM
from modules.afunc import REU, TanhExp

assert pathmagic

# some pytorch low-level memory management constant
# the minimal allocate memory size (Byte)
PYTORCH_MIN_ALLOCATE = 2 ** 9

ACT_FUNCS = ["reu", "tanhexp"]
POOL_OPS = ["max_pool", "avg_pool", "adap_avg_pool", "adap_max_pool"]
CUSTOM_OPS = ["jetconv", "cba", "cs", "eca", "sam", "cbam"]

SIZE_MAP = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1
}


def get_data_type_size(data_type):

    if data_type in SIZE_MAP:
        return SIZE_MAP[data_type]
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


class ModuleProfiler:
    def __init__(self, module, input, output, custom_ops=False):
        super(ModuleProfiler, self).__init__()

        # Getting module name
        self.module = module
        self.input = input
        self.output = output
        self.name = self.get_short_name(module.__class__.__name__.lower())
        self.custom_ops = custom_ops

        self.module_params = list(module.parameters())
        self.module_buffers = list(module.buffers())

        # Getting the input tensor data type
        self.bit_depth = get_data_type_size(input.dtype)

        self.flops = 0
        self.macs = 0
        self.model_mem = 0
        self.tot_mem = 0

    def get_element_mem(self, tensor):

        element_size = tensor.element_size()
        fact_numel = tensor.storage().size()
        fact_memory_size = fact_numel * element_size

        # For any tensor PyTorch Allocate at least 512 Bytes, so we round up to
        # a multiple of 512
        memory_size = math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) \
            * PYTORCH_MIN_ALLOCATE

        return memory_size

    def get_mem(self):

        if self.name in CUSTOM_OPS:
            return 0

        else:

            # For each parameter get its required memory
            params_mem = 0
            for param in self.module_params:

                # Get parameter memory
                params_mem += self.get_element_mem(param)

            # For each buffer get its required memory
            buffers_mem = 0
            for buffer in self.module_buffers:

                # Get buffer memory
                buffers_mem += self.get_element_mem(buffer)

            # Get required memory for input and output tensors
            input_mem = self.get_element_mem(self.input)
            output_mem = self.get_element_mem(self.output)

            # Compute the total required memory
            model_mem = params_mem + buffers_mem

        return model_mem + input_mem + output_mem

    def get_flops(self):

        if self.name in CUSTOM_OPS:
            return 0

        if self.name == "conv":
            # Formula (C_in x C_out x K_w x K_h x H_in x W_in) / s^2 x g

            # Getting the shape of the input tensor (H_in x W_in)
            input_dims = torch.tensor(self.input.shape[2:])

            # Getting the kernel size (K_w x K_h)
            kernel_dims = torch.tensor(self.module.kernel_size)

            # Getting the input and output (C_in x C_out)
            in_channels = self.module.in_channels
            out_channels = self.module.out_channels

            # Getting the stride size
            stride = list(self.module.stride)
            stride = max(stride)

            # Getting the number of groups
            g = self.module.groups

            # Compute FLOPs (Convolution groups formula)
            flops = in_channels * out_channels * torch.prod(kernel_dims) * \
                torch.prod(input_dims)

            flops = flops / ((stride ** 2) * g)

        elif self.name in POOL_OPS:
            # Formula (avg pool) --> C_in x H_out x W_out x K_w x K_h
            # Formula (max pool) --> C_in x H_out x W_out x K_w x K_h x 2

            # Getting the shape of the input tensor (W_out x H_out)
            output_dims = torch.tensor(self.output[-2:])

            # Getting the input channels (C_in)
            in_channels = self.input.shape[1]

            # Computing flops for pool operations
            if self.name == "adap_avg_pool":
                # flops = in_channels * torch.prod(output_dims)
                flops = 0

            elif self.name == "avg_pool" or self.name == "max_pool":

                # Getting the kernel size (K_w x K_h)
                kernel_dims = torch.tensor(self.module.kernel_size)

                # Computing flops
                flops = in_channels * torch.prod(output_dims) * \
                    torch.prod(kernel_dims)

            if self.name == "max_pool":
                flops *= 2

        elif self.name == "bn":
            # Formula --> 4 x C_in x W_out x H_out

            # Getting the input channels (C_in)
            in_channels = self.input.shape[1]

            # Getting the output tensor shape (W_out x H_out)
            output_dims = torch.tensor(self.output.shape[-2:])

            # Computing flops
            flops = 4 * in_channels * torch.prod(output_dims)

        elif self.name in ACT_FUNCS:
            # Formula --> C_in x W x H

            # Getting the input tensor shape
            input_shape = list(self.input.size())

            # Computing flops
            flops = reduce(lambda x, y: x * y, input_shape)

        elif self.name == "linear":
            # Formula --> (N_in x N_out) + bias

            # Getting the input tensor shape
            input_dims = torch.tensor(self.input.shape)

            # Getting the output tensor shape
            output_dims = torch.tensor(self.output.shape[-2:])

            # Getting the bias flops
            bias_flops = output_dims.shape[-1] if self.module.bias is not None else 0

            # Computing flops
            flops = torch.prod(input_dims) * torch.prod(output_dims) \
                + bias_flops
        else:
            raise ValueError(f"Unsupported operation name: {self.name}")

        if isinstance(flops, torch.Tensor):
            flops = flops.item()

        # Set infinity to zero
        if math.isinf(flops):
            flops = 0.0

        return flops

    def get_mac_in(self):

        # Getting the input tensor shape
        input_dims = torch.tensor(self.input.shape[-2:])

        # Computing MAC_in
        if self.name == "conv":
            # MAC_conv = (C_in / G) x W_in x H_in x K_h x K_w x (C_out / G) x B

            # Getting input and output channels (C_in, C_out)
            in_channels = self.module.in_channels
            out_channels = self.module.out_channels

            # Getting the kernel size (K_w x K_h)
            kernel_dims = torch.tensor(self.module.kernel_size)

            # Getting the number of groups
            g = self.module.groups

            # Computing Convolution MACs
            mac_in = (in_channels // g) * torch.prod(input_dims) \
                * torch.prod(kernel_dims) * (out_channels // g) * \
                self.bit_depth

        elif self.name in POOL_OPS:
            # MAC_pool = (C_in / G) x (W_in / S) x (H_in / S) x B

            # Getting the input channels (C_in)
            in_channels = self.input.shape[1]

            # Getting the number of groups
            if hasattr(self.module, 'groups'):
                g = self.module.num_groups
            else:
                g = 1

            # Getting the stride size
            if self.name == "adap_avg_pool":
                s = 1
            else:
                s = self.module.stride

            # Getting the Height and Width sizes of the input tensor
            _, _, height, width = self.input.shape

            # Computing Pooling MACs
            mac_in = (in_channels // g) * (torch.div(height, s)) \
                * (torch.div(width, s)) * self.bit_depth

        elif self.name == "bn":
            # MAC_bn = (2 x C_in x H_in x W_in x B) / G

            # Getting the input channels (C_in)
            in_channels = self.input.shape[1]

            # Estimate the number of groups
            g = self.module.weight.shape[0] // self.module.num_features
            g = torch.tensor(g)

            # Computing BatchNorm MACs
            mac_in = (2 * in_channels * torch.prod(input_dims) *
                      self.bit_depth)
            mac_in = torch.div(mac_in, g, rounding_mode='floor')

        elif self.name in ACT_FUNCS:
            # MAC_act = C x H_in x W_in x B / G

            # Getting the input channels (C_in)
            in_channels = self.input.shape[1]

            # Getting the number of groups
            if hasattr(self.module, 'groups'):
                g = self.module.num_groups
            else:
                g = 1

            # Computing Activation Function MACs
            mac_in = (in_channels * torch.prod(input_dims) * self.bit_depth)
            mac_in = torch.div(mac_in, g, rounding_mode='floor')

        elif self.name == "linear":
            # MAC_fc = (C_in / G) x (C_out / G) x B

            # Getting input and output channels
            in_channels = self.module.in_features
            out_channels = self.module.out_features

            # Getting the number of groups
            if hasattr(self.module, 'groups'):
                g = self.module.groups
            else:
                g = 1

            # Computing Fully Connected MACs
            mac_in = torch.div(in_channels, g) * torch.div(out_channels, g) \
                * self.bit_depth
        else:
            raise ValueError(f"Unsupported operation name: {self.name}")

        if isinstance(mac_in, torch.Tensor):
            mac_in = mac_in.item()

        return mac_in

    def get_mac_out(self):
        # MAC_out = C_out x H_out x W_out x B

        # We only need the output tensor shape
        if self.name == "conv":
            out_channels = self.module.out_channels

        elif self.name in POOL_OPS:
            out_channels = self.input.shape[1]

        elif self.name == "bn":
            out_channels = self.module.num_features

        elif self.name in ACT_FUNCS:
            out_channels = self.output.shape[1]

        elif self.name == "linear":
            out_channels = self.module.out_features

        else:
            raise ValueError(f"Unsupported operation name: {self.name}")

        # Getting the output dimensions
        output_dims = torch.tensor(self.output.shape[-2:])

        # Computing MAC Output
        mac_out = out_channels * torch.prod(output_dims) * self.bit_depth

        if isinstance(mac_out, torch.Tensor):
            mac_out = mac_out.item()

        return mac_out

    def get_mac_weights(self):

        # Getting the input tensor shape dimensions
        input_dims = torch.tensor(self.input.shape[-2:])

        # We only need both input and output tensor shapes
        if self.name == "conv":
            # MAC_conv = (K^2 x C_in / G) x (C_out / G) x (H_out x W_out) x B

            # Getting the input and output channels
            in_channels = self.module.in_channels
            out_channels = self.module.out_channels

            # Getting the kernel size
            kernel_dims = torch.tensor(self.module.kernel_size)

            # Getting the number of groups
            g = self.module.groups

            # Computing MACs
            mac_w = torch.div((torch.prod(kernel_dims) * in_channels), g) * \
                torch.div(out_channels, g) * torch.prod(input_dims) * \
                self.bit_depth

        elif self.name in POOL_OPS:
            mac_w = 0

        elif self.name == "bn":
            # MAC_bn = (4 x C / G) x B

            # Getting the input and output channels
            in_channels = self.input.shape[1]
            out_channels = self.module.num_features

            # Estimate the number of groups
            g = self.module.weight.shape[0] // out_channels
            g = torch.tensor(g)

            # Computing MACs
            mac_w = (torch.div(4 * in_channels, g, rounding_mode='floor')) \
                * self.bit_depth

        elif self.name in ACT_FUNCS:
            # MAC_act = (C / G) x W x H x B

            # Getting the input channels
            in_channels = self.input.shape[1]

            # Getting the number of groups
            if hasattr(self.module, 'groups'):
                g = self.module.num_groups
            else:
                g = 1

            # Computing MACs
            mac_w = (torch.div(in_channels, g, rounding_mode="floor")) * \
                torch.prod(input_dims) * self.bit_depth

        elif self.name == "linear":
            # MAC_fc = (C_in / G) x (C_out / G) x B

            # Getting the input and output channels
            in_channels = self.module.in_features
            out_channels = self.module.out_features

            # Getting the number of groups
            if hasattr(self.module, 'groups'):
                g = self.module.groups
            else:
                g = 1

            # Computing MACs
            mac_w = (torch.div(in_channels, g, rounding_mode="floor")) * \
                (torch.div(out_channels, g, rounding_mode="floor")) * \
                self.bit_depth

        if isinstance(mac_w, torch.Tensor):
            mac_w = mac_w.item()

        return mac_w

    def get_macs(self):

        if self.name in CUSTOM_OPS:
            return 0

        # In order to compute most precise MACs (Memory Access Cost) we have
        # to compute the memory required for the input, weights and output
        # memory

        # 1.-Retrieving the input vector or reading the feature map involves
        # accessing memory and transferring data to the computing unit, which
        # can be time-consuming (MAC_in).

        # 2.-Computing the dot product between the weights and the input
        # feature map, represented by the term MAC_w, includes the cost
        # of reading weights from memory and performing multiplication and
        # accumulation operations.

        # 3.-Writing the output vector or feature map back to memory,
        # represented by the term MAC_out, includes the cost of writing output
        # data to memory, which can also be time-consuming

        # Getting MACs (Input, Weights and Output)
        mac_in = self.get_mac_in()
        mac_w = self.get_mac_weights()
        mac_out = self.get_mac_out()

        # Computing the total estimated MAC Complexity
        return mac_in + mac_w + mac_out

    def get_short_name(self, name):

        if name == "conv2d":
            return "conv"

        elif name == "adaptiveavgpool2d":
            return "adap_avg_pool"

        elif name == "avgpool2d":
            return "avg_pool"

        elif name == "maxpool2d":
            return "max_pool"

        elif name == "adaptivemaxpool2d":
            return "adap_max_pool"

        elif name == "batchnorm2d":
            return "bn"

        else:
            return name

    def valid_params(self):

        if not hasattr(self, 'input_shape'):
            self.input_shape = ''

        if not hasattr(self, 'output_shape'):
            self.output_shape = ''

        if not hasattr(self, 'ks'):
            self.ks = ''

        if not hasattr(self, 'g'):
            self.g = ''

        if not hasattr(self, 's'):
            self.s = ''

        if not hasattr(self, 'dil'):
            self.dil = ''

        if not hasattr(self, 'residual'):
            self.residual = ''

        if not hasattr(self, 'lvl'):
            self.lvl = ''

    def get_non_custom_params(self):

        if self.name in CUSTOM_OPS:
            return

        # Getting Input Shape
        if self.name == "linear":
            self.input_shape = self.module.in_features
            self.output_shape = self.module.out_features
        else:
            self.input_shape = str(self.input.shape[-3]) + "x" + \
                str(self.input.shape[-2]) + "x" + str(self.input.shape[-1])

        if self.name in ACT_FUNCS or self.name == 'linear':

            if hasattr(self.module, 'groups'):
                self.g = self.module.num_groups
            else:
                self.g = 1

        if self.name == "bn" or self.name == "conv" or self.name in POOL_OPS:

            # Getting Output Shape
            self.output_shape = str(self.output.shape[-3]) + "x" + \
                str(self.output.shape[-2]) + "x" + str(self.output.shape[-1])

        if self.name == "conv":

            # Getting the kernel size (K_w x K_h)
            self.ks = list(self.module.kernel_size)
            self.ks = str(self.ks[-2]) + "x" + str(self.ks[-1])

        if self.name in POOL_OPS:

            if hasattr(self.module, 'groups'):
                self.g = self.module.num_groups
            else:
                self.g = 1

            if self.name == "adap_avg_pool":
                self.s = 1
            else:
                self.ks = list(self.module.kernel_size)
                self.ks = str(self.ks[-2]) + "x" + str(self.ks[-1])
                self.s = self.module.stride

        if self.name == "conv":

            # Getting the stride
            self.s = list(self.module.stride)
            self.s = str(self.s[-2]) + "x" + str(self.s[-1])

            # Getting the number of groups
            self.g = str(self.module.groups)

            # Getting dilation rate
            self.dil = list(self.module.dilation)
            self.dil = str(self.dil[-2]) + "x" + str(self.dil[-1])

    def get_custom_params(self):
        # print(f'[DEBUG] Operation Name: {self.name}')

        # Getting input and output shape
        self.input_shape = str(self.input.shape[-3]) + "x" + \
            str(self.input.shape[-2]) + "x" + str(self.input.shape[-1])

        # Getting Output Shape
        self.output_shape = str(self.output.shape[-3]) + "x" + \
            str(self.output.shape[-2]) + "x" + str(self.output.shape[-1])

        # CBAM or SAM modules only required input and output shapes
        if self.name == "sam" or self.name == "cbam":
            return

        # Jet Conv, Efficient Channel Attention or Convolution BatchNorm Act
        if self.name == "jetconv" or self.name == "eca" or self.name == "cba":

            # Getting the kernel size (K_w x K_h)
            self.ks = str(self.module.ks)

        # Channel Shuffle, Convolution BatchNorm Act or Jet Conv
        if self.name == "cs" or self.name == "cba" or self.name == "jetconv":

            # Getting the number of groups
            self.g = str(self.module.g)

        if self.name == "cba" or self.name == "jetconv":

            # Getting the stride
            self.s = str(self.module.s)

            # Getting the dilation rate
            self.dil = str(self.module.dil)

        if self.name == "jetconv":

            # Getting residual block, lvl and lvls
            self.residual = str(self.module.residual)
            self.lvl = str(self.module.lvl)

    def get_module_params(self):

        if not self.custom_ops:
            self.get_non_custom_params()
        else:
            self.get_custom_params()

    def get_module_info(self):

        # Getting module parameters info
        self.get_module_params()

        # Valid module parameters info
        self.valid_params()

        if not self.custom_ops:

            if self.name not in CUSTOM_OPS:
                return {"op_name": self.name, "input": self.input_shape,
                        "output": self.output_shape, "ks": self.ks,
                        "dil": self.dil, "g": self.g, "s": self.s}
            else:
                return {}
        else:

            if self.name in CUSTOM_OPS or self.name == "bn" \
                    or self.name == "reu" or self.name == "conv":
                return {"op_name": self.name, "input": self.input_shape,
                        "output": self.output_shape, "ks": self.ks,
                        "dil": self.dil, "g": self.g, "s": self.s,
                        "residual": self.residual, "lvl": self.lvl}
            else:
                return {}


def profiler_hook(module, input, output):

    # Create module profiler
    module_profiler = ModuleProfiler(module, input[0],
                                     output, custom_ops=False)
    # Getting layer info
    layer_info = module_profiler.get_module_info()

    # Getting block info
    module_profiler.custom_ops = True
    block_info = module_profiler.get_module_info()

    # Getting the Memory-Footprint (total memory) required for the module
    tot_mem = module_profiler.get_mem()

    # Getting the FLOPs (Floating Point Operations) used for the module
    flops = module_profiler.get_flops()

    # Getting the MACs (Memory Access Cost) used for the module
    macs = module_profiler.get_macs()

    # Add to module
    if len(layer_info) != 0:
        module.__layers__ = layer_info

    # Add to module
    if len(block_info) != 0:
        module.__blocks__ = block_info

    module.__flops__ += flops
    module.__mac__ += macs
    module.__tot_mem__ += tot_mem


HOOKS = {
    # Convolutions
    nn.Conv2d: profiler_hook,
    # Customs
    JetConv: profiler_hook,
    CBA: profiler_hook,
    CS: profiler_hook,
    ECA: profiler_hook,
    SAM: profiler_hook,
    CBAM: profiler_hook,
    # Activation Function
    REU: profiler_hook,
    TanhExp: profiler_hook,
    # Pools
    nn.AvgPool2d: profiler_hook,
    nn.MaxPool2d: profiler_hook,
    nn.AdaptiveAvgPool2d: profiler_hook,
    # Batch Normalization
    nn.BatchNorm2d: profiler_hook,
    # Fully Connected
    nn.Linear: profiler_hook
}
