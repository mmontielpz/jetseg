import pathmagic

import sys
import torch
import numpy as np
import torch.cuda as cuda

from tqdm import tqdm
from decimal import Decimal, getcontext

from config import hks

assert pathmagic

# Precision of 4
getcontext().prec = 4


def compute_avg(list_values):

    # Convert list to np array
    np_array = np.array(list_values)

    # Compute avg
    avg_val = float(round(np.mean(np_array), 2))

    return avg_val


def compute_fps(inference_time):

    # Convert inference times (ms to sec)
    inference_time = inference_time / 1000

    # Compute the frames per sec (FPS)
    fps = 1 / inference_time

    fps = round(fps, 1)

    return fps


class Performance:

    def __init__(self):
        self.hooks = hks
        self.handles = []
        self.tot_flops = 0
        self.tot_mac = 0
        self.tot_mem = 0
        self.layers = []
        self.blocks = []

    def reset_performance(self):
        self.tot_flops = 0
        self.tot_mac = 0
        self.tot_mem = 0
        self.layers = []
        self.blocks = []

    def add_module_variables(self, module):

        if not hasattr(module, '__blocks__'):
            module.__blocks__ = None

        if not hasattr(module, '__layers__'):
            module.__layers__ = None

        if not hasattr(module, '__mac__'):
            module.__mac__ = 0

        if not hasattr(module, '__flops__'):
            module.__flops__ = 0

        if not hasattr(module, '__tot_mem__'):
            module.__tot_mem__ = 0

    def register_hooks(self, module):

        if type(module) in self.hooks:

            self.handles += [
                module.register_forward_hook(self.hooks[type(module)])
            ]

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def get_flops(self, module):
        self.tot_flops += module.__flops__

    def get_mac(self, module):
        self.tot_mac += module.__mac__

    def get_mem(self, module):
        self.tot_mem += module.__tot_mem__

    def get_module_info(self, module):
        self.layers.append(module.__layers__)
        self.blocks.append(module.__blocks__)

    def get_tot_flops(self):
        return float(Decimal(self.tot_flops) / Decimal(1e9))

    def get_tot_mac(self):
        return float(Decimal(self.tot_mac) / Decimal(1e9))

    def get_tot_mem(self):
        return float(Decimal(self.tot_mem) / Decimal(1e6))

    def run_warmups(self, num_warmups, model, data):
        with torch.no_grad():
            for _ in tqdm(range(num_warmups),
                          desc="[Running] Warmups Dry-runs"):
                _ = model.forward(data)
        cuda.synchronize()

    def get_model_fwd_bwd_time(self, model, x, y):

        fwd_times = []
        bwd_times = []

        for _ in tqdm(range(100), desc="[Running] Measuring fwd/bwd time"):

            # Enable gradients
            for param in model.parameters():
                param.requires_grad = True

            # Synchronize GPU time to measure Forward and Backward pass time
            # correctly
            cuda.synchronize()

            # Create CUDA timers
            start_time = cuda.Event(enable_timing=True)
            end_time = cuda.Event(enable_timing=True)

            # Forward pass
            start_time.record()
            pred = model(x)
            end_time.record()
            cuda.synchronize()

            # Estimate forward pass time
            fwd_time = round(start_time.elapsed_time(end_time), 4)
            fwd_time = fwd_time / x.shape[0]
            fwd_times.append(fwd_time)

            # Backward pass
            model.zero_grad()
            start_time.record()
            pred.backward(y)
            end_time.record()
            cuda.synchronize()

            # Estimate backward pass time
            bwd_time = round(start_time.elapsed_time(end_time), 4)
            bwd_time = bwd_time / x.shape[0]
            bwd_times.append(bwd_time)

            # Disable gradients
            for param in model.parameters():
                param.requires_grad = False

        # Calculate the average
        fwd_time = compute_avg(fwd_times)
        bwd_time = compute_avg(bwd_times)

        return fwd_time, bwd_time

    def get_model_infer_time(self, model, x):

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        infer_times = []
        for _ in tqdm(range(100), desc="[Running] Measuring inference time"):

            # Synchronize GPU time ad measure Forward pass
            cuda.synchronize()

            # Create CUDA timers
            start_time = cuda.Event(enable_timing=True)
            end_time = cuda.Event(enable_timing=True)

            # Inference time
            start_time.record()
            _ = model(x)
            end_time.record()
            cuda.synchronize()

            # Estimate inference time
            infer_time = start_time.elapsed_time(end_time)
            infer_times.append(infer_time)

        # Calculate the average
        infer_time = compute_avg(infer_times)

        return infer_time

    def model_benchmark(self, model, input_shape=(32, 3, 224, 224),
                        nwarmup=50):

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        # Get model params
        params = sum(p.numel() for p in model.parameters())
        params = float(Decimal(params) / Decimal(1e6))

        # Create dummy input
        input_data = torch.randn(input_shape)
        input_data = input_data.to("cuda")

        # Create module variables
        model.apply(self.add_module_variables)

        # Register hooks
        model.apply(self.register_hooks)

        # Run model
        output = model(input_data)

        # Remove model hooks
        self.remove_hooks()

        # Get flops
        model.apply(self.get_flops)

        # Get mac
        model.apply(self.get_mac)

        # Get mem
        model.apply(self.get_mem)

        # Get layers info
        model.apply(self.get_module_info)

        # Remove None Values
        blocks_info = [block for block in self.blocks if block is not None]
        layers_info = [layer for layer in self.layers if layer is not None]

        # Warming up
        print("[INFO] Warmup begins...")
        self.run_warmups(nwarmup, model,
                         input_data.expand(input_shape[0], -1, -1, -1))
        print("[INFO] Warmup ends...")

        # Measure Inference Time
        print("[INFO] Measuring inference time begins...")
        infer_time = self.get_model_infer_time(model, input_data)
        print("[INFO] Measuring inference time ends...")

        # Get total flops
        flops = self.get_tot_flops()

        # Get total MACs
        mac = self.get_tot_mac()

        # Get total memory footprint
        memf = self.get_tot_mem()

        # Convert to float and rounding
        params = round(float(params), 4)
        flops = round(float(flops), 4)
        mac = round(float(mac), 4)
        memf = round(float(memf), 2)

        # Get fps
        fps = compute_fps(infer_time)

        infer_time = round(float(infer_time), 2)

        # Get model performance metrics (FLOPs, MACs, memory footprint)
        print("[INFO] Model Input Shape:", input_data.size())
        print("[INFO] Model Output Shape:", output.size())
        print(f'[RESULTS] Model Params (M): {params}')
        print(f'[RESULTS] Model FLOPs (G): {flops}')
        print(f'[RESULTS] Model MAC (G): {mac}')
        print(f'[RESULTS] Model Memory Footprint (MB): {memf}')
        print(f'[RESULTS] Model Average Inference Time: {infer_time}')
        print(f'[RESULTS] Model FPS: {fps}')

        info = {
            "layers": layers_info,
            "blocks": blocks_info,
        }

        return fps, infer_time, flops, mac, memf, params, info
