'''
Script to evaluate the model performance for Real-Time Semantic Segmentation
'''

import pathmagic

import os
import gc
import sys
import logging
import pathlib
import warnings
import argparse

import pandas as pd

import torch.cuda as cuda
import torch.backends.cudnn as cudnn

from modules.model import JetSeg
from modules.performance import Performance
from modules.utils import _init_logger, ModelConfigurator, pkl


# Ignore noqa pep8
assert pathmagic

# Ignore warnings
warnings.filterwarnings('ignore')

# Set benchmark mode ON
cudnn.benchmark = True


def benchmark(model, img_size):

    # Create Model Performance Evaluator
    perf_eval = Performance()

    # Benchmarking model for batch size of N
    fps, infer_time, flops, mac, memf, params, info = \
        perf_eval.model_benchmark(model,
                                  input_shape=(1, 3, img_size[0], img_size[1]))

    # Clean all memory
    gc.collect()
    cuda.empty_cache()
    del model

    # Saving benchmark results
    res = {}

    res["img_size"] = img_size
    res["params"] = params
    res["fps"] = fps
    res["infer_time"] = infer_time
    res["flops"] = flops
    res["mac"] = mac
    res["memf"] = memf
    res["info"] = info

    # Reset performance
    perf_eval.reset_performance()
    del perf_eval

    return res


def run_perf_berk(berk_cfg):

    # Create logger
    _init_logger("model_perf_benchmark")
    _logger = logging.getLogger("model_perf_benchmark")
    _logger.info("Model Performance Benchmark start")

    # Getting benchmark configuration
    debug = berk_cfg["debug"]
    jetseg_comb = berk_cfg["jsc"]
    jetseg_mode = berk_cfg["jsm"]
    platform_name = berk_cfg["platform_name"]
    img_size = berk_cfg["img_size"]

    # Get workspace directory
    workspace_path = pathlib.Path().absolute()
    workspace_path = workspace_path.parent
    workspace_path = str(workspace_path)

    # Get the model name
    model_name = "JetSeg-M" + str(jetseg_mode) + "-C" \
        + str(jetseg_comb) + "-camvid-"

    # Create model configuration
    cfg = ModelConfigurator(comb=jetseg_comb, img_sizes=args.img_size,
                            mode=jetseg_mode, debug=debug)

    # Build model and send to device
    model = JetSeg(cfg)
    model.eval()
    model = model.to("cuda")

    _logger.info("Model Loaded")

    # Benchmarking with batch size of N
    berk_res = benchmark(model, img_size)
    berk_res["platform_name"] = platform_name
    berk_res["jetseg_comb"] = jetseg_comb

    # Extract layers and blocks info
    info = berk_res.pop("info")
    layers_info = info.pop("layers")
    blocks_info = info.pop("blocks")

    # Create data frames
    df_perf = pd.DataFrame(berk_res)
    df_jsc = pd.DataFrame([cfg.jetseg_comb])
    df_layers = pd.DataFrame(layers_info)
    df_blocks = pd.DataFrame(blocks_info)

    # Convert data type
    img_size = str(img_size[0]) + "x" + str(img_size[1])

    # Save benchmark results
    results_path = workspace_path + '/' + "results" + '/'
    csv_metrics = results_path + \
        platform_name + "-" + img_size + "-" + model_name + "perf.csv"
    df_perf.to_csv(csv_metrics)
    csv_metrics = results_path + \
        platform_name + "-" + img_size + "-" + model_name + "jsc.csv"
    df_jsc.to_csv(csv_metrics)
    csv_metrics = results_path + \
        platform_name + "-" + img_size + "-" + model_name + "layers.csv"
    df_layers.to_csv(csv_metrics)
    csv_metrics = results_path + \
        platform_name + "-" + img_size + "-" + model_name + "blocks.csv"
    df_blocks.to_csv(csv_metrics)
    _logger.info('Saved model performance results')


def main(args):

    # Cleaning screen
    os.system("clear")

    # Cleaning memory
    gc.collect()
    cuda.empty_cache()

    # Create benchmark configuration
    berk_cfg = {
        "debug": args.debug,
        "jsc": args.jsc,
        "jsm": args.jsm,
        "platform_name": args.platform_name,
        "img_size": args.img_size}

    # Let's do it !!!
    run_perf_berk(berk_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        default=False, help='Debug model')

    parser.add_argument('--jsc', type=int, default=0,
                        help='Set JetSeg Combination [0-5]')

    parser.add_argument('--jsm', type=int, default=3,
                        help='Specify the JetSeg Mode')

    parser.add_argument('--platform_name', type=str, default='xavier',
                        help='Specify the platform name')

    parser.add_argument('--num_classes', type=int, default=32,
                        help='Set the number of output classes')

    parser.add_argument('--img_size', nargs='+', type=int, default=[512, 256],
                        help='Set img size [width, height]')

    args = parser.parse_args()

    main(args)
