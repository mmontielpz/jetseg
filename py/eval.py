'''
Script to Eval model for Semantic Segmentation
'''

import pathmagic

import os
import gc
import sys
import logging
import argparse

import torch.cuda as cuda

from modules.model import JetSeg
from modules.utils import _init_logger, get_path, ModelConfigurator, pkl

from modules.mltools import (
    load_dataset, load_dataloaders, load_ckp, evaluation, map_results
)

from modules.vis import plot_config, plot_clear, show_segmentation


# Ignore noqa pep8
assert pathmagic


def eval_inference(eval_cfg, logger):

    # Getting eval configuration
    dataset_name = eval_cfg["dataset_name"]
    jetseg_comb = eval_cfg["jetseg_comb"]
    jetseg_mode = eval_cfg["jetseg_mode"]

    # Read model configuration
    train_path = get_path("train")

    # Create model configuration
    cfg = ModelConfigurator(comb=jetseg_comb,
                            dataset_name=dataset_name, mode=jetseg_mode)

    # Get the model name
    model_name = cfg.model_name
    model_ckp = train_path + model_name + ".pth"

    # Check if model exists (He was already trained)
    if not os.path.isfile(model_ckp):
        raise FileNotFoundError(
            f"[ERROR] The model name '{model_name}' does not exist.")

    # Build model
    model = JetSeg(cfg)
    model = model.to("cuda")
    logger.info('Model built')

    # Loading model weights
    ckp_cfg = {"model": model, "model_ckp": model_ckp}
    model = load_ckp(ckp_cfg, mode=False)

    # Loading dataset
    datasets = load_dataset(dataset_name, 32)

    # Getting test dataset
    test_dataset = datasets["test"]

    # Getting the dataset color code
    color_code = test_dataset.id2code

    # Loading dataloaders
    dataloaders = load_dataloaders(datasets, batch_size=1)
    logger.info('Dataset read')

    # Getting test dataloader
    dataloader = dataloaders["test"]

    # Semantic Segmentation Evaluation
    eval_out = evaluation(dataloader, model, color_code)

    # Mapping the results (get the 4 best, worst and random semantic
    # segmentation)
    results = map_results(eval_out)

    # Saved Semantic Segmentation Inferences as png files
    results_path = get_path("results")

    # Set good plot configuration
    plot_cfg = {
        "font.family": "sans-serif",
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "lines.markersize": 2,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 18,
        "text.usetex": False,
        # "figure.figsize": [8, 8],
        "figure.figsize": [24, 24],
        "figure.dpi": 300,
    }

    plot_config(plot_cfg)

    # Best Semantic Segmentation Results
    png_name = results_path + model_name + \
        "-best-semantic-segmentation.png"
    show_segmentation(results["best"], png_name)
    plot_clear()

    # Worst Semantic Segmentation Results
    png_name = results_path + model_name + \
        "-worst-semantic-segmentation.png"
    show_segmentation(results["worst"], png_name)
    plot_clear()

    # Random Semantic Segmentation Results
    png_name = results_path + model_name + \
        "-random-semantic-segmentation.png"
    show_segmentation(results["random"], png_name)
    plot_clear()


def main(args):

    # Cleaning screen
    os.system("clear")

    # Create logger
    _init_logger('eval')
    _logger = logging.getLogger('eval')
    _logger.info('Evaluation started')

    # Cleaning memory
    gc.collect()
    cuda.empty_cache()

    # Setting eval configuration
    eval_cfg = {"dataset_name": args.dataset_name,
                "jetseg_comb": args.jsc,
                "jetseg_mode": args.jsm,
                }

    # Evaluating
    eval_inference(eval_cfg, _logger)
    _logger.info('Evaluation finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='camvid',
                        help='Specify dataset name')
    parser.add_argument('--jsc', type=int, default=0,
                        help='Set JetSeg Combination [0-5]')
    parser.add_argument('--jsm', type=int, default=3,
                        help='Specify the JetSeg Mode')

    args = parser.parse_args()

    main(args)
