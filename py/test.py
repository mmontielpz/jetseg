'''
Script to Test model for Semantic Segmentation
'''

import pathmagic

import os
import gc
import sys
import logging
import argparse

import pandas as pd
import torch.cuda as cuda

from modules.model import JetSeg
from modules.utils import _init_logger, get_path, ModelConfigurator

from modules.mltools import (
    load_dataset, load_dataloaders, build_loss_fn, build_optimizer,
    load_ckp, testing
)


# Ignore noqa pep8
assert pathmagic


def test(test_cfg, logger):

    # Getting test configuration
    dataset_name = test_cfg["dataset_name"]
    jetseg_comb = test_cfg["jsc"]
    jetseg_mode = test_cfg["jsm"]
    batch_size = test_cfg["batch_size"]

    # Read model configuration
    train_path = get_path("train")

    # Create model configuration
    cfg = ModelConfigurator(comb=jetseg_comb, mode=jetseg_mode,
                            dataset_name=dataset_name)

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
    dataset = load_dataset(dataset_name, cfg.num_classes)

    # Loading dataloaders
    dataloaders = load_dataloaders(dataset, batch_size)
    logger.info('Dataset read')

    # Building loss fn and optimizer
    loss_fn = build_loss_fn(loss_name="jet",
                            pixels_per_class=cfg.pixels_per_class)

    optimizer = build_optimizer(optimizer_name="adamw", model=model)
    logger.info('Loss and optimizer built')

    # Setting test parameters
    params = {}

    params["debug"] = True
    params["dataset_name"] = dataset_name.lower()
    params["num_epochs"] = 1
    params["early_stop"] = False
    params["dataloaders"] = dataloaders
    params["model"] = model.train()
    params["model_path"] = train_path
    params["device"] = "cuda"
    params["loss_fn"] = loss_fn
    params["optimizer"] = optimizer
    params["scheduler"] = None
    params["num_classes"] = cfg.num_classes

    logger.info('Testing set up')

    # Model validation
    val_loss, val_miou = testing(params, val=True)

    # Model testing
    test_loss, test_miou = testing(params)

    # Save results as dictionary
    out = {"dataset_name": dataset_name,
           "model_name": model_name,
           "jetseg_comb": jetseg_comb,
           "val_loss": val_loss, "test_loss": test_loss,
           "val_miou": val_miou, " test_miou": test_miou}

    return out


def save_test_res(test_out, logger):

    # Getting path to save the test results
    results_path = get_path('results')

    # Create data frame in order to save in a csv file
    df_test = pd.DataFrame([test_out])

    # Save results as csv in results path
    csv_test = results_path + test_out["model_name"] + "-test.csv"
    df_test.to_csv(csv_test)


def main(args):

    # Cleaning screen
    os.system("clear")

    # Create logger
    _init_logger('test')
    _logger = logging.getLogger('test')
    _logger.info('Test started')

    # Cleaning memory
    gc.collect()
    cuda.empty_cache()

    # Setting test configuration
    test_cfg = {"dataset_name": args.dataset_name,
                "jsc": args.jsc,
                "jsm": args.jsm,
                "batch_size": args.batch_size}
    # Testing
    test_out = test(test_cfg, _logger)
    _logger.info('Test finished')

    # Saving results
    save_test_res(test_out, _logger)
    _logger.info('Results saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='camvid',
                        help='Specify dataset name')
    parser.add_argument('--jsc', type=int, default=0,
                        help='Set JetSeg Combination [0-5]')
    parser.add_argument('--jsm', type=int, default=3,
                        help='Specify the JetSeg Mode')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Set the batch size')

    args = parser.parse_args()

    main(args)
