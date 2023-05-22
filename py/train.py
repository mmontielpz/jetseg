'''
Script to train model for Semantic Segmentation
'''

import pathmagic

import os
import gc
import sys
import logging
import argparse

import torch
import torch.cuda as cuda

from modules.model import JetSeg
from modules.utils import _init_logger, get_path, pkl, ModelConfigurator

from modules.mltools import (
    load_dataset, load_dataloaders, build_loss_fn,
    build_optimizer, training
)


# Ignore noqa pep8
assert pathmagic


def train(train_cfg, logger):

    # Getting train configuration
    dataset_name = train_cfg["dataset_name"]
    jetseg_comb = train_cfg["jsc"]
    jetseg_mode = train_cfg["jsm"]
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]

    # Create model configuration
    cfg = ModelConfigurator(comb=jetseg_comb, mode=jetseg_mode,
                            dataset_name=dataset_name)

    # Build model
    model = JetSeg(cfg)
    model = model.to("cuda")
    logger.info('Model built')

    # Loading dataset
    dataset = load_dataset(dataset_name, cfg.num_classes)

    # Get mean and std of dataset
    # mean, std = dataset["train"].get_dataset_mean_std()
    # print(f'[DEBUG] Mean: {mean}')
    # print(f'[DEBUG] STD: {std}')

    # Loading dataloaders
    dataloaders = load_dataloaders(dataset, batch_size)
    logger.info('Dataset read')

    # Getting train path
    train_path = get_path("train")

    # Building loss fn and optimizer
    loss_fn = build_loss_fn(loss_name="jet",
                            pixels_per_class=cfg.pixels_per_class)

    optimizer = build_optimizer(optimizer_name="adamw", model=model)
    logger.info('Loss and optimizer built')

    # Setting train parameters
    params = {}

    # Create model name
    model_name = "JetSeg-M" + str(jetseg_mode) + "-C" + \
        str(jetseg_comb) + "-" + dataset_name.lower()

    # Getting the dataset color codes
    color_code = {"id2code": dataset["train"].id2code,
                  "code2id": dataset["train"].code2id}

    params["debug"] = True
    params["model_name"] = model_name
    params["num_epochs"] = num_epochs
    params["early_stop"] = False
    params["dataloaders"] = dataloaders
    params["model"] = model.train()
    params["model_path"] = train_path
    params["device"] = "cuda"
    params["loss_fn"] = loss_fn
    params["optimizer"] = optimizer
    params["scheduler"] = None
    params["num_classes"] = cfg.num_classes
    params["color_code"] = color_code

    logger.info('Training set up')

    # Model training
    dict_losses = training(params)

    out = {}
    out["dataset_name"] = dataset_name
    out["model_name"] = model_name
    out["jetseg_comb"] = jetseg_comb
    out["jetseg_mode"] = jetseg_mode
    out["losses"] = dict_losses
    out["model"] = model
    out["loss_fn"] = loss_fn
    out["optimizer"] = optimizer

    return out


def save_train_res(train_out, logger):

    # Getting model and losses
    dataset_name = train_out["dataset_name"]
    model_name = train_out["model_name"]
    jetseg_comb = train_out["jetseg_comb"]
    jetseg_mode = train_out["jetseg_mode"]
    loss_fn = train_out["loss_fn"]
    optimizer = train_out["optimizer"]
    model = train_out["model"]
    dict_losses = train_out["losses"]
    train_path = get_path('train')

    # Save model weights
    ckp_model = train_path + model_name + ".pth"

    # Create the model checkpoint
    checkpoint = {
        'arch': "JetSeg",
        'epoch': len(dict_losses["train_losses"]),
        'valid_loss': dict_losses["val_losses"][-1],
        'model_state_dict': model.state_dict(),
        'loss_fn': loss_fn,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Save model configuration
    model_cfg = {"dataset_name": dataset_name.lower(),
                 "jsm": jetseg_mode,
                 "jsc": jetseg_comb}

    # Save the model configuration
    pkl_file = train_path + model_name + "-cfg.pkl"
    pkl(pkl_file, model_cfg, mode=True)

    # Saving checkpoint
    torch.save(checkpoint, ckp_model)
    logger.info('Model saved')

    # Saving losses
    pkl_file = train_path + model_name + "-train-losses.pkl"
    pkl(pkl_file, dict_losses, mode=True)
    logger.info('Losses saved')


def main(args):

    # Cleaning screen
    os.system("clear")

    # Create logger
    _init_logger('train')
    _logger = logging.getLogger('train')
    _logger.info('Train started')

    # Cleaning memory
    gc.collect()
    cuda.empty_cache()

    # Setting train configuration
    train_cfg = {"dataset_name": args.dataset_name,
                 "jsc": args.jsc,
                 "jsm": args.jsm,
                 "num_epochs": args.num_epochs,
                 "batch_size": args.batch_size,
                 }
    # Training
    train_out = train(train_cfg, _logger)
    _logger.info('Train finished')

    # Saving results
    save_train_res(train_out, _logger)
    _logger.info('Results saved')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='camvid',
                        help='Specify dataset name')
    parser.add_argument('--jsc', type=int, default=0,
                        help='Set JetSeg Combination [0-5]')
    parser.add_argument('--jsm', type=int, default=3,
                        help='Specify the JetSeg Mode')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Set the number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Set the batch size')

    args = parser.parse_args()

    main(args)
