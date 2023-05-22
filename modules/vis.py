import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_config(plot_cfg=None, latex=False):
    """
    Function to set plotting configuration
    """

    sns.set_theme()
    sns.despine()
    sns.set_context("paper")

    plt.tight_layout()

    if not plot_cfg:
        # Set default configuration

        plot_cfg = {
            # 'backend': 'wxAgg',
            'font.family': "sans-serif",
            'font.weight': "normal",
            'axes.labelweight': "normal",
            'lines.markersize': 2,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 10,
            'legend.title_fontsize': 10,
            'text.usetex': latex,
            'figure.figsize': [20, 20],
            'figure.dpi': 96
        }

    # Update rcPrams
    plt.rcParams.update(plot_cfg)


def plot_clear():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def show_sample(img, semantic=False):

    if not isinstance(img, np.ndarray):
        img = img.numpy()

    if not semantic:
        img = img.transpose((1, 2, 0))

        # Rescale pixel values to [0, 1]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Apply inverse normalization
        img = torch.from_numpy(img)

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img)


def show_batch(n_samples, samples):

    # Get nrows and ncols
    n_elements = int(math.sqrt(n_samples))

    # Check if n_elements are odd or even
    if not (n_elements % 2) == 0:
        raise ValueError(f'{n_samples} is not odd number')

    if not samples:
        raise ValueError(f'{samples} is empty')

    fig, axes = plt.subplots(nrows=n_elements, ncols=n_elements)

    for ax, sample in zip(axes.flat, samples):

        # Get img and label of sample
        img, mask = sample

        # Configure axes
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Apply transpose to visualize
        img = img.transpose((1, 2, 0))
        mask = mask.transpose((1, 2, 0))

        # Display
        ax.imshow(img)
        ax.imshow(mask)

    # Show Batch
    plt.show()


def show_semantic(samples, title, cols=4,
                  rows=1, plot_size=(16, 16), semantic=False):

    # Create figure and sub figures
    fig = plt.figure(figsize=plot_size)

    # For each sample display
    for i in range(1, cols * rows + 1):

        # Adding figure to subplot
        fig.add_subplot(rows, cols, i)

        # Setting title and axis off
        plt.axis('off')
        plt.title(title + " " + str(i))

        # Get sample
        sample = samples[i-1]
        sample = sample.squeeze()

        if not semantic:

            # Apply transpose to visualize
            sample = sample.transpose((1, 2, 0))

            # Rescale pixel values to [0, 1]
            sample = (sample - np.min(sample)) / \
                (np.max(sample) - np.min(sample))

            sample = torch.from_numpy(sample)

        # Display img
        plt.imshow(sample)

    plt.show()


def plot_losses(data, curve_type, fig_name, debug=True):

    fig, axes = plt.subplots()

    if curve_type.lower() == 'loss':
        data_key = 'losses'
        y_label = 'loss'
        title_ = 'Losses'

    elif curve_type.lower() == 'mious':
        data_key = 'mious'
        y_label = 'mIoU'
        title_ = 'mIoUs'

    train_curve = data["train_" + data_key]
    val_curve = data["val_" + data_key]

    train_last = train_curve[-1]
    train_last = float(train_last)
    train_last = round(train_last, 4)

    val_last = val_curve[-1]
    val_last = float(val_last)
    val_last = round(val_last, 4)

    epochs = [epoch for epoch in range(1, len(train_curve) + 1)]

    plt.plot(epochs, train_curve,
             label='Train ' + y_label + ' = ' + str(train_last), marker='o')
    plt.plot(epochs, val_curve,
             label='Val ' + y_label + ' = ' + str(val_last), marker='o')

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Val ' + title_)
    plt.legend(loc="upper right")

    if fig_name:
        plt.savefig(fig_name, format='png', dpi=300)

    if debug:
        plt.show()

    plt.close(fig)

import sys
def show_segmentation(samples, fig_name=None, debug=False):

    num_samples = len(samples)
    fig, ax = plt.subplots(nrows=num_samples, ncols=3,
                           figsize=(10, 5*num_samples),
                           gridspec_kw={'hspace': 0.0025, 'wspace': 0.0025})

    for i in range(num_samples):

        # Getting Original Image, Ground Truth Mask, Predicted Semantic
        # Segmentation and its mIoU
        img = samples[i][0]
        mask = samples[i][1]
        miou = samples[i][2]
        pred = samples[i][3]

        # print()
        # print(f'[DEBUG] Img (dtype): {img.dtype}')
        # print(f'[DEBUG] Img (values): {img}')
        # print()
        # print(f'[DEBUG] Mask (dtype): {mask.dtype}')
        # print(f'[DEBUG] Mask (values): {mask}')
        # print()
        # print(f'[DEBUG] Prediction (dtype): {pred.dtype}')
        # print(f'[DEBUG] Prediction (values): {pred}')
        # print()

        # Rescale pixel values to [0, 1]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        # print()
        # print(f'[DEBUG] Img rescale (dtype): {img.dtype}')
        # print(f'[DEBUG] Img rescale (values): {img}')
        # print()
        # print(f'[DEBUG] Mask rescale (dtype): {mask.dtype}')
        # print(f'[DEBUG] Mask rescale (values): {mask}')
        # print()
        # print(f'[DEBUG] Prediction rescale (dtype): {pred.dtype}')
        # print(f'[DEBUG] Prediction rescale (values): {pred}')
        # print()

        # Display original image
        ax[i][0].imshow(img)
        ax[i][0].axis('off')
        ax[i][0].set_title("Original Image")

        # Display ground truth mask
        ax[i][1].imshow(mask, alpha=0.5)
        ax[i][1].axis('off')
        ax[i][1].set_title("Original Mask")

        # Display predicted mask
        ax[i][2].imshow(pred)
        ax[i][2].axis('off')
        ax[i][2].set_title("Predicted Mask " + "mIoU =" + str(miou))

        print('[INFO] Displayed Semantic Segmentation Sample')

    # Adjust the spacing between subplots and figure edges
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1,
                        right=0.9, hspace=0.0025, wspace=0.0025)
    # plt.subplots_adjust(wspace=0.025, hspace=0.1)

    # Saved figure
    if fig_name:
        plt.savefig(fig_name, format='png', dpi=300)

    if debug:
        plt.show()

    plt.close(fig)
