import pathmagic

import os
import sys
import math
import random
import shutil
import pickle
import pathlib
import logging
import torch

import pandas as pd
import numpy as np

from config import jsc


assert pathmagic

DATASETS = ["camvid", "cityscapes"]


class ModelConfigurator:
    def __init__(self, comb=0, dataset_name=None, img_sizes=None,
                 mode=3, debug=False):
        super(ModelConfigurator, self).__init__()

        self.debug = debug
        self.mode = mode
        self.combinations = jsc

        self.is_valid_model_config(dataset_name, img_sizes, mode)
        self.is_valid_jet_combination(comb)

        self.jetseg_comb = self.get_jet_combination(comb)
        self.get_base_cfg()

        # Create Model Name
        if dataset_name:
            self.model_name = "JetSeg-M" + str(mode) + "-C" + \
                str(comb) + "-" + dataset_name.lower()
        else:
            self.model_name = "JetSeg-M" + str(mode) + "-C" + \
                str(comb) + "-camvid"

        print()
        print(f'[INFO] {self.model_name} Configuration')
        print(self.jetseg_comb)
        print()

        self.dataset_name = dataset_name
        self.img_sizes = img_sizes
        self.pixels_per_class = self.count_pixels_per_class()
        self.get_data_config(dataset_name)

        # Get fmaps and its configuration
        self.fmaps = self.get_fmaps()
        self.fmaps_stages = self.get_fmaps_stages()
        self.fmaps_stages = self.set_residuals()

        self.cbam_fmaps = self.get_attention_fmaps("cbam")
        self.sam_fmaps = self.get_attention_fmaps("sam")
        self.ecam_fmaps = self.get_attention_fmaps("ecam")

        self.build_model_config()

    def is_valid_jet_combination(self, combination):
        if combination >= len(self.combinations):
            raise ValueError("[ERROR] The combination value is greater than \
                             the available combinations [0-5]")

    def is_valid_jet_mode(self, mode):
        if mode > self.mode or mode < self.mode:
            raise ValueError("[ERROR] The mode value is greater/smaller than \
                             the available modes [1-3]")

    def is_valid_model_config(self, dataset_name, img_sizes, mode):

        if img_sizes is None and dataset_name is None:
            raise ValueError("[ERROR] The img_sizes and dataset_name \
                             can't be both None")
        if mode is None:
            raise ValueError("[ERROR] The mode can't be None")

    def get_jet_combination(self, combination):
        return self.combinations[combination]

    def get_data_config(self, dataset_name):

        if self.img_sizes is None:

            self.dataset_name = dataset_name.lower()
            self.is_valid_dataset_name()
            self.img_sizes = self.get_dataset_sizes()
            self.num_classes = self.get_dataset_classes()

        else:
            self.dataset_name = None
            self.num_classes = 32

    def is_valid_dataset_name(self):

        if self.dataset_name not in DATASETS:
            raise ValueError(f"[ERROR] The dataset {self.dataset_name} \
                             doesn't exists")

    def get_dataset_classes(self):

        if self.dataset_name == 'camvid':
            return 32
        elif self.dataset_name == 'cityscapes':
            return 30
        else:
            raise ValueError(f"[ERROR] The dataset {self.dataset_name} \
                             doesn't exists")

    def get_dataset_sizes(self):

        if self.dataset_name == 'camvid':
            return [960, 720]
        elif self.dataset_name == 'cityscapes':
            return [2048, 1024]

    def count_pixels_per_class(self):

        if self.img_sizes is not None:
            return None

        if self.dataset_name == 'camvid':
            pixel_count = [1004, 1844, 6739, 21618, 160306, 23887, 556,
                           851, 7198, 20116, 13026, 2599, 5280, 2011, 5008,
                           8807, 4757, 197301, 18807, 49328, 1420, 105106,
                           10884, 760, 4040, 0, 78482, 5539, 4,
                           13402, 18140, 13037]
        elif self.dataset_name == 'cityscapes':
            pixel_count = [1004, 1844, 6739, 21618, 160306, 23887, 556,
                           851, 7198, 20116, 13026, 2599, 5280, 2011, 5008,
                           8807, 4757, 197301, 18807, 49328, 1420, 105106,
                           10884, 760, 4040, 0, 78482, 5539, 4,
                           13402, 18140, 13037]
        else:
            raise ValueError(f'[ERROR] Not valid dataset name: {self.dataset}')

        # find the minimum and maximum values
        min_pixel = min(pixel_count)
        max_pixel = max(pixel_count)

        # normalize each value in the range of 0-1
        normalized_pixels = [(v - min_pixel) / (max_pixel - min_pixel) for v in pixel_count]

        return normalized_pixels

    def get_fmaps(self):

        # Set the original feature map
        fmap = self.img_sizes

        # Get the max and min feature maps (Width and Height)
        fmap_w = max(fmap)
        fmap_h = min(fmap)

        # Get each feature map until to match the last fmap size established
        fmaps = []
        while fmap_w > self.last_fmap:

            # Getting the max and min sizes of feature map dimensions
            fmap_w = max(fmap)
            fmap_h = min(fmap)

            # Reduce feature map
            fmap = [x/2 for x in fmap]
            fmap = [math.ceil(x) for x in fmap]
            fmap = [int(x) for x in fmap]

            # Adding the feature map
            fmaps.append([fmap_w, fmap_h])

        return fmaps

    def split_fmaps(self, attention_modules=False):

        if attention_modules:
            fmaps = [fmap[0] for fmap in self.fmaps]
        else:
            fmaps = self.fmaps

        # Validate input
        if not all(isinstance(fmap, (int, list, tuple)) for fmap in fmaps):
            raise ValueError("Input must be a list or tuple of integers")

        # Convert tuples to lists
        fmaps = [list(fmap) if isinstance(fmap, tuple) else fmap for fmap in fmaps]

        # Calculate the indices for dividing the feature maps
        fmaps_length = len(fmaps)
        one_third = fmaps_length // 3
        two_thirds = 2 * (fmaps_length // 3)

        # Validate that feature maps can be divided into three parts
        if one_third == 0 or two_thirds == one_third:
            raise ValueError("Feature maps cannot be divided into three parts")

        # Divide the list into three parts
        part1 = fmaps[:one_third]
        part2 = fmaps[one_third:two_thirds]
        part3 = fmaps[two_thirds:]

        return part1, part2, part3

    def get_base_cfg(self):

        # Select JetSeg mode
        if self.mode == 1:
            # Workstation mode (better mIoU ???)

            self.input_base_features = 32
            self.input_expand_features = 2
            self.output_base_features = 32
            self.stage_features = 2
            self.expand_features = 2
            self.max_features = 512
            self.max_expand_features = 1024
            self.last_features = 512
            self.decoder_heads_output_features = 512
            self.residuals = [0, 2, 4, 2]
            self.last_fmap = 16

        elif self.mode == 2:
            # AGX Xavier mode (best trade-off mIoU/FPS)

            self.input_base_features = 16
            self.input_expand_features = 2
            self.output_base_features = 16
            self.stage_features = 2
            self.expand_features = 2
            self.max_features = 256
            self.max_expand_features = 256
            self.last_features = 256
            self.decoder_heads_output_features = 256
            self.residuals = [0, 2, 4, 2]
            self.last_fmap = 16

        elif self.mode == 3:
            # Nano mode (best for Low-Power Embedded Systems)

            self.input_base_features = 8
            self.input_expand_features = 2
            self.output_base_features = 8
            self.stage_features = 2
            self.expand_features = 2
            self.max_features = 64
            self.max_expand_features = 64
            self.last_features = 64
            self.decoder_heads_output_features = 64
            self.residuals = [0, 0, 1, 1]
            self.last_fmap = 16

        else:
            raise ValueError('[ERROR] Not valid JetSeg mode: {self.mode}')

        assert self.input_base_features >= 1, "input_base_features \
            must be at least 1"
        assert self.input_expand_features >= 1, "input_expand_features \
            must be at least 1"
        assert self.output_base_features >= 1, "output_base_features \
            must be at least 1"
        assert self.stage_features >= 1, "stage_features must be at least 1"
        assert self.expand_features >= 1, "expand_features must be at least 1"
        assert self.max_features >= 1, "max_features must be at least 1"
        assert self.max_expand_features >= 1, "max_expand_features \
            must be at least 1"
        assert self.last_features >= 1, "last_features must be at least 1"
        assert self.max_features >= self.last_features, "max_features \
            must be greater than or equal to last_features"
        assert self.max_expand_features >= self.last_features, "\
            max_expand_features must be greater than or equal to last_features"
        assert self.decoder_heads_output_features >= 1, "\
            decoder_heads_output_features must be at least 1"

    def get_arch_stage(self, fmap):

        # Iterate over each stage in the architecture
        for i, stage_fmaps in enumerate(self.fmaps_stages.values()):

            # Check if the fmap size is in the range of the current stage
            if fmap in stage_fmaps:
                return i

        # If the fmap size is not in any stage, return -1 as an error code
        return -1

    def get_fmaps_stages(self):

        # Division of feature maps in three parts (for stage 1, 2 and 3)
        fmaps_stage1, fmaps_stage2, \
            fmaps_stage3 = self.split_fmaps()

        # Get the first feature map for stage 0
        fmap_stage0 = fmaps_stage1.pop(0)

        # Return a dictionary containing the feature maps for each stage
        return {0: [fmap_stage0], 1: fmaps_stage1,
                2: fmaps_stage2, 3: fmaps_stage3}

    def set_residuals(self):
        updated_stages = {}

        # Get the residual number for each stage
        for stage, fmaps in self.fmaps_stages.items():

            # Get the number of residuals for the stage
            n_residuals = self.residuals[stage]
            n_residuals = [n_residuals]

            # Get new stage configuration with residuals
            stage_cfg = [[fmap + n_residuals] for fmap in fmaps]

            # Update new stage configuration
            updated_stages[stage] = stage_cfg

        return updated_stages

    def get_block_cfg(self, stage, in_channels):

        # Define the different cases as functions
        def case_0():
            return self.input_base_features, \
                self.input_base_features * self.input_expand_features, \
                self.input_base_features

        def case_1():
            return in_channels + self.output_base_features, \
                in_channels * self.expand_features, None

        def case_2():
            return in_channels + self.output_base_features * \
                self.stage_features, in_channels * self.expand_features, None

        def case_3():
            return in_channels * self.stage_features, in_channels * \
                self.expand_features, None

        # Map the stage number to its corresponding case function
        case_functions = {0: case_0, 1: case_1, 2: case_2, 3: case_3}
        out_ch, exp_ch, _ = case_functions[stage]()

        # Validate max features
        out_ch = min(out_ch, self.max_features)
        exp_ch = min(exp_ch, self.max_expand_features)

        return int(in_channels), int(exp_ch), int(out_ch)

    def get_attention_fmaps(self, block_type=None):

        if not block_type:
            return None

        # Get the corresponding blocks
        if block_type.lower() == "cbam":
            fmaps = self.split_fmaps(attention_modules=True)[0]

        elif block_type.lower() == "sam":
            fmaps = self.split_fmaps(attention_modules=True)[1]

        elif block_type.lower() == "ecam":
            fmaps = self.split_fmaps(attention_modules=True)[2]

        else:
            raise ValueError("[ERROR] Invalid attention module type. \
                             Supported values are 'cbam', 'sam', and 'ecam'")
        return fmaps

    def build_encoder_config(self, in_ch=3):

        # Initialize architecture config list
        arch_cfg = []

        for stage_idx, stage_cfg in self.fmaps_stages.items():
            # For stage configuration
            for cfg in stage_cfg:
                # For each feature map in stage configuration
                for fmaps_cfg in cfg:
                    # Extract feature map sizes (width and height)
                    fmap_w = fmaps_cfg[0]
                    fmap_h = fmaps_cfg[1]

                    # Get the number of residuals
                    n_residuals = fmaps_cfg[2]

                    # Get block configuration for current stage and block idx
                    in_ch, exp_ch, out_ch = self.get_block_cfg(stage_idx, in_ch)

                    # For n residuals, repeat the block (add residual)
                    for residual in range(n_residuals):

                        # If residual connection, the input and output channels should be the same
                        exp_ch = in_ch
                        out_ch = in_ch

                        # Create Block Configuration
                        block_cfg = [fmap_w, fmap_h, stage_idx, True, in_ch, exp_ch, out_ch]

                        # Add block configuration to the architecture config list
                        arch_cfg.append(block_cfg)

                        # Update input channels for the next block
                        in_ch = out_ch

                    # Get block configuration for current stage and block idx
                    in_ch, exp_ch, out_ch = self.get_block_cfg(stage_idx, in_ch)

                    # Add non-residual blocks
                    block_cfg = [fmap_w, fmap_h, stage_idx, False, in_ch, exp_ch, out_ch]

                    # Add block configuration to the architecture config list
                    arch_cfg.append(block_cfg)

                    # Update input channels for the next block
                    in_ch = out_ch

        return arch_cfg

    # def build_encoder_config(self, in_ch=3):

    #     # Initialize architecture config list
    #     arch_cfg = []

    #     for stage_idx, stage_cfg in self.fmaps_stages.items():

    #         # For stage configuration
    #         for cfg in stage_cfg:

    #             # For each feature map in stage configuration
    #             for fmaps_cfg in cfg:

    #                 # Extract feature map sizes (width and height)
    #                 fmap_w = fmaps_cfg[0]
    #                 fmap_h = fmaps_cfg[1]

    #                 # Get the number of residuals
    #                 n_residuals = cfg[-1].pop()

    #                 # Get block configuration for current stage and block idx
    #                 # in_ch, exp_ch, out_ch = self.get_block_cfg(stage_idx, in_ch)

    #                 # For n residuals repeats block (add residual)
    #                 for residual in range(0, n_residuals):

    #                     # Get block configuration for current stage and block idx
    #                     in_ch, exp_ch, out_ch = self.get_block_cfg(stage_idx, in_ch)

    #                     # If residual connection the input and output channels
    #                     # should be the same
    #                     exp_ch = in_ch
    #                     out_ch = in_ch

    #                     # Create Block Configuration
    #                     block_cfg = [fmap_w, fmap_h, stage_idx,
    #                                  True, in_ch, exp_ch, out_ch]

    #                     # Add block configuration to the architecture config list
    #                     arch_cfg.append(block_cfg)

    #                     # Update input channels for next block
    #                     in_ch = out_ch

    #             # Add non residual blocks
    #             block_cfg = [fmap_w, fmap_h, stage_idx,
    #                         False, in_ch, exp_ch, out_ch]

    #             # Add block configuration to the architecture config list
    #             arch_cfg.append(block_cfg)

    #             # Update input channels for next block
    #             in_ch = out_ch

    #     return arch_cfg

    def build_decoder_config(self):

        # Get the cbam feature map based on the combination
        cbam_fmaps = self.get_attention_fmaps("cbam")
        if len(cbam_fmaps) > 1:
            cbam_fmaps = cbam_fmaps[1:]

        first_idx = self.jetseg_comb.pop("decoder_head_stage1")
        first_fmap = cbam_fmaps[first_idx]

        # Get the sam feature map based on the combination
        sam_fmaps = self.get_attention_fmaps("sam")
        middle_idx = self.jetseg_comb.pop("decoder_head_stage2")

        if middle_idx is None:
            middle_idx = len(sam_fmaps) // 2

        middle_fmap = sam_fmaps[middle_idx]

        # Get the ecam feature map based on the combination
        ecam_fmaps = self.get_attention_fmaps("ecam")
        last_idx = self.jetseg_comb.pop("decoder_head_stage3")
        last_fmap = ecam_fmaps[last_idx]

        # Create the encoded FMAPS
        decoder_fmaps = [last_fmap, middle_fmap, first_fmap]

        # Remove residual blocks
        decoder_blocks = [decoder_block for decoder_block in
                          self.encoder_arch if not
                          any(element is True for element in decoder_block)]
        decoder_blocks = [decode_block for decode_block in
                          decoder_blocks if decode_block[0] in decoder_fmaps]
        decoder_blocks.reverse()

        return decoder_blocks

    def build_model_config(self):

        # Build encoder configuration
        encoder_blocks = self.build_encoder_config()

        # Update encoder configuration in order to build decoder
        self.encoder_arch = encoder_blocks
        # print(f'[DEBUG] JetNet Architecture: {self.encoder_arch}')

        # Build decoder configuration
        self.decoder_arch = self.build_decoder_config()
        # print(f'[DEBUG] Decoder Architecture: {self.decoder_arch}')


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:

        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def min_difference_value(lst, num):
    min_diff = float('inf')
    min_val = None
    for val in lst:
        diff = abs(val - num)
        if diff < min_diff:
            min_diff = diff
            min_val = val

    if min_val is None:
        raise ValueError("The min value is invalid, check and select an correct number")

    return min_val


def get_path(path_name):
    # Getting path workspace
    workspace_path = pathlib.Path().absolute()
    workspace_path = workspace_path.parent
    workspace_path = str(workspace_path)
    # Settng path name
    path = workspace_path + '/' + path_name + '/'
    return path


def _init_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
           '%(asctime)-15s:%(levelname)s:%(module)s.py:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def pkl(file, objects=None, mode=False):
    data = None

    if mode:
        # Open pickle file in write bytes mode
        with open(file, "wb") as pickle_file:
            # Dump objects to pickle file
            pickle.dump(objects, pickle_file)
    else:
        # Open pickle file in read bytes mode
        with open(file, "rb") as pickle_file:
            # Load data from pickle file
            data = pickle.load(pickle_file)

    return data


def clean_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def make_directory(path):
    try:
        os.mkdir(path)
        return True
    except Exception:
        return False


def color_map(csv_file):

    '''
    Returns the reversed String.
    Parameters:
        dataframe: A Dataframe with rgb values with class maps.
    Returns:
        code2id: A dictionary with color as keys and class id as values.
        id2code: A dictionary with class id as keys and color as values.
        name2id: A dictionary with class name as keys and class id as values.
        id2name: A dictionary with class id as keys and class name as values.
    '''

    cls = pd.read_csv(csv_file)
    color_code = [tuple(cls.drop("name", axis=1).loc[idx]) for idx in range(len(cls.name))]
    code2id = {v: k for k, v in enumerate(list(color_code))}
    id2code = {k: v for k, v in enumerate(list(color_code))}

    color_name = [cls['name'][idx] for idx in range(len(cls.name))]
    name2id = {v: k for k, v in enumerate(list(color_name))}
    id2name = {k: v for k, v in enumerate(list(color_name))}

    return code2id, id2code, name2id, id2name


def label_colors(rgb, mask, color_dict, num_classes):

    # Set the label for each color map in order to generate the mask
    for label, color in color_dict.items():

        # Create a np array color
        color = np.array(color, dtype=np.uint8)

        if label < num_classes:
            mask[np.all(rgb == color, axis=-1)] = label

    return mask


def binary_mask(rgb, label_mask, num_classes):
    out_mask = np.zeros((num_classes,) + rgb.shape[:2], dtype=np.uint8)

    for label in range(num_classes):
        out_mask[label][label_mask == label] = 1

    return out_mask


def batch_binary_mask(rgb, label_mask, num_classes):
    batch_size, _, width, height = label_mask.shape
    out_mask = np.zeros((batch_size, num_classes, width, height),
                        dtype=np.uint8)

    for label in range(num_classes):
        out_mask[:, label, :, :][label_mask[:, label, :, :] == label] = 1

    return torch.from_numpy(out_mask)


def rgb_to_mask(rgb, color_dict):

    # Get the number of classes
    num_classes = len(color_dict.keys())

    # rgb shape: (h,w,3); arr shape: (h,w)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    # Get the label mask
    label_mask = label_colors(rgb, mask, color_dict, num_classes)

    # Get binary mask
    out_mask = binary_mask(rgb, label_mask, num_classes)

    return out_mask, label_mask


def mask_to_rgb(mask, color_dict):
    num_classes, height, width = mask.shape

    # Create an empty RGB image
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Set the color for each class label
    for label, color in color_dict.items():
        if label < num_classes:

            # Set the pixels with the current label to the corresponding color
            rgb[mask[label] == 1] = color

    return rgb


def val_residual_op(in_ch, out_ch, res):
    if not res:
        return 0
    if in_ch != out_ch:
        raise ValueError("In channels != out channels can't add residual")


def gen_conv_permutations(k_sizes):
    convolutions = []
    for k in k_sizes:
        if k == 1:
            continue
        convolutions.append((k, k))
        for i in range(1, k):
            if i == 1:
                convolutions.append([(k, 1), (1, k)])
            elif k % i == 0:
                convolutions.append([(k, i), (i, k)])

    # Flatten combinations
    convolutions = [item for sublist in convolutions for item in ([sublist] if isinstance(sublist, tuple) else sublist)]

    return convolutions


def get_conv_types(convs):
    std_convs = []
    asym_convs = []
    for conv in convs:
        if isinstance(conv, tuple):
            if conv[0] == conv[1]:
                std_convs.append(conv)
            else:
                asym_convs.append(conv)

    std_convs = [[conv] for conv in std_convs]

    aconvs = []
    for i in range(0, len(asym_convs), 2):
        aconvs.append([asym_convs[i], asym_convs[i+1]])

    return std_convs, aconvs


def get_results(results, results_type="best", n_results=4):

    if results_type.lower() == "best":
        # Find N best results
        indices = sorted(range(len(results)),
                         key=lambda i: results[i], reverse=True)[:n_results]

    elif results_type.lower() == "worst":
        # Find N worst results
        indices = sorted(range(len(results)),
                         key=lambda i: results[i])[:n_results]

    elif results_type.lower() == "random":
        # Find N random results
        indices = random.sample(range(len(results)), n_results)

    else:
        raise ValueError(f'[ERROR] Not valid type of results {results_type}')

    return indices
