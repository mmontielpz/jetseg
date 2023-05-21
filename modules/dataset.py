import os
import sys
import random
import natsort
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset


from modules.utils import rgb_to_mask


CLASS_MAP = {
    0: (64, 128, 64), # animal
    1: (192, 0, 128), # archway
    2: (0, 128, 192), # bicyclist
    3: (0, 128, 64), #bridge
    4: (128, 0, 0), # building
    5: (64, 0, 128), #car
    6: (64, 0, 192), # car luggage pram...???...
    7: (192, 128, 64), # child
    8: (192, 192, 128), # column pole
    9: (64, 64, 128), # fence
    10: (128, 0, 192), # lane marking driving
    11: (192, 0, 64), # lane maring non driving
    12: (128, 128, 64), # misc text
    13: (192, 0, 192), # motor cycle scooter
    14: (128, 64, 64), # other moving
    15: (64, 192, 128), # parking block
    16: (64, 64, 0), # pedestrian
    17: (128, 64, 128), # road
    18: (128, 128, 192), # road shoulder
    19: (0, 0, 192), # sidewalk
    20: (192, 128, 128), # sign symbol
    21: (128, 128, 128), # sky
    22: (64, 128, 192), # suv pickup truck
    23: (0, 0, 64), # traffic cone
    24: (0, 64, 64), # traffic light
    25: (192, 64, 128), # train
    26: (128, 128, 0), # tree
    27: (192, 128, 192), # truck/bus
    28: (64, 0, 64), # tunnel
    29: (192, 192, 0), # vegetation misc.
    30: (0, 0, 0),  # 0=background/void
    31: (64, 192, 0), # wall
}

# all the classes that are present in the dataset
ALL_CLASSES = [
    'animal', 'archway', 'bicyclist', 'bridge', 'building', 'car',
    'cartluggagepram', 'child', 'columnpole', 'fence', 'lanemarkingdrve',
    'lanemarkingnondrve', 'misctext', 'motorcyclescooter', 'othermoving',
    'parkingblock', 'pedestrian', 'road', 'road shoulder', 'sidewalk',
    'signsymbol', 'sky', 'suvpickuptruck', 'trafficcone', 'trafficlight',
    'train', 'tree', 'truckbase', 'tunnel', 'vegetationmisc', 'void', 'wall']


class Transforms:
    def __init__(self, img=None, mask=None, img_size=None):
        self.img = img
        self.mask = mask
        self.img_size = img_size

        if img is None and mask is None:
            raise ValueError("[ERROR] Both the image and the mask cannot be None")

        # Build transforms
        self.transform_img, self.transform_mask = self.build_transform()

    def get_normalize_info(self, dataset_name="camvid"):

        if dataset_name.lower() == "camvid":
            mean = [0.4132, 0.4229, 0.4301]
            std = [0.1096, 0.1011, 0.0963]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        return mean, std

    def is_pil_image(self, img):
        return isinstance(img, Image.Image)

    def build_transform(self):
        ops_img = []
        ops_mask = []

        if self.img is not None:

            # Resize image
            if self.img_size is not None:
                ops_img.append(T.Resize(self.img_size))

            # Convert to tensor and normalize image
            ops_img.append(T.ToTensor())

            # Get Dataset Normalization
            mean, std = self.get_normalize_info()
            ops_img.append(T.Normalize(mean, std))

        if self.mask is not None:

            # Resize mask
            if self.img_size is not None:
                ops_mask.append(T.Resize(self.img_size,
                                         interpolation=Image.NEAREST))

            # Convert to tensor
            ops_mask.append(T.ToTensor())

            # Convert to PIL image
            ops_mask.append(T.ToPILImage())

        return T.Compose(ops_img), T.Compose(ops_mask)

    def apply_transform(self):
        if self.img is not None:
            img = self.transform_img(self.img)
        else:
            img = None

        if self.mask is not None:
            mask = self.transform_mask(self.mask)
        else:
            mask = None

        return img, mask


class SSegmDataset(Dataset):
    def __init__(self, dataset_name, num_classes,
                 root_path, mode, img_size=None):

        # Getting dataset info
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.pixels_per_class = [[] for _ in range(self.num_classes)]

        self.id2code = CLASS_MAP
        self.code2id = {v: k for k, v in CLASS_MAP.items()}

        self.img_path = root_path + mode
        self.mask_path = root_path + mode + "_labels/"

        all_imgs = os.listdir(self.img_path)
        all_masks = [img_name[:-4] + "_L" + img_name[-4:] for img_name in all_imgs]

        self.tot_imgs = natsort.natsorted(all_imgs)
        self.tot_masks = natsort.natsorted(all_masks)
        self.img_size = img_size

    def get_dataset_mean_std(self):

        # Define a transformation to convert the images to PyTorch tensors
        transform = T.ToTensor()

        # Initialize lists to store the pixel values for each channel
        red_pixels = []
        green_pixels = []
        blue_pixels = []

        # Loop through all images in the dataset and store the pixel values for each channel
        loop = tqdm(range(0, self.__len__()), leave=True,
                    desc="[INFO] Computing mean and std of dataset")

        for idx in loop:

            # Getting sample (img and mask)
            img, _ = self.get_sample(idx)

            red_pixels.append(torch.mean(transform(img)[0]))
            green_pixels.append(torch.mean(transform(img)[1]))
            blue_pixels.append(torch.mean(transform(img)[2]))

        # Calculate the mean and standard deviation for each channel
        mean = (torch.mean(torch.tensor(red_pixels)), torch.mean(torch.tensor(green_pixels)), torch.mean(torch.tensor(blue_pixels)))
        std = (torch.std(torch.tensor(red_pixels)), torch.std(torch.tensor(green_pixels)), torch.std(torch.tensor(blue_pixels)))

        return mean, std

    def get_background_class_id(self):

        if self.dataset_name == "camvid":
            return 30

        elif self.dataset_name == "cityscapes":
            return 0

        else:
            raise ValueError(f'[ERROR] Not valid dataset name: {self.dataset_name}')

    def get_color_map(self):

        # Create the color map array
        color_map = np.zeros((self.num_classes, 3), dtype=np.uint8)

        for k, v in self.class_map.items():

            # Get color map for class i
            color_map[k, :] = v

        return color_map

    def get_sample(self, idx):

        # Getting img name
        img_name = os.path.join(self.img_path, self.tot_imgs[idx])

        # Load img
        img = Image.open(img_name).convert('RGB')

        # Getting label name
        mask_name = os.path.join(self.mask_path, self.tot_masks[idx])

        # Load label
        mask = Image.open(mask_name).convert('RGB')

        return img, mask

    def get_random_sample(self):

        # Generate a random idx
        idx = random.randint(0, len(self.tot_imgs)-1)

        # Getting random sample
        return self.__getitem__(idx)

    def get_pixels_per_class(self):

        # Creating the loop
        loop = tqdm(range(0, self.__len__()),
                    leave=True, desc="[INFO] Computing pixels per class")

        for idx in loop:

            # Getting sample (img and mask)
            img, mask = self.get_sample(idx)

            # Convert to np array
            mask = np.asarray(mask)

            # Reshape mask to 2D array
            mask = mask.reshape(-1, 3)

            # Get unique rows, and counter
            unique_rows, counts = np.unique(mask, axis=0, return_counts=True)

            # Add pixels per class
            for row, count in zip(unique_rows, counts):

                # Get class id
                row = tuple(row)

                class_id = self.color_map["code2id"][row]
                self.pixels_per_class[class_id] += [count]

        for idx in range(0, len(self.pixels_per_class)):

            if not self.pixels_per_class[idx]:
                self.pixels_per_class[idx] = [0]

        return [int(np.round(np.mean(np.array(pixels_per_class)))) for pixels_per_class in self.pixels_per_class]

    def __len__(self):
        return len(self.tot_imgs)

    def __getitem__(self, idx):

        # Getting sample (img and mask)
        img, mask = self.get_sample(idx)

        if self.img_size is None:
            self.img_size = list(img.size)

        # Create Transforms object
        transformer = Transforms(img=img, mask=mask, img_size=self.img_size)

        # Apply transformations
        img, rgb_mask = transformer.apply_transform()

        # Convert rgb to mask
        out_mask, _ = rgb_to_mask(np.array(rgb_mask, dtype=np.uint8),
                                  self.id2code)

        # Convert masks to torch tensor
        rgb_mask = np.array(rgb_mask).transpose(2, 0, 1)
        rgb_mask = torch.from_numpy(rgb_mask)
        out_mask = torch.from_numpy(out_mask)

        return img, out_mask, rgb_mask
