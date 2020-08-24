import logging
import numpy as np
import pandas as pd
import torch
import random

from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
from torch.utils.data.dataset import Dataset
from torchxrayvision.datasets import normalize


LOGGER = logging.getLogger(__name__)

DATA_MEAN = [0.485, 0.456, 0.406]
DATA_STD = [0.229, 0.224, 0.225]

class COVIDX_Dataset(Dataset):
    """COVIDx dataset

    Dataset: https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md

    Paper: https://arxiv.org/pdf/2003.09871.pdf

    """

    def __init__(self,
                 img_path,
                 label_path,
                 transform=None,
                 data_aug=None,
                 seed=None):

        # image files
        types = ('*.png', '*.jpg', '*.jpeg', '*.JPG')
        files = list()
        for t in types:
            files.extend(img_path.glob(t))

        # label data frame
        label_df = pd.read_csv(label_path, header=None, delimiter=" ",
                               index_col=0, names=["img", "label", "dataset"])

        self.img_path = img_path
        all_img = set([f.name for f in files])
        self.csv = label_df.query(f"img in @all_img").copy()
        self.pathologies = ['normal', 'pneumonia', 'COVID-19']
        self.c = 3
        self.classes_dict = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
        self.csv["label_id"] = [self.classes_dict[l]
                                for l in self.csv.label]
        self.labels = pd.get_dummies(self.csv['label_id']).to_numpy().astype(np.float32)  # useless?
        # TODO: check that get_dummies() methods sorts label_id
        self.transform = transform
        self.data_aug = data_aug
        self.seed = seed
        self.MAXVAL = 255  # for consistency with torchxrayvision dataset classes

    def load_img(self, img_f, transform, to_greyscale=False):
        image = imread(img_f)  # interest of RGB conversion?
        image = normalize(image, self.MAXVAL)  # rxv's method to scale images to be ~ [-1024 1024]
        if len(image.shape) > 2:
            assert len(image.shape) == 3
            if to_greyscale:  # TODO: check relevance
                image = rgb2gray(image)
            else:
                image = image[:, :, 0]
        if len(image.shape) < 2:  # TODO: log instead of print
            print("error, dimension lower than 2 for image")
        image = image[None, :, :]
        if transform is not None:
            image = transform(image)
        return image

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.csv.img.values[idx]
        label = self.csv.label_id.values[idx]
        image = self.load_img(self.img_path / img_name, self.transform)
        if self.data_aug is not None:
            if self.seed is not None:
                random.seed(self.seed)
            image = self.data_aug(image)

        return {"img": image, "lab": label, "idx": idx}

    def get_size_stats(self):
        """Extract and export statistics on image dimensions.
        :return: DataFrame with one row per image and columns 'nb_dim' (image dimension, e.g., 2 or
        3D), 'lowest_dim_loc' (index of the smallest dimension, e.g. "2" for image 224*224*3),
        'lowest_dim_value' (number of channels) and 'channel_equality' (True if all channels contain
        the same information).
        """
        info_list = []
        for idx in range(len(self)):
            img_name = self.csv.img.values[idx]
            image_path = self.img_path / img_name
            image = imread(image_path)
            nb_dim = len(image.shape)
            if nb_dim > 2:
                lowest_dim_loc = np.argmin(image.shape)
                lowest_dim_value = np.min(image.shape)
                channel_equality = (np.diff(image, axis=lowest_dim_loc).sum() == 0)
            else:
                lowest_dim_loc = None
                lowest_dim_value = None
                channel_equality = None
            info_list.append(
                {'nb_dim': nb_dim,
                 'lowest_dim_loc': lowest_dim_loc,
                 'lowest_dim_value': lowest_dim_value,
                 "channel_equality": channel_equality})
        df_image_stats = pd.DataFrame(info_list)
        df_image_stats.to_csv(self.img_path / 'df_images_stats.csv')
        return df_image_stats


class COVIDX_3D_Dataset(Dataset):
    """COVIDx dataset with grayscales images returned as 3-channel images (3 identical channels if
    only one channel does exist).
    May be merge with COVIDX_Dataset

    Dataset: https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md

    Paper: https://arxiv.org/pdf/2003.09871.pdf

    """

    def __init__(self,
                 img_path,
                 label_path,
                 transform=None,
                 data_aug=None,
                 seed=None):

        # image files
        types = ('*.png', '*.jpg', '*.jpeg', '*.JPG')
        files = list()
        for t in types:
            files.extend(img_path.glob(t))

        # label data frame
        label_df = pd.read_csv(label_path, header=None, delimiter=" ",
                               index_col=0,
                               names=["img", "label", "dataset"])

        self.img_path = img_path
        all_img = set([f.name for f in files])
        self.csv = label_df.query(f"img in @all_img").copy()
        self.pathologies = ['normal', 'pneumonia', 'COVID-19']
        self.c = 3
        self.classes_dict = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
        self.csv["label_id"] = [self.classes_dict[l]
                                for l in self.csv.label]
        self.labels = pd.get_dummies(
            self.csv['label_id']).to_numpy().astype(np.float32)  # useless?
        # TODO: check that get_dummies() methods sorts label_id
        self.transform = transform
        self.data_aug = data_aug
        self.seed = seed

    def load_img(self, img_f, transform, to_greyscale=False, verbose=False):
        image = imread(img_f)  # interest of RGB conversion?
        if len(image.shape) == 2:  # only (x,y) image (no "color" channel)
            # duplicating grayscale channel to get a 3-channel image
            # to get dimension : (color, x, y) 
            image = np.concatenate((image[None, :, :], image[None, :, :], image[None, :, :]),
                                   axis=0)
        elif len(image.shape) == 3:  # x*y*C
            image = np.moveaxis(image, -1, 0)  # (x, y, color) -> (color, x, y)
            if image.shape[0] == 4:  # 4 colors instead of 3 like RGB
                if verbose:
                    print("4 channels instead of 3, taking the 3 first channels")
                image = image[:3, :, :]  # cf. example where the last dimension contains only "255"
                # TODO: improve rigour of image preprocessing
        else:  # TODO: log instead of print
            print("error, dimension {} for image".format(len(image.shape)))
            image = None
        if transform is not None and image is not None:
            image = transform(image)
        if image is None:
            print('None image')
        return image

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.csv.img.values[idx]
        label = self.csv.label_id.values[idx]
        image = self.load_img(self.img_path / img_name, self.transform)
        if self.data_aug is not None:
            if self.seed is not None:
                random.seed(self.seed)
            image = self.data_aug(image)

        return {"img": image, "lab": label, "idx": idx}

