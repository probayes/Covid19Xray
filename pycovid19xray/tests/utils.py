from pycovid19xray.tests import DATA_DIR
from pycovid19xray.data import COVIDX_3D_Dataset,\
    DATA_MEAN, DATA_STD
from pycovid19xray.fastai import FastaiModel

import torch
from torchvision import transforms
import torchxrayvision as xrv
import numpy as np
import pandas as pd


def load_fastai_model(modelname="rs50-dataaug"):

    # dataset
    fastaimodel = FastaiModel(modelname=modelname,
                              bs=64)
    data, test = fastaimodel.create_covidx_databunch()

    # get model
    learn = fastaimodel.create_learner(data)
    learn = fastaimodel.load_learner(learn, "stage-2")
    device = torch.device("cuda")
    model = learn.model
    model.eval()
    model.to(device)

    return model, fastaimodel.target_names


def load_test_df(label_filename="test_split_v3.txt"):
    data_path = DATA_DIR / "COVIDx"
    assert data_path.exists()
    img_path = data_path / 'test'
    label_path = data_path / label_filename
    label_df = pd.read_csv(label_path, header=None, delimiter=" ",
                           index_col=0,
                           names=["img", "label", "dataset"])
    return label_df


def load_test_dataset():
    data_path = DATA_DIR / "COVIDx"
    assert data_path.exists()
    img_path = data_path / 'test'
    label_path = data_path / "test_split_v3.txt"
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
        transforms.Lambda(
            lambda x: np.moveaxis(x, 0, -1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DATA_MEAN,
            std=DATA_STD)
    ])
    dataset = COVIDX_3D_Dataset(img_path, label_path, transform, data_aug=None)

    return dataset

def denormalize_f(img):
    inp = img[0,...].data.cpu().numpy().transpose((1, 2, 0))
    inp = DATA_STD * inp + DATA_MEAN
    inp = np.clip(inp, 0, 1)
    return inp
