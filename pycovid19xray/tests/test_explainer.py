from pathlib import Path
from pycovid19xray.utils import configure_logging, set_gpu
configure_logging()
# set_gpu(1)

import torch
import torchvision
import torchxrayvision as xrv
from pycovid19xray.data import COVIDX_3D_Dataset
import numpy as np
import random
from pycovid19xray.tests.utils import (load_fastai_model, load_test_dataset,
                                       denormalize_f, load_test_df)
from pycovid19xray.tests import DATA_DIR
from pycovid19xray.explainer import Explainer
from torchvision import models, transforms
import json

#######
# data

def given_resnet50():
    model = models.resnet50(pretrained=True)
    layer = model.layer4
    img_path = DATA_DIR / "torch/resnet50/exemples/cat_dog.png"
    # read image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    with open(DATA_DIR / "torch/imagenet/imagenet_class_index.json") as f:
        class_idx = json.load(f)
    class_idx = {int(k): v[1] for k, v in class_idx.items()}

    return model, layer, transform, transform_normalize, class_idx, img_path

def given_resnet50_covidx():

    # given
    model, target_names = load_fastai_model()
    data_df = load_test_df()
    # read image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resnet_core = next(model.children())
    last_cnn_layer = list(resnet_core.children())[7]
    class_idx = {0: "COVID-19", 1: "normal", 2: "pneumonia"}



    return model, last_cnn_layer, transform, transform_normalize, class_idx, data_df

def given_resnet50_covidxv4():

    modelname = "COVIDx4_3D-resnet50-frozen_pretrained_balanced_normed3D_resnet50"
    model_dump_path = DATA_DIR / f"COVIDx/models/output_resnet50_frozen_pretrained_COVIDx4/{modelname}-best.pt"
    model = torch.load(model_dump_path)
    device = torch.device("cuda")
    model = model.to(device)
    last_cnn_layer = model.layer4
    data_df = load_test_df("test_split_v4.txt")
    data_path = DATA_DIR / "COVIDx/test"
    one_img_path = data_path / data_df.img.values[0]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    class_idx = {2: "COVID-19", 0: "normal", 1: "pneumonia"}
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
        torchvision.transforms.Lambda(
            lambda x: np.moveaxis(x, 0, -1)),
        torchvision.transforms.ToTensor()])
    transform_normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # dataset to load images
    imgs_path = DATA_DIR / 'COVIDx/test'
    label_path = DATA_DIR / "COVIDx/test_split_v4.txt"
    test_dataset = COVIDX_3D_Dataset(imgs_path,
                                     label_path,
                                     transform=transforms,
                                     data_aug=None)



    return (modelname, model, last_cnn_layer, transform,
            transform_normalize, class_idx, data_df,
            one_img_path, test_dataset)


######
# test

def test_cache_explainer():

    # given
    model, layer, transform, transform_normalize, class_idx, img_path =\
        given_resnet50()


    # with
    explainer = Explainer(model,
                          layer,
                          "resnet50",
                          transform,
                          transform_normalize,
                          class_idx,
                          cache_path="./cache")
    # self = explainer

    # then
    explainer.arraycache.clear_cache()
    t = explainer.methods["occlusion"]
    arr0 = t[0](img_path, 0)
    arr1 = t[0](img_path, 0)
    assert np.allclose(arr0, arr1)
    arr0 = t[0](img_path, 0, True)
    assert np.allclose(arr0, arr1)
    arr2 = t[0](img_path, 1, False)
    assert not np.allclose(arr0, arr2)


def test_explainer_resnet50():

    # given
    model, layer, transform, transform_normalize, class_idx, img_path =\
        given_resnet50()


    # with
    explainer = Explainer(model,
                          layer,
                          "resnet50",
                          transform,
                          transform_normalize,
                          class_idx)

    # then
    true_class = "dog"
    explainer.plot_explanations(img_path, true_class, False)

def test_explainer_resnet50_covidx():

    # given
    model, layer, transform, transform_normalize, class_idx, data_df =\
        given_resnet50_covidx()
    data_path = DATA_DIR / "COVIDx/test"

    # with
    explainer = Explainer(model,
                          layer,
                          "resnet50-covidx",
                          transform,
                          transform_normalize,
                          class_idx,
                          grayscale_img=True)

    # then
    def plot_some(c):
        df = data_df.query(f"label == '{c}'")
        for i in random.sample(list(range(df.shape[0])), 10):
            img_path = data_path / df.img.values[i]
            explainer.plot_explanations(img_path, c, blurring=False)
    plot_some("COVID-19")
    plot_some("pneumonia")
    plot_some("normal")


def test_explainer_resnet50_covidxv4():

    # given
    (modelname, model, last_cnn_layer, transform,
     transform_normalize, class_idx, data_df,
     img_path, test_dataset) = given_resnet50_covidxv4()
    data_path = DATA_DIR / "COVIDx/test"

    # with
    explainer = Explainer(model,
                          last_cnn_layer,
                          f"{modelname}",
                          transform,
                          transform_normalize,
                          class_idx,
                          dataset=test_dataset,
                          cache_path="./cache")

    # then
    img, transformed_img, inp = explainer.open_image(img_path)
    assert transformed_img.shape == (3, 224, 224)
    # then
    def plot_some(c):
        df = data_df.query(f"label == '{c}'")
        for i in random.sample(list(range(df.shape[0])), 10):
            img_path = data_path / df.img.values[i]
            explainer.plot_explanations(img_path, c, blurring=False)
    plot_some("COVID-19")
    plot_some("normal")
    plot_some("pneumonia")

