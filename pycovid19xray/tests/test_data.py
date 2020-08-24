from pycovid19xray.tests import DATA_DIR
from pycovid19xray.data import COVIDX_Dataset, COVIDX_3D_Dataset,\
    DATA_MEAN, DATA_STD
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torchxrayvision as xrv
import numpy as np


def test_COVIDx_Dataset():

    # given
    data_path = DATA_DIR / "COVIDx"
    assert data_path.exists()
    img_path = data_path / 'train'
    label_path = data_path / "train_split_v3.txt"
    transform = (torchvision
                 .transforms
                 .Compose([xrv.datasets.XRayCenterCrop(),
                           xrv.datasets.XRayResizer(224),
                           torchvision.transforms.Lambda(
                               lambda x: np.moveaxis(x, 0, -1)),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                               mean=DATA_MEAN,
                               std=DATA_STD)
                 ]))
    data_aug = None

    # with
    dat = COVIDX_3D_Dataset(img_path, label_path, transform, data_aug)

    # then
    d = dat[3]
    assert len(dat) == 13793
    assert d["lab"] in set([0,1,2])
    image = d["img"]
    inp = image.numpy().transpose((1, 2, 0))
    inp = inp * DATA_STD + DATA_MEAN
    assert inp.shape == (224, 224, 3)

    plt.imshow(inp)
    plt.savefig("test_COVIDx_Dataset.png")


def test_COVIDx_dataloader():

    # given
    data_path = DATA_DIR / "COVIDx"
    assert data_path.exists()
    img_path = data_path / 'train'
    label_path = data_path / "train_split_v3.txt"
    transform = (torchvision
                 .transforms
                 .Compose([xrv.datasets.XRayCenterCrop(),
                           xrv.datasets.XRayResizer(224),
                           torchvision.transforms.Lambda(
                               lambda x: np.moveaxis(x, 0, -1)),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
                 ]))
    data_aug = None

    # with
    train_dat = COVIDX_Dataset(img_path, label_path, transform, data_aug)
    dataloader = DataLoader(train_dat, batch_size=32,
                            shuffle=True, num_workers=4)
    # ensure actually not shuffled
    batch, l = next(iter(dataloader))

    # then
    assert batch.shape == (32, 3, 224, 224)
    img_grid = torchvision.utils.make_grid(batch)
    fig = plt.figure(figsize=(10,10), dpi=150)
    plt.imshow(img_grid.permute(1,2,0))
    plt.savefig("test_COVIDx_dataloader.png")

