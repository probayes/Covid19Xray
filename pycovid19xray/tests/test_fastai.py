from pycovid19xray.utils import configure_logging, set_gpu
configure_logging()
set_gpu(1)

from pycovid19xray.tests import DATA_DIR
from pycovid19xray.fastai import get_dataaug_transformations
import matplotlib.pyplot as plt
import numpy as np
import fastai.vision as fv


def test_data_aug():

    # given
    data_path = DATA_DIR / "COVIDx"
    assert data_path.exists()
    img_path = data_path / 'test/0a51f668-b7b1-4d8d-9ab9-de1f702f071a.png'
    img = fv.open_image(img_path)

    # then brightness
    plt.close("all")
    fig, axs = plt.subplots(1,5,figsize=(12,4))
    for change, ax in zip([0.4, 0.5, 0.6, 1.0,1.1], axs):
        img = fv.open_image(img_path)
        fv.brightness(img, change).show(ax=ax, title=f'change={change:.1f}')
    plt.savefig("test_data_aug_bitghtness.jpg")

    # then rotation
    plt.close("all")
    tfm = [fv.rotate(degrees=(-10,10), p=0.75)]
    fig, axs = plt.subplots(1,5,figsize=(12,4))
    for ax in axs:
        img = fv.open_image(img_path)
        img = img.apply_tfms(tfm)
        title = f"Done, deg={tfm[0].resolved['degrees']:.1f}" if tfm[0].do_run else f'Not done'
        img.show(ax=ax, title=title)
    plt.savefig("test_data_aug_rotation.jpg")


def test_get_dataaug_transformations():

    # given
    data_path = DATA_DIR / "COVIDx"
    assert data_path.exists()
    img_path = data_path / 'test/0a51f668-b7b1-4d8d-9ab9-de1f702f071a.png'
    img = fv.open_image(img_path)

    # with
    tfms = get_dataaug_transformations()

    # then
    fig, axs = plt.subplots(1,5,figsize=(12,4))
    for ax in axs:
        img = fv.open_image(img_path)
        img = img.apply_tfms(tfms[0])
        img.show(ax=ax)
    plt.savefig("test_get_dataaug_transformations.jpg")


