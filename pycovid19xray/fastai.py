import logging
import torch
from pathlib import Path
from pycovid19xray.tests import DATA_DIR
from pycovid19xray.eval import report_metrics
import numpy as np
import pandas as pd
from fastai.vision import cnn_learner, models, DatasetType, ImageList,\
    imagenet_stats, get_transforms
import fastai.vision as fv
from fastai.metrics import error_rate
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def get_dataaug_transformations():
    tfms = get_transforms(flip_vert=False, max_lighting=0.1,
                          max_rotate=10,
                          max_zoom=1.15, max_warp=None)
    return tfms

class FastaiModel():

    def __init__(self, modelname, bs=64, version="v3"):
        self.modelname = modelname
        self.bs = bs
        self.version = version
        self.figure_path = Path(f"./figures/{self.modelname}")
        self.target_names = ['COVID-19', "Normal", "Pneumonia"]
        if not self.figure_path.exists():
            self.figure_path.mkdir()
    def save_fig(self, figname):
        plt.savefig(self.figure_path/figname)

    def create_covidx_databunch(self):
        bs = self.bs

        data_path = DATA_DIR / "COVIDx"
        assert data_path.exists()
        train_df_path = data_path / f"train_split_{self.version}.txt"
        # train set
        LOGGER.info(f'Reading train_df from {train_df_path}')
        self.train_df = (pd.read_csv(train_df_path,
                                header=None, delimiter=" ",
                                index_col=0, names=["name", "label", "dataset"])
                .reset_index(drop=True))
        self.train_df["name"] = ["train/" + f for f in self.train_df["name"]]
        self.train_df["is_valid"] = False

        # validation set
        test_df_path = data_path / f"test_split_{self.version}.txt"
        # train set
        LOGGER.info(f'Reading test_df from {test_df_path}')
        self.test_df = (pd.read_csv(data_path / test_df_path,
                               header=None, delimiter=" ",
                               index_col=0, names=["name", "label", "dataset"])
                .reset_index(drop=True))
        self.test_df["name"] = ["test/" + f for f in self.test_df["name"]]
        self.test_df["is_valid"] = True

        # merge
        data_df = pd.concat([self.train_df, self.test_df]).reset_index(drop=True)
        data_df = data_df.drop("dataset", axis=1)

        # import covidnet test set which is included in testset
        covidnet_test_df = (pd.read_csv(data_path / "test_COVIDx4.txt",
                                        header=None, delimiter=" ",
                                        index_col=0, names=["name", "label"])
                            .reset_index(drop=True))
        covidnet_test_df["name"] = ["test/" + f for f in covidnet_test_df["name"]]
        # sanity check
        a = set(covidnet_test_df.name)
        b = set(self.train_df.name)
        c = set(self.test_df.name)
        assert a.intersection(b) == set()
        assert a.intersection(c) == a

        # create fastai databunch
        tfms = get_dataaug_transformations()

        np.random.seed(42)
        src = (ImageList.from_df(data_df, data_path)
            .split_from_df()
            .label_from_df()
            .transform(tfms, size=224))
        test = (ImageList.from_df(covidnet_test_df, data_path)
                .split_none()
                .label_from_df()
                .transform(None, size=224))

        data = (src.databunch(bs=bs)
                .normalize(imagenet_stats))
        data.add_test(test.train.x)

        # check that proportion classes are same in train and valid
        train_counts = np.unique(data.train_ds.y.items, return_counts=True)
        LOGGER.info(f'prop in train set: {train_counts[1]/ train_counts[1].sum()}')
        valid_counts = np.unique(data.valid_ds.y.items, return_counts=True)
        LOGGER.info(f'prop in valid set: {valid_counts[1]/ valid_counts[1].sum()}')
        test_counts = np.unique(test.y.items, return_counts=True)
        LOGGER.info(f'prop in test set: {test_counts[1]/ test_counts[1].sum()}')

        return data, test

    def create_learner(self, data, loss_func=torch.nn.CrossEntropyLoss()):
        return cnn_learner(data, models.resnet50, metrics=error_rate,
                           loss_func=loss_func)

    def load_learner(self, learn, stage="stage-2"):
        learn.load(f'{stage}-{self.modelname}')
        return learn

    def save_learner(self, learn, stage):
        learn.save(f'{stage}-{self.modelname}')
        return learn



def show_metrics(learn, test):
    # eval on test dataset
    preds, y = learn.get_preds(DatasetType.Valid)
    pred_label = torch.argmax(preds, axis=1)
    print("= VALIDATION SET: covidx whole test set=")
    report_metrics(y, pred_label)
    preds, _ = learn.get_preds(DatasetType.Test)
    pred_label = torch.argmax(preds, axis=1)
    y = test.train.y.items
    print("= TEST SET: On covidx test set (100 of each class)=")
    report_metrics(y, pred_label)
