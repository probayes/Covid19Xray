from pycovid19xray.utils import configure_logging, set_gpu
configure_logging()
# set_gpu(1)

import torch
from pycovid19xray.fastai import FastaiModel, show_metrics
from pycovid19xray.tests import DATA_DIR

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import fastai.vision as fv
from PIL import Image
import numpy as np

fastaimodel = FastaiModel(modelname="rs50-dataaug-covidxv3",
                          bs=64)

data, test = fastaimodel.create_covidx_databunch()
test_data = test.databunch()

# show a grid
plt.close("all")
data.show_batch(rows=4, figsize=(12,9))
fastaimodel.save_fig("covidx_grid.jpg")

plt.close("all")
data.show_batch(rows=4, figsize=(12,9), ds_type=fv.DatasetType.Valid)
fastaimodel.save_fig("covidx_grid_valid.jpg")

plt.close("all")
test_data.show_batch(rows=3, figsize=(12,9))
fastaimodel.save_fig("covidx_grid_test.jpg")

#######
# model
# w = torch.cuda.FloatTensor([1., 1., 6.])
# learn = fastaimodel.create_learner(data, loss_func=torch.nn.CrossEntropyLoss(weight=w))
learn = fastaimodel.create_learner(data, loss_func=torch.nn.CrossEntropyLoss())
learn.lr_find()
plt.close("all")
learn.recorder.plot()
fastaimodel.save_fig("train_resnet_covidx_find_lr.jpg")

lr = 1e-2
learn.fit_one_cycle(5, slice(lr))
fastaimodel.save_learner(learn, 'stage-1')

# eval on evaluation dataset
interp = fv.ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
fastaimodel.save_fig("val_confmat.jpg")

# show metrics
show_metrics(learn, test)

# unfreeze and fine tune
learn.unfreeze()
learn.lr_find()
plt.close("all")
learn.recorder.plot()
fastaimodel.save_fig("rain_resnet_covidx_find_lr_unfreeze.jpg")

learn.fit_one_cycle(15, slice(1e-5, 1e-4))
fastaimodel.save_learner(learn, 'stage-2')

# eval on evaluation dataset
interp = fv.ClassificationInterpretation.from_learner(learn)
plt.close("all")
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
fastaimodel.save_fig("eval_confmat_2.jpg")

show_metrics(learn, test)

