# Requirement

Python 3 is required to run the training script. The conda environment used to
run the training script is described in `../env.yml`.

# Training the resnet50

The model was train using the `fastai` library. You can reproduce the training
by running the script `./train_resnet_covidx.py`.

We considered data augmentation with the `fastai` function `get_transforms` with
the following parameters:
- max_rotate: 10 degrees
- max_zoom: 1.15 
- max_lighting: 0.1 (control brightness and contrast)
