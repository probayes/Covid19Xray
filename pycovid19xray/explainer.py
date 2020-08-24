from pathlib import Path
from PIL import Image
import pickle
import shutil
import cv2
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from captum.attr import (IntegratedGradients,
                         NoiseTunnel,
                         Occlusion,
                         Saliency,
                         GuidedBackprop,
                         LayerGradCam,
                         LayerAttribution,
                         GradientShap,
                         GuidedGradCam)
from captum.attr import visualization as viz

LOGGER = logging.getLogger(__name__)


class Explainer():

    def __init__(self, model,
                 layer, # layer on which guided gradcam is applied
                 modelname,
                 transform,
                 transform_normalize,
                 class_idx,
                 dataset=None,
                 grayscale_img=False,
                 cache_path=None):
        self.transform = transform
        self.transform_normalize = transform_normalize
        self.device = torch.device("cuda")
        self.layer = layer
        self.model = model.to(self.device)
        self.model = self.model.eval()
        self.modelname = modelname
        self.figure_path = Path(f"./figures/{self.modelname}_explainer")
        if not self.figure_path.exists():
            self.figure_path.mkdir()
        self.class_idx = class_idx

        # dataset
        self.dataset = dataset # class used to load transformed img if none
        # transformed img are loaded with default procedure (see open_img).

        # plots parameters
        self.default_cmap = plt.get_cmap("plasma")
        self.grayscale_img = grayscale_img

        # all methods
        self.methods = {
            "occlusion": (self.compute_occlusion,
                          self.draw_occlusion),
            "integrated gradient": (self.compute_integrated_gradients,
                                    self.draw_integrated_gradients),
            "integrated gradient with noise tunnel":
            (self.compute_integrated_gradients_noise_tunnel,
             self.draw_integrated_gradients_noise_tunnel),
             "Guided grad cam": (self.compute_guided_gradcam,
                                 self.draw_guided_gradcam),
            "Guided gradient shap": (self.compute_gradient_shap,
                                     self.draw_gradient_shap),
            "Saliency": (self.compute_saliency,
                         self.draw_saliency),
            "GradCam": (self.compute_gradcam,
                        self.draw_gradcam),
        }

        # cache
        if cache_path is not None:
            self.arraycache = ArrayCache(cache_path, modelname)
            methods = {}
            for (k, t) in self.methods.items():
                methods[k] = (self.arraycache(t[0], k), t[1])
            self.methods = methods

    def read_from_cache(self, methodname, filename):
        if (self.cache_path is not None):
            dump_path = self.cache_path / f"{filename}_{methodname}"
            array = pickle.load(str(dump_path))
            return array
        else:
            return None

    def dump_to_cache(self, array, methodname, filename):
        if (self.cache_path is not None):
            dump_path = self.cache_path / f"{filename}_{methodname}"
            pickle.dump(array, str(dump_path))

    def open_image(self, img_path):
        # read image with this default procedure
        img = Image.open(str(img_path)).convert("RGB")
        if not self.dataset is None:
           # using dataset load img to load transform image
            transformed_img = self.dataset.load_img(img_path, self.transform)
        else:
            transformed_img = self.transform(img)
        input = self.transform_normalize(transformed_img)
        input = input.unsqueeze(0)
        input = input.to(self.device)

        return img, transformed_img, input

    def compute_prediction(self, input):

        # compute prediction
        output = self.model(input)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        return prediction_score, pred_label_idx

    def compute_occlusion(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        # run occlusion
        occlusion = Occlusion(self.model)
        attributions_occ = occlusion.attribute(input,
                                               strides=(3, 8, 8),
                                               target=target,
                                               sliding_window_shapes=(3,20,20),
                                               baselines=0)
        attributions_occ = np.transpose(attributions_occ
                                        .squeeze().cpu()
                                        .detach().numpy(), (1,2,0))
        return attributions_occ

    def draw_occlusion(self, toplot_img, attributions_occ, fig, ax):
        _ = viz.visualize_image_attr(
            attributions_occ,
            toplot_img,
            'blended_heat_map',
            'positive',
            plt_fig_axis=(fig, ax),
            show_colorbar=False,
            cmap=self.default_cmap,
            outlier_perc=2,
            use_pyplot=False)
        ax.title.set_text("occlusion")

    def compute_guided_gradcam(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        # guided grad cam
        input.requires_grad=True
        gradcam = GuidedGradCam(self.model, self.layer)
        attributions_gradcam = gradcam.attribute(input, target)
        attributions_gradcam = np.transpose(
            attributions_gradcam.squeeze()
            .cpu().detach().numpy(), (1,2,0))

        return attributions_gradcam

    def draw_guided_gradcam(self, toplot_img, attributions_gradcam, fig, ax):
        toplot_gcam = deprocess_image(attributions_gradcam)
        # toplot_cam = merge_cam_on_image(toplot_img, cam)
        if self.grayscale_img:
            gray_toplot_gcam = cv2.cvtColor(toplot_gcam, cv2.COLOR_BGR2GRAY)
            ax.imshow(gray_toplot_gcam, cmap="gray")
        else:
            ax.imshow(toplot_gcam)
        ax.title.set_text("Guided gradcam")

    def compute_gradcam(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        # grad cam
        input.requires_grad=True
        gradcam = LayerGradCam(self.model, self.layer)
        attr = gradcam.attribute(input, target)
        cam = (attr.squeeze()
               .cpu().detach().numpy())

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def draw_gradcam(self, toplot_img, cam, fig, ax):
        toplot_cam = merge_cam_on_image(toplot_img, cam)
        # toplot_cam = merge_cam_on_image(toplot_img, cam)
        ax.imshow(toplot_cam)
        ax.title.set_text("GradCam")

    def compute_integrated_gradients(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        integrated_gradients = IntegratedGradients(self.model)
        attributions_ig = integrated_gradients.attribute(input,
                                                         target=target,
                                                         n_steps=200)
        attributions_ig = np.transpose(attributions_ig
                                       .squeeze().cpu()
                                       .detach().numpy(), (1,2,0))
        return attributions_ig

    def draw_integrated_gradients(self, toplot_img, attributions_ig, fig, ax):
        _ = viz.visualize_image_attr(
            attributions_ig,
            toplot_img,
            'blended_heat_map',
            'positive',
            plt_fig_axis=(fig, ax),
            cmap=self.default_cmap,
            show_colorbar=False,
            use_pyplot=False,
            outlier_perc=2)
        ax.title.set_text("Integrated gradient")

    def compute_integrated_gradients_noise_tunnel(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        integrated_gradients = IntegratedGradients(self.model)
        # attributions_ig = integrated_gradients.attribute(input,
        #                                                  target=target,
        #                                                  n_steps=200)
        noise_tunnel = NoiseTunnel(integrated_gradients)
        attributions_ig_nt = noise_tunnel.attribute(input, n_samples=10,
                                                    nt_type='smoothgrad',
                                                    internal_batch_size=8,
                                                    n_steps=200,
                                                    target=target)
        attributions_ig_nt = np.transpose(attributions_ig_nt
                                       .squeeze().cpu()
                                       .detach().numpy(), (1,2,0))
        return attributions_ig_nt

    def draw_integrated_gradients_noise_tunnel(self, toplot_img,
                                               attributions_ig_nt, fig, ax):
        _ = viz.visualize_image_attr(
            attributions_ig_nt,
            toplot_img,
            'blended_heat_map',
            'positive',
            plt_fig_axis=(fig, ax),
            show_colorbar=False,
            cmap=self.default_cmap,
            use_pyplot=False,
            outlier_perc=2)
        ax.title.set_text("Integrated gradient with noise tunnel")

    def compute_gradient_shap(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        rand_img_dist = torch.cat([input * 0, input * 1])

        gradient_shap = GradientShap(self.model)
        attributions_gs = gradient_shap.attribute(input,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=target)
        attributions_gs = np.transpose(attributions_gs
                                       .squeeze().cpu()
                                       .detach().numpy(), (1,2,0))
        return attributions_gs

    def draw_gradient_shap(self, toplot_img,
                           attributions_gs, fig, ax):
        _ = viz.visualize_image_attr(
            attributions_gs,
            toplot_img,
            'blended_heat_map',
            'absolute_value',
            plt_fig_axis=(fig, ax),
            cmap=self.default_cmap,
            show_colorbar=False,
            use_pyplot=False,
            outlier_perc=2)
        ax.title.set_text("Gradient shap")

    def compute_saliency(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        gradient_saliency = Saliency(self.model)
        attributions_sa = gradient_saliency.attribute(input,
                                                      target=target)
        attributions_sa = np.transpose(attributions_sa
                                       .squeeze().cpu()
                                       .detach().numpy(), (1,2,0))
        return attributions_sa

    def compute_saliency_noise_tunnel(self, img_path, target):

        # open image
        img, transformed_img, input = self.open_image(img_path)

        gradient_saliency = Saliency(self.model)
        noise_tunnel = NoiseTunnel(gradient_saliency)
        attributions_sa_nt = noise_tunnel.attribute(input, n_samples=10,
                                                    nt_type='smoothgrad',
                                                    # internal_batch_size=8,
                                                    target=target)
        attributions_sa_nt = np.transpose(attributions_sa_nt
                                       .squeeze().cpu()
                                       .detach().numpy(), (1,2,0))
        return attributions_sa_nt

    def draw_saliency(self, toplot_img,
                      attributions_sa, fig, ax):
        _ = viz.visualize_image_attr(
            attributions_sa,
            toplot_img,
            'blended_heat_map',
            'absolute_value',
            plt_fig_axis=(fig, ax),
            cmap=self.default_cmap,
            show_colorbar=False,
            use_pyplot=False,
            outlier_perc=2)
        ax.title.set_text("Gradient saliency")

    def plot_explanations(self, img_path, true_class, blurring=False,
                          methods_name="all", plot_size=(2,4)):

        filename = get_filename(img_path)
        # open image
        img, transformed_img, input = self.open_image(img_path)

        if (methods_name == "all"):
            methods = self.methods
        else:
            methods = {k: v for (k, v) in methods.items() if k in methods_name}

        # get prediction
        prediction_score, pred_label_idx = self.compute_prediction(input)
        pred_class = self.class_idx[pred_label_idx.cpu().numpy()[0,0]]
        LOGGER.info(f"Prediction: {pred_class}")

        # run explanations methods
        results = dict()
        for (k, fs) in methods.items():
            LOGGER.info(f"Run: {k}")
            results[k] = fs[0](img_path, pred_label_idx)

        # plot grid
        plt.close("all")
        toplot_img = np.transpose(transformed_img
                                  .cpu()
                                  .detach().numpy(), (1,2,0))
        fig, axs = plt.subplots(plot_size[0], plot_size[1],
                                figsize=(15, 8),
                                dpi=200,
                                sharex=False, sharey=False)
        # original image
        axs[0,0].imshow(toplot_img)
        axs[0,0].title.set_text("Image")
        for ((k, fs), ax) in zip(methods.items(), axs.flatten()[1:]):
            LOGGER.info(f"Draw: {k}")
            att = results[k]
            if blurring:
                att = cv2.GaussianBlur(att, (5,5), 0)
            fs[1](toplot_img, att, fig, ax)
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(self.figure_path/f"{true_class}_{pred_class}_{filename}.jpg",
                    bbox_inches="tight")

##########
## helpers

def get_filename(img_path):
    return ".".join(Path(img_path).name.split(".")[:-1])


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def merge_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def retrieve_array(cached_f):
    if cached_f.exists():
        return np.load(str(cached_f))
    else:
        return None


def cache_array(array, cached_f):
    np.save(str(cached_f), array)


class ArrayCache():

    def __init__(self, cache_dir, modelname):
        self.cache_dir = Path(cache_dir) / modelname
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()

    def clear_cache(self):
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir()

    def __call__(self, func, methodname):
        def function_wrapper(f, target, force_compute=False):
            filename = get_filename(f)
            cached_f = self.cache_dir / f"{filename}_{methodname}_{target}.npy"
            array = retrieve_array(cached_f)
            if (array is None) or force_compute:
                array = func(f, target)
                cache_array(array, cached_f)
            return array
        return function_wrapper
