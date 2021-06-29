from typing import Any, Dict, Union, Iterable, Optional, List

import torch
import torchvision
import torch.nn as nn
import os
import json
from datetime import datetime
import random
import numpy as np


class Logger(object):
    """
    Class to log different metrics.
    """

    def __init__(self,
                 experiment_path: str =
                 os.path.join(os.getcwd(), "experiments", datetime.now().strftime("%d_%m_%Y__%H_%M_%S")),
                 experiment_path_extension: str = "",
                 path_metrics: str = "metrics",
                 path_hyperparameters: str = "hyperparameters",
                 path_plots: str = "plots",
                 path_models: str = "models") -> None:
        """
        Constructor method
        :param path_metrics: (str) Path to folder in which all metrics are stored
        :param experiment_path_extension: (str) Extension to experiment folder
        :param path_hyperparameters: (str)  Path to folder in which all hyperparameters are stored
        :param path_plots: (str)  Path to folder in which all plots are stored
        :param path_models: (str)  Path to folder in which all models are stored
        """
        experiment_path = experiment_path + experiment_path_extension
        # Save parameters
        self.path_metrics = os.path.join(experiment_path, path_metrics)
        self.path_hyperparameters = os.path.join(experiment_path, path_hyperparameters)
        self.path_plots = os.path.join(experiment_path, path_plots)
        self.path_models = os.path.join(experiment_path, path_models)
        # Init folders
        os.makedirs(self.path_metrics, exist_ok=True)
        os.makedirs(self.path_hyperparameters, exist_ok=True)
        os.makedirs(self.path_plots, exist_ok=True)
        os.makedirs(self.path_models, exist_ok=True)
        # Init dicts to store the metrics and hyperparameters
        self.metrics = dict()
        self.temp_metrics = dict()
        self.hyperparameters = dict()

    def log_metric(self, metric_name: str, value: Any) -> None:
        """
        Method writes a given metric value into a dict including list for every metric.
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(float(value))
        else:
            self.metrics[metric_name] = [float(value)]

    def log_temp_metric(self, metric_name: str, value: Any) -> None:
        """
        Method writes a given metric value into a dict including temporal metrics.
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.temp_metrics:
            self.temp_metrics[metric_name].append(float(value))
        else:
            self.temp_metrics[metric_name] = [float(value)]

    def save_temp_metric(self, metric_name: Union[Iterable[str], str]) -> Dict[str, float]:
        """
        Method writes temporal metrics into the metrics dict by averaging.
        :param metric_name: (Union[Iterable[str], str]) One temporal metric name ore a list of names
        """
        averaged_temp_dict = dict()
        # Case if only one metric is given
        if isinstance(metric_name, str):
            # Calc average
            value = float(torch.tensor(self.temp_metrics[metric_name]).mean())
            # Save metric in log dict
            self.log_metric(metric_name=metric_name, value=value)
            # Put metric also in dict to be returned
            averaged_temp_dict[metric_name] = value
        # Case if multiple metrics are given
        else:
            for name in metric_name:
                # Calc average
                value = float(torch.tensor(self.temp_metrics[name]).mean())
                # Save metric in log dict
                self.log_metric(metric_name=name, value=value)
                # Put metric also in dict to be returned
                averaged_temp_dict[name] = value
        # Reset temp metrics
        self.temp_metrics = dict()
        # Save logs
        self.save()
        return averaged_temp_dict

    def log_hyperparameter(self, hyperparameter_name: str = None, value: Any = None,
                           hyperparameter_dict: Dict[str, Any] = None) -> None:
        """
        Method writes a given hyperparameter into a dict including all other hyperparameters.
        :param hyperparameter_name: (str) Name of the hyperparameter
        :param value: (Any) Value of the hyperparameter, must by convertible to str
        :param hyperparameter_dict: (Dict[str, Any]) Dict of multiple hyperparameter to be saved
        """
        # Case if name and value are given
        if (hyperparameter_name is not None) and (value is not None):
            if hyperparameter_name in self.hyperparameters:
                self.hyperparameters[hyperparameter_name].append(str(value))
            else:
                self.hyperparameters[hyperparameter_name] = [str(value)]
        # Case if dict of hyperparameters is given
        if hyperparameter_dict is not None:
            # Iterate over given dict, cast data and store in internal hyperparameters dict
            for key in hyperparameter_dict.keys():
                if key in self.hyperparameters.keys():
                    self.hyperparameters[key].append(str(hyperparameter_dict[key]))
                else:
                    self.hyperparameters[key] = [str(hyperparameter_dict[key])]

    def save_checkpoint(self, file_name: str, checkpoint_dict: Dict) -> None:
        """
        This method saves a given checkpoint.
        :param name: (str) File name with file format
        :param model: (Dict) Dict including all modules
        """
        torch.save(checkpoint_dict, os.path.join(self.path_models, file_name))

    def save_prediction(self, prediction: torch.Tensor, name: str) -> None:
        """
        This method saves the image predictions as an png image
        :param prediction: (torch.Tensor) Prediction of the shape [batch size, 2, time steps, height, width]
        :param name: (torch.Tensor) Name of the images without ending!
        """
        for batch_index in range(prediction.shape[0]):
            # Get images and normalize shape [time steps, 1, height, width]
            bf_images = prediction[batch_index, 0][:, None]
            # Make bf to rgb
            bf_images = bf_images.repeat_interleave(3, dim=1)
            if prediction.shape[1] > 1:
                gfp_images = prediction[batch_index, 1][:, None]
                # Make gfp to rgb only green shades
                gfp_images = gfp_images.repeat_interleave(3, dim=1)
                gfp_images[:, 0] = 0.0
                gfp_images[:, 2] = 0.0
            if prediction.shape[1] > 2:
                rfp_images = prediction[batch_index, 2][:, None]
                # Make rfp to rgb only red shades
                rfp_images = rfp_images.repeat_interleave(3, dim=1)
                rfp_images[:, 1] = 0.0
                rfp_images[:, 2] = 0.0
            # Save images
            torchvision.utils.save_image(tensor=bf_images,
                                         fp=os.path.join(self.path_plots, name + "_bf_{}.png".format(batch_index)),
                                         nrow=bf_images.shape[0], padding=0)
            if prediction.shape[1] > 1:
                torchvision.utils.save_image(tensor=gfp_images,
                                             fp=os.path.join(self.path_plots, name + "_gfp_{}.png".format(batch_index)),
                                             nrow=gfp_images.shape[0], padding=0)
            if prediction.shape[1] > 2:
                torchvision.utils.save_image(tensor=rfp_images,
                                             fp=os.path.join(self.path_plots, name + "_rfp_{}.png".format(batch_index)),
                                             nrow=gfp_images.shape[0], padding=0)

    def save(self) -> None:
        """
        Method saves all current logs (metrics and hyperparameters). Plots are saved directly.
        """
        # Save dict of hyperparameter as json file
        with open(os.path.join(self.path_hyperparameters, 'hyperparameter.txt'), 'w') as json_file:
            json.dump(self.hyperparameters, json_file)
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(self.path_metrics, '{}.pt'.format(metric_name)))


@torch.no_grad()
def exponential_moving_average(model_ema: Union[torch.nn.Module, nn.DataParallel],
                               model_train: Union[torch.nn.Module, nn.DataParallel], decay: float = 0.999) -> None:
    """
    Function apples one exponential moving average step to a given model to be accumulated and a given training model
    :param model_ema: (Union[torch.nn.Module, nn.DataParallel]) Model to be accumulated
    :param model_train: (Union[torch.nn.Module, nn.DataParallel]) Training model
    :param decay: (float) Decay factor
    """
    # Check types
    assert type(model_ema) is type(model_train), 'EMA can only be performed on networks of the same type!'
    # Get parameter dicts
    model_ema_dict = dict(model_ema.named_parameters())
    model_train_dict = dict(model_train.named_parameters())
    # Apply ema
    for key in model_ema_dict.keys():
        model_ema_dict[key].data.mul_(decay).add_(1 - decay, model_train_dict[key].data)


def random_permutation(n: int) -> torch.Tensor:
    """
    Function generates a random permutation without current permutation ([0, 1, 2, ...]).
    :param n: (int) Number of elements
    :return: (torch.Tensor) Permutation tensor
    """
    # Get random permutation
    permutation = torch.from_numpy(np.random.choice(range(n), size=n))
    # Check of default permutation is present
    if torch.equal(permutation, torch.arange(n)):
        permutation = torch.arange(start=n - 1, end=-1, step=-1)
    return permutation


def normalize_0_1_batch(input: torch.tensor) -> torch.tensor:
    """
    Normalize a given tensor batch wise to a range of [0, 1]
    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    """
    input_flatten = input.view(input.shape[0], -1)
    return ((input - torch.min(input_flatten, dim=1)[0][:, None, None, None, None]) / (
            torch.max(input_flatten, dim=1)[0][:, None, None, None, None] -
            torch.min(input_flatten, dim=1)[0][:, None, None, None, None])).clamp(min=1e-03)


def normalize_m1_1_batch(input: torch.tensor) -> torch.tensor:
    """
    Normalize a given tensor batch wise to a range of [-1, 1]
    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    """
    output = 2. * normalize_0_1_batch(input) - 1.
    return output


def get_noise(batch_size: int, latent_dimension, p_mixed_noise: float = 0.9, device: str = 'cuda') -> Union[
    torch.Tensor, List]:
    """
    Function returns an input noise for the style gan 2 generator.
    Iter a list of two noise vectors or one noise vector will be returned.
    :param batch_size: (int) Batch size to be used
    :param latent_dimension: (int) Latent dimensions to be utilized
    :param p_mixed_noise: (int) Probability that a mixed noise will be returned
    :param device: (str) Device to be utilized
    :return: List of noise tensors or single noise tensor
    """
    if (p_mixed_noise > 0) and (random.random() < p_mixed_noise):
        return list(torch.randn(2, batch_size, latent_dimension, dtype=torch.float32, device=device).unbind(0))
    else:
        return torch.randn(batch_size, latent_dimension, dtype=torch.float32, device=device)
