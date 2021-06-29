from typing import Union, Tuple

import torch
import torch.nn as nn
import kornia.augmentation.functional as kaf
import numpy as np
import random
import math


class AdaptiveDiscriminatorAugmentation(nn.Module):
    """
    This class implements adaptive discriminator augmentation proposed in:
    https://arxiv.org/pdf/2006.06676.pdf
    The adaptive discriminator augmentation model wraps a given discriminator network.
    """

    def __init__(self, discriminator: Union[nn.Module, nn.DataParallel], r_target: float = 0.6,
                 p_step: float = 5e-03, r_update: int = 8, p_max: float = 0.8) -> None:
        """
        Constructor method
        :param discriminator: (Union[nn.Module, nn.DataParallel]) Discriminator network
        :param r_target: (float) Target value for r
        :param p_step: (float) Step size of p
        :param r_update: (int) Update frequency of r
        :param p_max: (float) Global max value of p
        """
        # Call super constructor
        super(AdaptiveDiscriminatorAugmentation, self).__init__()
        # Save parameters
        self.discriminator = discriminator
        self.r_target = r_target
        self.p_step = p_step
        self.r_update = r_update
        self.p_max = p_max
        # Init augmentation variables
        self.r = []
        self.p = 0.05
        self.r_history = []
        # Init augmentation pipeline
        self.augmentation_pipeline = AugmentationPipeline()

    @torch.no_grad()
    def __calc_r(self, prediction_scalar: torch.Tensor, prediction_pixel_wise: torch.Tensor) -> float:
        """
        Method computes the overfitting heuristic r.
        :param prediction_scalar: (torch.Tensor) Scalar prediction [batch size, 1]
        :param prediction_pixel_wise: (torch.Tensor) Pixel-wise prediction [batch size, 1, height, width]
        :return: (float) Value of the overfitting heuristic r
        """
        return (0.5 * torch.mean(torch.sign(prediction_scalar))
                + 0.5 * torch.mean(torch.sign(prediction_pixel_wise.mean(dim=(-1, -2))))).item()

    def forward(self, images: torch.Tensor, is_real: bool = False,
                is_cut_mix: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        :param images: (torch.Tensor) Mini batch of images (real or fake) [batch size, channels, time steps, height, width]
        :param is_real: (bool) If true real images are utilized as the input
        :param is_cut_mix: (bool) If true cut mix is utilized and no augmentation is performed
        :return: (Tuple[torch.Tensor, torch.Tensor]) Scalar and pixel-wise real/fake prediction of the discriminator
        """
        # Case if cut mix is utilized
        if is_cut_mix:
            return self.discriminator(images)
        # Reshape images to [batch size, channels * time steps, height, width]
        original_shape = images.shape
        images = images.flatten(start_dim=1, end_dim=2)
        # Apply augmentations
        images: torch.Tensor = self.augmentation_pipeline(images, self.p)
        # Reshape images again to original shape
        images = images.view(original_shape)
        # Discriminator prediction
        prediction_scalar, prediction_pixel_wise = self.discriminator(images)
        # If fake images are given compute overfitting heuristic
        if not is_real:
            self.r.append(self.__calc_r(prediction_scalar=prediction_scalar.detach(),
                                        prediction_pixel_wise=prediction_pixel_wise.detach()))
        # Update p
        if len(self.r) >= self.r_update:
            # Calc r over the last epochs
            r = np.mean(self.r)
            # If r above target value increment p else reduce
            if r > self.r_target:
                self.p += self.p_step
            else:
                self.p -= self.p_step
            # Check if p is negative
            self.p = self.p if self.p >= 0. else 0.
            # Check if p is larger than 1
            self.p = self.p if self.p < self.p_max else self.p_max
            # Reset r
            self.r = []
            # Save current r in history
            self.r_history.append(r)
        return prediction_scalar, prediction_pixel_wise


class AugmentationPipeline(nn.Module):
    """
    This class implement the differentiable augmentation pipeline for ADA.
    """

    def __init__(self) -> None:
        # Call super constructor
        super(AugmentationPipeline, self).__init__()

    def forward(self, images: torch.Tensor, p: float) -> torch.Tensor:
        """
        Forward pass applies augmentation to mini-batch of given images
        :param images: (torch.Tensor) Mini-batch images [batch size, channels, height, width]
        :param p: (float) Probability of augmentation to be applied
        :return: (torch.Tensor) Augmented images [batch size, channels, height, width]
        """
        # Perform vertical flip
        images_flipped = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_flipped) > 0:
            images[images_flipped] = images[images_flipped].flip(dims=(-1,))
        # Perform rotation
        images_rotated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_rotated) > 0:
            angle = random.choice([torch.tensor(0.), torch.tensor(-90.), torch.tensor(90.), torch.tensor(180.)])
            angle = angle.to(images.to(images.device))
            images[images_rotated] = kaf.rotate(images[images_rotated],
                                                angle=angle)
        # Perform integer translation
        images_translated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_translated) > 0:
            images[images_translated] = integer_translation(images[images_translated])
        # Perform isotropic scaling
        images_scaling = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_scaling) > 0:
            images[images_scaling] = kaf.apply_affine(
                images[images_scaling],
                params={"angle": torch.zeros(len(images_scaling), device=images.device),
                        "translations": torch.zeros(len(images_scaling), 2, device=images.device),
                        "center": torch.ones(len(images_scaling), 2, device=images.device)
                                  * 0.5 * torch.tensor(images.shape[2:], device=images.device),
                        "scale": torch.ones(len(images_scaling), 2, device=images.device) *
                                 torch.from_numpy(
                                     np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2,
                                                         size=(len(images_scaling), 1))).float().to(images.device),
                        "sx": torch.zeros(len(images_scaling), device=images.device),
                        "sy": torch.zeros(len(images_scaling), device=images.device)},
                flags={"resample": torch.tensor(1, device=images.device),
                       "padding_mode": torch.tensor(2, device=images.device),
                       "align_corners": torch.tensor(True, device=images.device)})
        # Perform rotation
        images_rotated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= (1 - math.sqrt(1 - p)))
                          if value == True]
        if len(images_rotated) > 0:
            images[images_rotated] = kaf.apply_affine(
                images[images_rotated],
                params={"angle": torch.from_numpy(
                    np.random.uniform(low=-180, high=180, size=len(images_rotated))).to(images.device),
                        "translations": torch.zeros(len(images_rotated), 2, device=images.device),
                        "center": torch.ones(len(images_rotated), 2, device=images.device)
                                  * 0.5 * torch.tensor(images.shape[2:], device=images.device),
                        "scale": torch.ones(len(images_rotated), 2, device=images.device),
                        "sx": torch.zeros(len(images_rotated), device=images.device),
                        "sy": torch.zeros(len(images_rotated), device=images.device)},
                flags={"resample": torch.tensor(1, device=images.device),
                       "padding_mode": torch.tensor(2, device=images.device),
                       "align_corners": torch.tensor(True, device=images.device)})
        # Perform anisotropic scaling
        images_scaling = [index for index, value in enumerate(torch.rand(images.shape[0]) <= p) if value == True]
        if len(images_scaling) > 0:
            images[images_scaling] = kaf.apply_affine(
                images[images_scaling],
                params={"angle": torch.zeros(len(images_scaling), device=images.device),
                        "translations": torch.zeros(len(images_scaling), 2, device=images.device),
                        "center": torch.ones(len(images_scaling), 2, device=images.device)
                                  * 0.5 * torch.tensor(images.shape[2:], device=images.device),
                        "scale": torch.ones(len(images_scaling), 2, device=images.device) *
                                 torch.from_numpy(
                                     np.random.lognormal(mean=0, sigma=(0.2 * math.log(2)) ** 2,
                                                         size=(len(images_scaling), 2))).float().to(images.device),
                        "sx": torch.zeros(len(images_scaling), device=images.device),
                        "sy": torch.zeros(len(images_scaling), device=images.device)},
                flags={"resample": torch.tensor(1, device=images.device),
                       "padding_mode": torch.tensor(2, device=images.device),
                       "align_corners": torch.tensor(True, device=images.device)})
        # Perform rotation
        images_rotated = [index for index, value in enumerate(torch.rand(images.shape[0]) <= (1 - math.sqrt(1 - p)))
                          if value == True]
        if len(images_rotated) > 0:
            images[images_rotated] = kaf.apply_affine(
                images[images_rotated],
                params={"angle": torch.from_numpy(
                    np.random.uniform(low=-180, high=180, size=len(images_rotated))).to(images.device),
                        "translations": torch.zeros(len(images_rotated), 2, device=images.device),
                        "center": torch.ones(len(images_rotated), 2, device=images.device)
                                  * 0.5 * torch.tensor(images.shape[2:], device=images.device),
                        "scale": torch.ones(len(images_rotated), 2, device=images.device),
                        "sx": torch.zeros(len(images_rotated), device=images.device),
                        "sy": torch.zeros(len(images_rotated), device=images.device)},
                flags={"resample": torch.tensor(1, device=images.device),
                       "padding_mode": torch.tensor(2, device=images.device),
                       "align_corners": torch.tensor(True, device=images.device)})
        return images


def integer_translation(images: torch.Tensor) -> torch.Tensor:
    """
    Function implements integer translation augmentation
    :param images: (torch.Tensor) Input images
    :return: (torch.Tensor) Augmented images
    """
    # Get translation index
    translation_index = (int(images.shape[-2] * random.uniform(-0.125, 0.125)),
                         int(images.shape[-1] * random.uniform(-0.125, 0.125)))
    # Apply translation
    return torch.roll(images, shifts=translation_index, dims=(-2, -1))
