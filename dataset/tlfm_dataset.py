from typing import Optional, Tuple, Union, List

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import math

from dataset import utils


class TFLMDatasetGAN(Dataset):
    """
    This class implements the unsupervised TFLM dataset including data from a trapped yeast cell TFLM experiment for
    the generation task.
    """

    def __init__(self, path: str,
                 sequence_length: int = 3,
                 overlap: bool = True,
                 transformations: transforms.Compose = transforms.Compose(
                     [transforms.RandomHorizontalFlip(p=0.5)]),
                 z_position_indications: Tuple[str] = ("_000_", "_001_", "_002_"),
                 gfp_min: Union[float, int] = 150.0,
                 gfp_max: Union[float, int] = 2200.0,
                 rfp_min: Union[float, int] = 20.0,
                 rfp_max: Union[float, int] = 2000.0,
                 flip: bool = True,
                 positions: Optional[Tuple[str, ...]] = None,
                 no_rfp: bool = False,
                 no_gfp: bool = False) -> None:
        """
        Constructor method
        :param path: (str) Path to dataset
        :param sequence_length: (int) Length of sequence to be returned
        :param overlap: (bool) If true sequences can overlap
        :param transformations: (transforms.Compose) Transformations and augmentations to be applied
        :param z_position_indications: (Tuple[str]) String to indicate each z position
        :param gfp_min: (Union[float, int]) Minimal value assumed gfp value
        :param gfp_max: (Union[float, int]) Maximal value assumed gfp value
        :param rfp_min: (Union[float, int]) Minimal value assumed rfp value
        :param rfp_max: (Union[float, int]) Maximal value assumed rfp value
        :param flip: (bool) If true images are flipped vertically
        :param positions: (Optional[Tuple[str, ...]]) If given only positions which are given are loaded
        :param no_rfp: (bool) If true no rfp channel is utilized
        :param no_rfp: (bool) If true nogfp channel is utilized
        """
        # Save parameters
        self.transformations = transformations
        self.gfp_min = gfp_min
        self.gfp_max = gfp_max
        self.rfp_min = rfp_min
        self.rfp_max = rfp_max
        self.flip = flip
        self.no_rfp = no_rfp
        self.no_gfp = no_gfp
        # Load data sample paths
        self.paths_to_dataset_samples = []
        # Iterate over all position folders
        for position_folder in os.listdir(path=path):
            if (positions is None) or (position_folder in positions):
                # Check that current folder is really a folder
                if os.path.isdir(os.path.join(path, position_folder)):
                    # Load images all in folder
                    all_images = [os.path.join(path, position_folder, image_file) for image_file in
                                  os.listdir(os.path.join(path, position_folder)) if "tif" in image_file]
                    # Get all BF images
                    all_bf_images = [image_file for image_file in all_images if "-BF0_" in image_file]
                    # Get all GFP images
                    all_gfp_images = [image_file for image_file in all_images if "-GFP" in image_file]
                    # Get all RFP images
                    all_rfp_images = [image_file for image_file in all_images if "-RFP" in image_file]
                    # Convert list of images to list of z positions including images
                    bf_images = []
                    for z_position_indication in z_position_indications:
                        bf_images.append(
                            [image_file for image_file in all_bf_images if z_position_indication in image_file])
                        # Sort images by time steps and trap number
                        bf_images[-1].sort(key=lambda item:
                        item.split("-")[-1].split("_")[-1].replace(".tif", "") +
                        item.split("_")[-5])
                    gfp_images = []
                    for z_position_indication in z_position_indications:
                        gfp_images.append(
                            [image_file for image_file in all_gfp_images if z_position_indication in image_file])
                        # Sort images by time steps and trap number
                        gfp_images[-1].sort(key=lambda item:
                        item.split("-")[-1].split("_")[-1].replace(".tif", "") +
                        item.split("_")[-5])
                    rfp_images = []
                    for z_position_indication in z_position_indications:
                        rfp_images.append(
                            [image_file for image_file in all_rfp_images if z_position_indication in image_file])
                        # Sort images by time steps and trap number
                        rfp_images[-1].sort(key=lambda item:
                        item.split("-")[-1].split("_")[-1].replace(".tif", "") +
                        item.split("_")[-5])
                    # Construct image sequences
                    for z_position in range(len(z_position_indications)):
                        for index in range(0, len(bf_images[z_position]) - sequence_length + 1,
                                           1 if overlap else sequence_length):
                            if self._check_if_same_trap(bf_images[z_position][index:index + sequence_length]):
                                # Save paths
                                self.paths_to_dataset_samples.append(
                                    (tuple(bf_images[z_position][index:index + sequence_length]),
                                     tuple(gfp_images[z_position][index:index + sequence_length]),
                                     tuple(rfp_images[z_position][index:index + sequence_length])))

    def _check_if_same_trap(self, path_list: List[str]) -> bool:
        """
        Method checks of a sequence of images paths include the same trap.
        :param path_list: (List[str]) List of strings
        :return: (bool) If same trap true else false
        """
        traps = [path[path.find("trap"):path.find("trap") + 8] for path in path_list]
        return all(trap == traps[0] for trap in traps)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        :return: (int) Length of the dataset.
        """
        return len(self.paths_to_dataset_samples)

    def __getitem__(self, item: int) -> torch.Tensor:
        """
        Method returns one instance with the index item of the dataset.
        :param item: (int) Index of the dataset element to be returned
        :return: (torch.Tensor) Image sequence of n images
        """
        # Get paths
        path_bf_images, path_gfp_images, path_rfp_images = self.paths_to_dataset_samples[item]
        # Load bf images
        bf_images = []
        for path_bf_image in path_bf_images:
            image = cv2.imread(path_bf_image, -1).astype(np.float32)
            image = torch.from_numpy(image)
            bf_images.append(image)
        bf_images = torch.stack(bf_images, dim=0)
        # Load gfp images
        if not self.no_gfp:
            gfp_images = []
            for path_gfp_image in path_gfp_images:
                image = cv2.imread(path_gfp_image, -1).astype(np.float32)
                image = torch.from_numpy(image)
                gfp_images.append(image)
            gfp_images = torch.stack(gfp_images, dim=0)
        # Load rfp images
        if not self.no_rfp:
            rfp_images = []
            for path_rfp_image in path_rfp_images:
                image = cv2.imread(path_rfp_image, -1).astype(np.float32)
                image = torch.from_numpy(image)
                rfp_images.append(image)
            rfp_images = torch.stack(rfp_images, dim=0)
        if self.no_gfp:
            # Concat images
            images = torch.cat([bf_images], dim=0)
            # Perform transformations
            images = self.transformations(images)
            # Remove batch dimension
            images = images[0] if images.ndimension() == 4 else images
            # Reshape images to [1, sequence length, height, width]
            images = images.unsqueeze(dim=0)
        elif self.no_rfp:
            # Concat images
            images = torch.cat([bf_images, gfp_images], dim=0)
            # Perform transformations
            images: torch.Tensor = self.transformations(images)
            # Remove batch dimension
            images = images[0] if images.ndimension() == 4 else images
            # Reshape images to [2, sequence length, height, width]
            images = torch.stack(images.split(split_size=images.shape[0] // 2, dim=0), dim=0)
        else:
            # Concat images
            images = torch.cat([bf_images, gfp_images, rfp_images], dim=0)
            # Perform transformations
            images = self.transformations(images)
            # Remove batch dimension
            images = images[0] if images.ndimension() == 4 else images
            # Reshape images to [3, sequence length, height, width]
            images = torch.stack(images.split(split_size=images.shape[0] // 3, dim=0), dim=0)
        # Normalized bf images
        images[0] = utils.normalize_0_1(images[0])
        # Normalize gfp images
        if not self.no_gfp:
            # images[1] = utils.normalize_0_1(images[1])
            images[1] = ((images[1] - self.gfp_min).clamp(min=0.0) / self.gfp_max).clamp(max=1.0)
        # Normalize rfp images
        if not self.no_rfp:
            # images[2] = utils.normalize_0_1(images[2])
            images[2] = ((images[2] - self.rfp_min).clamp(min=0.0) / self.rfp_max).clamp(max=1.0)
        # Flip images if utilized
        images = images.flip(dims=(-2,)) if self.flip else images
        return images


class ElasticDeformation(nn.Module):
    """
    This module implements random elastic deformation.
    """

    def __init__(self, sample_mode: str = "bilinear", alpha: int = 80,
                 sigma: int = 16) -> None:
        """
        Constructor method
        :param sample_mode: (str) Resmapling mode
        :param alpha: (int) Scale factor of the deformation
        :param sigma: (int) Standard deviation of the gaussian kernel to be applied
        """
        # Call super constructor
        super(ElasticDeformation, self).__init__()
        # Save parameters
        self.sample_mode = sample_mode
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applies random elastic deformation
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Augmented output tensor
        """
        return elastic_deformation(img=input, sample_mode=self.sample_mode, alpha=self.alpha, sigma=self.sigma)


def elastic_deformation(img: torch.Tensor, sample_mode: str = "bilinear", alpha: int = 50,
                        sigma: int = 12) -> torch.Tensor:
    """
    Performs random elastic deformation to the given Tensor image
    :param img: (torch.Tensor) Input image
    :param sample_mode: (str) Resmapling mode
    :param alpha: (int) Scale factor of the deformation
    :param sigma: (int) Standard deviation of the gaussian kernel to be applied
    """
    # Get image shape
    height, width = img.shape[-2:]
    # Get kernel size
    kernel_size = (sigma * 4) + 1
    # Get mean of gaussian kernel
    mean = (kernel_size - 1) / 2.
    # Make gaussian kernel
    # https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    x_cord = torch.arange(kernel_size, device=img.device)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    gaussian_kernel = (1. / (2. * math.pi * sigma ** 2)) \
                      * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2. * sigma ** 2))
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)
    gaussian_kernel.requires_grad = False
    # Make random deformations in the range of [-1, 1]
    dx = (torch.rand((height, width), dtype=torch.float, device=img.device) * 2. - 1.).view(1, 1, height, width)
    dy = (torch.rand((height, width), dtype=torch.float, device=img.device) * 2. - 1.).view(1, 1, height, width)
    # Apply gaussian filter to deformations
    dx, dy = torch.nn.functional.conv2d(input=torch.cat([dx, dy], dim=0), weight=gaussian_kernel, stride=1,
                                        padding=kernel_size // 2).squeeze(dim=0) * alpha
    # Add deformations to coordinate grid
    grid = torch.stack(torch.meshgrid([torch.arange(height, dtype=torch.float, device=img.device),
                                       torch.arange(width, dtype=torch.float, device=img.device)]),
                       dim=-1).unsqueeze(dim=0).flip(dims=(-1,))
    grid[..., 0] += dx
    grid[..., 1] += dy
    # Convert grid to relative sampling location in the range of [-1, 1]
    grid[..., 0] = 2 * (grid[..., 0] - (height // 2)) / height
    grid[..., 1] = 2 * (grid[..., 1] - (width // 2)) / width
    # Resample image
    img_deformed = torch.nn.functional.grid_sample(input=img[None] if img.ndimension() == 3 else img,
                                                   grid=grid, mode=sample_mode, padding_mode='border',
                                                   align_corners=False)[0]
    return img_deformed
