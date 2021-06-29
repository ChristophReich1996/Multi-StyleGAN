from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import copy
import numpy as np
import math
from scipy.linalg import sqrtm
import kornia

from . import misc


class IS(object):
    """
    This class implements the inception score.
    """

    def __init__(self, device: Union[str, torch.device] = "cuda",
                 data_parallel: bool = True, batch_size: int = 1,
                 data_samples: int = 5000, no_rfp: bool = False,
                 no_gfp: bool = False) -> None:
        """
        Constructor method
        :param device: (Union[str, torch.device]) Device to be utilized
        :param data_parallel: (bool) If true data parallel is used
        :param batch_size: (bool) Batch size to be utilized
        :param data_samples: (int) Number of real and fake sample to be utilized during evaluation
        :param no_rfp: (bool) If true no RFP channel is utilized
        :param no_rfp: (bool) If true no GFP channel is utilized
        """
        # Save parameters
        self.device = device
        self.data_parallel = data_parallel
        self.batch_size = batch_size
        self.data_samples = data_samples
        self.no_rfp = no_rfp
        self.no_gfp = no_gfp
        # Init inception net
        self.inception_net = torchvision.models.inception_v3(pretrained=True).cpu()

    def __preprocessing(self, input: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Output tensor
        """
        input = kornia.resize(input[:, :, 0], size=(299, 299), interpolation='bilinear', antialias=True)
        output = misc.normalize_m1_1_batch(input[:, :, None])[:, :, 0]
        return output

    @torch.no_grad()
    def __call__(self, generator: Union[nn.Module, nn.DataParallel],
                 **kwargs) -> Union[Tuple[float, float, float], Tuple[float, float]]:
        """
        Method to compute FID score
        :param generator: (Union[nn.Module, nn.DataParallel]) Generator network
        :param **kwargs: Not used
        :return: (Union[Tuple[float, float, float], Tuple[float, float]]) IS scores of bf, gfp and rfp (optional)
        """
        # Apply data parallel if utilized
        if self.data_parallel:
            inception_net = nn.DataParallel(copy.deepcopy(self.inception_net))
        else:
            inception_net = copy.deepcopy(self.inception_net)
        # Inception net to device
        inception_net = inception_net.to(self.device)
        # Ensure eval mode is present
        inception_net.eval()
        # Generator to device
        generator.to(self.device)
        # Generator into eval mode
        generator.eval()
        # Generate activation of real samples
        predictions_fake_bf = []
        predictions_fake_gfp = []
        if not self.no_rfp:
            predictions_fake_rfp = []
        for _ in range(math.ceil(self.data_samples / self.batch_size)):
            # Generate noise input
            noise_input = misc.get_noise(
                batch_size=self.batch_size,
                latent_dimension=generator.module.latent_dimensions
                if isinstance(generator, nn.DataParallel) else generator.latent_dimensions,
                p_mixed_noise=0.0,
                device=self.device
            )
            # Predict fake images
            fake_images = generator(input=noise_input)
            # Get random bf images [batch size, 3, height, width]
            fake_images_bf = fake_images[:, 0, torch.randint(0, fake_images.shape[2], (1,))].unsqueeze(
                dim=1).repeat_interleave(dim=1, repeats=3)
            if not self.no_gfp:
                # Get random gfp images [batch size, 3, height, width]
                fake_images_gfp = fake_images[:, 1, torch.randint(0, fake_images.shape[2], (1,))].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
            if not self.no_rfp:
                # Get random rfp images [batch size, 3, height, width]
                fake_images_rfp = fake_images[:, 2, torch.randint(0, fake_images.shape[2], (1,))].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
            # Make predictions for bf images
            prediction_fake_bf = inception_net(self.__preprocessing(fake_images_bf)) \
                .softmax(dim=1).cpu().unbind(dim=0)
            # Save bf activation
            predictions_fake_bf.extend(prediction_fake_bf)
            if not self.no_gfp:
                # Make predictions for gfp images
                prediction_fake_gfp = inception_net(self.__preprocessing(fake_images_gfp)) \
                    .softmax(dim=1).cpu().unbind(dim=0)
                # Save bf activation
                predictions_fake_gfp.extend(prediction_fake_gfp)
            # Make predictions for rfp images
            if not self.no_rfp:
                prediction_fake_rfp = inception_net(self.__preprocessing(fake_images_rfp)) \
                    .softmax(dim=1).cpu().unbind(dim=0)
                # Save bf activation
                predictions_fake_rfp.extend(prediction_fake_rfp)
        # Remove to many activations
        predictions_fake_bf = torch.stack(predictions_fake_bf[:self.data_samples], dim=0)
        if not self.no_gfp:
            predictions_fake_gfp = torch.stack(predictions_fake_gfp[:self.data_samples], dim=0)
        if not self.no_rfp:
            predictions_fake_rfp = torch.stack(predictions_fake_rfp[:self.data_samples], dim=0)
        # Calc inception score
        p_y_bf = predictions_fake_bf.mean(dim=0, keepdim=True)
        if not self.no_gfp:
            p_y_gfp = predictions_fake_gfp.mean(dim=0, keepdim=True)
        if not self.no_rfp:
            p_y_rfp = predictions_fake_rfp.mean(dim=0, keepdim=True)
        # Calc kl divergence
        kl_bf = torch.sum(predictions_fake_bf * torch.log(predictions_fake_bf / p_y_bf), dim=-1)
        if not self.no_gfp:
            kl_gfp = torch.sum(predictions_fake_gfp * torch.log(predictions_fake_gfp / p_y_gfp), dim=-1)
        if not self.no_rfp:
            kl_rfp = torch.sum(predictions_fake_rfp * torch.log(predictions_fake_rfp / p_y_rfp), dim=-1)
        # Calc inception score
        inception_score_bf = kl_bf.mean().exp()
        if not self.no_gfp:
            inception_score_gfp = kl_gfp.mean().exp()
        if not self.no_rfp:
            inception_score_rfp = kl_rfp.mean().exp()
        # Model to cpu
        inception_net.cpu()
        # Remove model
        del inception_net
        # Empty cuda cache
        torch.cuda.empty_cache()
        if not self.no_gfp:
            return inception_score_bf.item(), inception_score_gfp.item()
        if not self.no_rfp:
            return inception_score_bf.item(), inception_score_gfp.item(), inception_score_rfp.item()
        return inception_score_bf.item()


class FID(object):
    """
    This class implements the Frechet inception distance.
    """

    def __init__(self, device: Union[str, torch.device] = "cuda",
                 data_parallel: bool = True, batch_size: int = 1,
                 data_samples: int = 5000, no_rfp: bool = False,
                 no_gfp: bool = False) -> None:
        """
        Constructor method
        :param device: (Union[str, torch.device]) Device to be utilized
        :param data_parallel: (bool) If true data parallel is used
        :param batch_size: (bool) Batch size to be utilized
        :param data_samples: (int) Number of real and fake sample to be utilized during evaluation
        :param no_rfp: (bool) If true no RFP channel is utilized
        :param no_rfp: (bool) If true no GFP channel is utilized
        """
        # Save parameters
        self.device = device
        self.data_parallel = data_parallel
        self.batch_size = batch_size
        self.data_samples = data_samples
        self.no_rfp = no_rfp
        self.no_gfp = no_gfp
        # Init inception net
        self.model = InceptionNetworkFID().cpu()
        # Init activations
        self.activations_real_bf: np.ndarray = None
        if not self.no_gfp:
            self.activations_real_gfp: np.ndarray = None
        if not self.no_rfp:
            self.activations_real_rfp: np.ndarray = None

    @staticmethod
    def _calc_fid(real_activations: np.ndarray, fake_activations: np.ndarray):
        """
        Method to compute fid score
        :param real_activations: (np.ndarray) Real activation of the shape [samples, 2048]
        :param fake_activations: (np.ndarray) Fake activation of the shape [samples, 2048]
        :return: (float) FID score
        """
        # Calc statistics of real activations
        real_mu = np.mean(real_activations, axis=0)
        real_cov = np.cov(real_activations, rowvar=False)
        # Calc statistics of fake activations
        fake_mu = np.mean(fake_activations, axis=0)
        fake_cov = np.cov(fake_activations, rowvar=False)
        # Check that mu and cov arrays of real and fake have the same shapes
        assert real_mu.shape == fake_mu.shape
        assert real_cov.shape == fake_cov.shape
        # Calc diff of mu real and fake
        diff = real_mu - fake_mu
        # Square diff
        diff_squared = diff @ diff
        # Calc cov mean of fake and real cov
        cov_mean, _ = sqrtm(real_cov @ fake_cov, disp=False)
        # Remove imaginary path of cov mean
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        # Calc FID
        fid = diff_squared + np.trace(real_cov) + np.trace(fake_cov) - 2 * np.trace(cov_mean)
        return fid

    @torch.no_grad()
    def __call__(self, generator: Union[nn.Module, nn.DataParallel],
                 dataset: DataLoader) -> Union[Tuple[float, float, float], Tuple[float, float]]:
        """
        Method to compute FID score
        :param generator: (Union[nn.Module, nn.DataParallel]) Generator network
        :return: (Union[Tuple[float, float, float], Tuple[float, float]]) FID scores of bf, gfp and rfp (optional)
        """
        # Apply data parallel if utilized
        if self.data_parallel:
            inception_net = nn.DataParallel(copy.deepcopy(self.model))
        else:
            inception_net = copy.deepcopy(self.model)
        # Inception net to device
        inception_net = inception_net.to(self.device)
        # Ensure eval mode is present
        inception_net.eval()
        # Generate activation of real samples if not already computed!
        if self.activations_real_bf is None:
            activations_real_bf = []
            if not self.no_gfp:
                activations_real_gfp = []
            if not self.no_rfp:
                activations_real_rfp = []
            for real_images in dataset:
                # Get random bf images [batch size, 3, height, width]
                real_images_bf = real_images[:, 0, torch.randint(0, real_images.shape[2], (1,))].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
                if not self.no_gfp:
                    # Get random gfp images [batch size, 3, height, width]
                    real_images_gfp = real_images[:, 1, torch.randint(0, real_images.shape[2], (1,))].unsqueeze(
                        dim=1).repeat_interleave(dim=1, repeats=3)
                if not self.no_rfp:
                    # Get random rfp images [batch size, 3, height, width]
                    real_images_rfp = real_images[:, 2, torch.randint(0, real_images.shape[2], (1,))].unsqueeze(
                        dim=1).repeat_interleave(dim=1, repeats=3)
                # Make predictions for bf images
                activation_real_bf = inception_net(misc.normalize_m1_1_batch(real_images_bf)[:, :, 0]).cpu().unbind(
                    dim=0)
                # Save bf activation
                activations_real_bf.extend(activation_real_bf)
                if not self.no_gfp:
                    # Make predictions for gfp images
                    activation_real_gfp = inception_net(
                        misc.normalize_m1_1_batch(real_images_gfp)[:, :, 0]).cpu().unbind(
                        dim=0)
                    # Save bf activation
                    activations_real_gfp.extend(activation_real_gfp)
                if not self.no_rfp:
                    # Make predictions for rfp images
                    activation_real_rfp = inception_net(
                        misc.normalize_m1_1_batch(real_images_rfp)[:, :, 0]).cpu().unbind(dim=0)
                    # Save bf activation
                    activations_real_rfp.extend(activation_real_rfp)
                # Check if number of samples is reached
                if len(activations_real_bf) >= self.data_samples:
                    # Remove to many activations
                    self.activations_real_bf: np.ndarray = torch.stack(activations_real_bf[:self.data_samples],
                                                                       dim=0).numpy()
                    if not self.no_gfp:
                        self.activations_real_gfp: np.ndarray = torch.stack(activations_real_gfp[:self.data_samples],
                                                                            dim=0).numpy()
                    if not self.no_rfp:
                        self.activations_real_rfp: np.ndarray = torch.stack(activations_real_rfp[:self.data_samples],
                                                                            dim=0).numpy()
                    # Break for loop
                    break
        # Generator to device
        generator.to(self.device)
        # Generator into eval mode
        generator.eval()
        # Generate activation of real samples
        activations_fake_bf = []
        activations_fake_gfp = []
        if not self.no_rfp:
            activations_fake_rfp = []
        for _ in range(math.ceil(self.data_samples / self.batch_size)):
            # Generate noise input
            noise_input = misc.get_noise(
                batch_size=self.batch_size,
                latent_dimension=generator.module.latent_dimensions
                if isinstance(generator, nn.DataParallel) else generator.latent_dimensions,
                p_mixed_noise=0.0,
                device=self.device
            )
            # Predict fake images
            fake_images = generator(input=noise_input)
            # Get random bf images [batch size, 3, height, width]
            fake_images_bf = fake_images[:, 0, torch.randint(0, fake_images.shape[2], (1,))].unsqueeze(
                dim=1).repeat_interleave(dim=1, repeats=3)
            if not self.no_gfp:
                # Get random gfp images [batch size, 3, height, width]
                fake_images_gfp = fake_images[:, 1, torch.randint(0, fake_images.shape[2], (1,))].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
            if not self.no_rfp:
                # Get random rfp images [batch size, 3, height, width]
                fake_images_rfp = fake_images[:, 2, torch.randint(0, fake_images.shape[2], (1,))].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
            # Make predictions for bf images
            activation_fake_bf = inception_net(misc.normalize_m1_1_batch(fake_images_bf)[:, :, 0]).cpu().unbind(dim=0)
            # Save bf activation
            activations_fake_bf.extend(activation_fake_bf)
            if not self.no_gfp:
                # Make predictions for gfp images
                activation_fake_gfp = inception_net(misc.normalize_m1_1_batch(fake_images_gfp)[:, :, 0]).cpu().unbind(
                    dim=0)
                # Save bf activation
                activations_fake_gfp.extend(activation_fake_gfp)
            if not self.no_rfp:
                # Make predictions for rfp images
                activation_fake_rfp = inception_net(misc.normalize_m1_1_batch(fake_images_rfp)[:, :, 0]).cpu().unbind(
                    dim=0)
                # Save bf activation
                activations_fake_rfp.extend(activation_fake_rfp)
        # Remove to many activations
        activations_fake_bf: np.ndarray = torch.stack(activations_fake_bf[:self.data_samples],
                                                      dim=0).numpy()
        if not self.no_gfp:
            activations_fake_gfp: np.ndarray = torch.stack(activations_fake_gfp[:self.data_samples],
                                                           dim=0).numpy()
        if not self.no_rfp:
            activations_fake_rfp: np.ndarray = torch.stack(activations_fake_rfp[:self.data_samples],
                                                           dim=0).numpy()
        # Model to cpu
        inception_net.cpu()
        # Remove model
        del inception_net
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Calc fid score for bf and gfp
        if not self.no_gfp:
            return self._calc_fid(self.activations_real_bf, activations_fake_bf), \
                   self._calc_fid(self.activations_real_gfp, activations_fake_gfp)
        if not self.no_rfp:
            return self._calc_fid(self.activations_real_bf, activations_fake_bf), \
                   self._calc_fid(self.activations_real_gfp, activations_fake_gfp), \
                   self._calc_fid(self.activations_real_rfp, activations_fake_rfp)
        return self._calc_fid(self.activations_real_bf, activations_fake_bf)


class FVD(object):
    """
    This class implements the Frechet video distance.
    https://openreview.net/pdf?id=rylgEULtdN
    """

    def __init__(self, device: Union[str, torch.device] = "cuda",
                 data_parallel: bool = True, batch_size: int = 1,
                 data_samples: int = 5000, no_rfp: bool = False,
                 no_gfp: bool = False,
                 network_path: str = "multi_stylegan/pretrained_i3d/rgb_imagenet.pt") -> None:
        """
        Constructor method
        :param device: (Union[str, torch.device]) Device to be utilized
        :param data_parallel: (bool) If true data parallel is used
        :param batch_size: (bool) Batch size to be utilized
        :param data_samples: (int) Number of real and fake sample to be utilized during evaluation
        :param no_rfp: (bool) If true no RFP channel is utilized
        :param no_rfp: (bool) If true no GFP channel is utilized
        :param network_path: (str) Path to trained 3D incrption net
        """
        # Save parameters
        self.device = device
        self.data_parallel = data_parallel
        self.batch_size = batch_size
        self.data_samples = data_samples
        self.no_rfp = no_rfp
        self.no_gfp = no_gfp
        # Init inception net
        self.model = InceptionI3d(num_classes=400, in_channels=3).cpu()
        self.model.load_state_dict(torch.load(network_path))
        self.model.VALID_ENDPOINTS = self.model.VALID_ENDPOINTS[:-2]
        # Init activations
        self.activations_real_bf: np.ndarray = None
        if not self.no_gfp:
            self.activations_real_gfp: np.ndarray = None
        if not self.no_rfp:
            self.activations_real_rfp: np.ndarray = None

    @staticmethod
    def _calc_fvd(real_activations: np.ndarray, fake_activations: np.ndarray):
        """
        Method to compute fvd score
        :param real_activations: (np.ndarray) Real activation of the shape [samples, 2048]
        :param fake_activations: (np.ndarray) Fake activation of the shape [samples, 2048]
        :return: (float) FID score
        """
        # Calc statistics of real activations
        real_mu = np.mean(real_activations, axis=0)
        real_cov = np.cov(real_activations, rowvar=False)
        # Calc statistics of fake activations
        fake_mu = np.mean(fake_activations, axis=0)
        fake_cov = np.cov(fake_activations, rowvar=False)
        # Check that mu and cov arrays of real and fake have the same shapes
        assert real_mu.shape == fake_mu.shape
        assert real_cov.shape == fake_cov.shape
        # Calc diff of mu real and fake
        diff = real_mu - fake_mu
        # Square diff
        diff_squared = diff @ diff
        # Calc cov mean of fake and real cov
        cov_mean, _ = sqrtm(real_cov @ fake_cov, disp=False)
        # Remove imaginary path of cov mean
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        # Calc FID
        fid = diff_squared + np.trace(real_cov) + np.trace(fake_cov) - 2 * np.trace(cov_mean)
        return fid

    @torch.no_grad()
    def __call__(self, generator: Union[nn.Module, nn.DataParallel], dataset: DataLoader) -> Union[
        float, Tuple[float, float], Tuple[float, float, float]]:
        """
        Method to compute FID score
        :param generator: (Union[nn.Module, nn.DataParallel]) Generator network
        :return: (Tuple[float, float, float]) FID scores for bf, gfp and rfp
        """
        # Apply data parallel if utilized
        if self.data_parallel:
            inception_net = nn.DataParallel(copy.deepcopy(self.model))
        else:
            inception_net = copy.deepcopy(self.model)
        # Inception net to device
        inception_net = inception_net.to(self.device)
        # Ensure eval mode is present
        inception_net.eval()
        # Generate activation of real samples if not already computed!
        if self.activations_real_bf is None:
            activations_real_bf = []
            if not self.no_gfp:
                activations_real_gfp = []
            if not self.no_rfp:
                activations_real_rfp = []
            for real_images in dataset:
                # Get random bf images [batch size, 3, time steps, height, width]
                real_images_bf = real_images[:, 0].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
                if not self.no_gfp:
                    # Get random gfp images [batch size, 3, time steps, height, width]
                    real_images_gfp = real_images[:, 1].unsqueeze(
                        dim=1).repeat_interleave(dim=1, repeats=3)
                if not self.no_rfp:
                    # Get random rfp images [batch size, 3, time steps, height, width]
                    real_images_rfp = real_images[:, 2].unsqueeze(
                        dim=1).repeat_interleave(dim=1, repeats=3)
                # Make predictions for bf images
                activation_real_bf = inception_net(
                    misc.normalize_m1_1_batch(real_images_bf)).cpu().flatten(start_dim=1).unbind(dim=0)
                # Save bf activation
                activations_real_bf.extend(activation_real_bf)
                if not self.no_gfp:
                    # Make predictions for gfp images
                    activation_real_gfp = inception_net(
                        misc.normalize_m1_1_batch(real_images_gfp)).cpu().flatten(start_dim=1).unbind(dim=0)
                    # Save bf activation
                    activations_real_gfp.extend(activation_real_gfp)
                if not self.no_rfp:
                    # Make predictions for rfp images
                    activation_real_rfp = inception_net(
                        misc.normalize_m1_1_batch(real_images_rfp)).cpu().flatten(start_dim=1).unbind(dim=0)
                    # Save bf activation
                    activations_real_rfp.extend(activation_real_rfp)
                # Check if number of samples is reached
                if len(activations_real_bf) >= self.data_samples:
                    # Remove to many activations
                    self.activations_real_bf: np.ndarray = torch.stack(activations_real_bf[:self.data_samples],
                                                                       dim=0).numpy()
                    if not self.no_gfp:
                        self.activations_real_gfp: np.ndarray = torch.stack(activations_real_gfp[:self.data_samples],
                                                                            dim=0).numpy()
                    if not self.no_rfp:
                        self.activations_real_rfp: np.ndarray = torch.stack(activations_real_rfp[:self.data_samples],
                                                                            dim=0).numpy()
                    # Break for loop
                    break
        # Generator to device
        generator.to(self.device)
        # Generator into eval mode
        generator.eval()
        # Generate activation of real samples
        activations_fake_bf = []
        if not self.no_gfp:
            activations_fake_gfp = []
        if not self.no_rfp:
            activations_fake_rfp = []
        for _ in range(math.ceil(self.data_samples / self.batch_size)):
            # Generate noise input
            noise_input = misc.get_noise(
                batch_size=self.batch_size,
                latent_dimension=generator.module.latent_dimensions
                if isinstance(generator, nn.DataParallel) else generator.latent_dimensions,
                p_mixed_noise=0.0,
                device=self.device
            )
            # Predict fake images
            fake_images = generator(input=noise_input)
            # Get random bf images [batch size, 3, height, width]
            fake_images_bf = fake_images[:, 0].unsqueeze(
                dim=1).repeat_interleave(dim=1, repeats=3)
            if not self.no_gfp:
                # Get random gfp images [batch size, 3, height, width]
                fake_images_gfp = fake_images[:, 1].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
            if not self.no_rfp:
                # Get random rfp images [batch size, 3, height, width]
                fake_images_rfp = fake_images[:, 2].unsqueeze(
                    dim=1).repeat_interleave(dim=1, repeats=3)
            # Make predictions for bf images
            activation_fake_bf = inception_net(
                (misc.normalize_m1_1_batch(fake_images_bf))).cpu().flatten(start_dim=1).unbind(dim=0)
            # Save bf activation
            activations_fake_bf.extend(activation_fake_bf)
            if not self.no_gfp:
                # Make predictions for gfp images
                activation_fake_gfp = inception_net(
                    (misc.normalize_m1_1_batch(fake_images_gfp))).cpu().flatten(start_dim=1).unbind(dim=0)
                # Save bf activation
                activations_fake_gfp.extend(activation_fake_gfp)
            if not self.no_rfp:
                # Make predictions for gfp images
                activation_fake_rfp = inception_net(
                    (misc.normalize_m1_1_batch(fake_images_rfp))).cpu().flatten(start_dim=1).unbind(dim=0)
                # Save bf activation
                activations_fake_rfp.extend(activation_fake_rfp)
        # Remove to many activations
        activations_fake_bf: np.ndarray = torch.stack(activations_fake_bf[:self.data_samples],
                                                      dim=0).numpy()
        if not self.no_gfp:
            activations_fake_gfp: np.ndarray = torch.stack(activations_fake_gfp[:self.data_samples],
                                                           dim=0).numpy()
        if not self.no_rfp:
            activations_fake_rfp: np.ndarray = torch.stack(activations_fake_rfp[:self.data_samples],
                                                           dim=0).numpy()
        # Model to cpu
        inception_net.cpu()
        # Remove model
        del inception_net
        # Empty cuda cache
        torch.cuda.empty_cache()
        # Calc fid score for bf and gfp
        if not self.no_gfp:
            return self._calc_fvd(self.activations_real_bf, activations_fake_bf), \
                   self._calc_fvd(self.activations_real_gfp, activations_fake_gfp)
        if not self.no_rfp:
            return self._calc_fvd(self.activations_real_bf, activations_fake_bf), \
                   self._calc_fvd(self.activations_real_gfp, activations_fake_gfp), \
                   self._calc_fvd(self.activations_real_rfp, activations_fake_rfp)
        return self._calc_fvd(self.activations_real_bf, activations_fake_bf)


class InceptionNetworkFID(nn.Module):
    """
    This class implements a pre trained inception network for getting the output of layer 7c.
    """

    def __init__(self):
        # Call super constructor
        super(InceptionNetworkFID, self).__init__()
        # Init pre trained inception net
        self.inception_net = torchvision.models.inception_v3(pretrained=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param input: (torch.Tensor) Input tensor normalized to a range of [-1, 1]
        :return: (torch.Tensor) Rescaled intermediate output of layer 7c of the inception net
        """
        # Reshape input if needed
        if input.shape[-1] != 299 or input.shape[-2] != 299:
            input = kornia.resize(input, size=(299, 299), interpolation='bilinear', antialias=True)
            # input = nn.functional.interpolate(input, size=(299, 299), mode='bilinear', align_corners=False)
        # Forward pass of inception to get produce output
        x = self.inception_net.Conv2d_1a_3x3(input)
        x = self.inception_net.Conv2d_2a_3x3(x)
        x = self.inception_net.Conv2d_2b_3x3(x)
        x = self.inception_net.maxpool1(x)
        x = self.inception_net.Conv2d_3b_1x1(x)
        x = self.inception_net.Conv2d_4a_3x3(x)
        x = self.inception_net.maxpool2(x)
        x = self.inception_net.Mixed_5b(x)
        x = self.inception_net.Mixed_5c(x)
        x = self.inception_net.Mixed_5d(x)
        x = self.inception_net.Mixed_6a(x)
        x = self.inception_net.Mixed_6b(x)
        x = self.inception_net.Mixed_6c(x)
        x = self.inception_net.Mixed_6d(x)
        x = self.inception_net.Mixed_6e(x)
        x = self.inception_net.Mixed_7a(x)
        x = self.inception_net.Mixed_7b(x)
        x = self.inception_net.Mixed_7c(x)
        # Get intermediate output and downscale tensor to get a tensor of shape (batch size, 2024, 1, 1)
        output = F.adaptive_avg_pool2d(x, (1, 1))
        # Reshape output to shape (batch size, 2024)
        output = output.view(input.shape[0], 2048)
        return output


# i3d model implementation taken from: https://github.com/piergiaj/pytorch-i3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        batch_size, channels, time_steps = x.shape[:3]
        x = kornia.resize(x.flatten(start_dim=1, end_dim=2), size=(224, 224), interpolation='bilinear', antialias=True)
        x = x.reshape(batch_size, channels, time_steps, 224, 224)
        return self.extract_features(x)

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        output = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).flatten(start_dim=1)
        return output
