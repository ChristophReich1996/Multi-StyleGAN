from typing import Tuple, Dict, Any, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from .op_static import FusedLeakyReLU, upfirdn2d

from . import equalized_layer


class Discriminator(nn.Module):
    """
    This class implements a 3D U-Net discriminator inspired by:
    https://arxiv.org/pdf/2002.12655.pdf
    """

    def __init__(self, config: Dict[str, Any], no_rfp: bool = False, no_gfp: bool = False) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Dict with network configurations
        :param no_rfp: (bool) If true no rfp channels if predicted
        :param no_gfp: (bool) If true no gfp channels if predicted
        """
        # Call super constructor
        super(Discriminator, self).__init__()
        # Get parameters from config dict
        encoder_channels: Tuple[Tuple[int, int], ...] = config["encoder_channels"]
        decoder_channels: Tuple[Tuple[int, int], ...] = config["decoder_channels"]
        self.fft: bool = config["fft"]
        # Init encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for index, encoder_channel in enumerate(encoder_channels):
            if index == 0:
                if no_gfp:
                    input_channels: int = 3
                elif no_rfp:
                    input_channels: int = 6
                else:
                    input_channels: int = 9
                if self.fft:
                    self.encoder_blocks.append(
                        ResNetBlock(in_channels=input_channels + (input_channels * 2),
                                    out_channels=encoder_channel[1]))
                else:
                    self.encoder_blocks.append(
                        ResNetBlock(in_channels=input_channels,
                                    out_channels=encoder_channel[1]))
            elif index == 2:
                self.encoder_blocks.append(
                    NonLocalBlock(in_channels=encoder_channel[0], out_channels=encoder_channel[1]))
            else:
                self.encoder_blocks.append(
                    ResNetBlock(in_channels=encoder_channel[0], out_channels=encoder_channel[1],
                                mini_batch_std_dev=index >= (len(encoder_channels) - 2)))
        # Init downscale convolutions
        self.downscale_convolutions = nn.ModuleList(
            [nn.Sequential(
                equalized_layer.EqualizedConv2d(in_channels=encoder_channel[1], out_channels=encoder_channel[1],
                                                kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)), Blur()) for
                encoder_channel in encoder_channels[:-1]])
        # Init classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            equalized_layer.EqualizedLinear(in_channels=encoder_channels[-1][-1], out_channels=128, bias=False),
            FusedLeakyReLU(channel=128),
            equalized_layer.EqualizedLinear(in_channels=128, out_channels=1, bias=False)
        )
        # Init decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for index, decoder_channel in enumerate(decoder_channels):
            if index == 1:
                self.decoder_blocks.append(
                    NonLocalBlock(in_channels=decoder_channel[0], out_channels=decoder_channel[1]))
            else:
                self.decoder_blocks.append(ResNetBlock(in_channels=decoder_channel[0], out_channels=decoder_channel[1]))
        # Init transposed convolutions
        self.transposed_convolutions = nn.ModuleList()
        for current_channel, past_channel, decoder_channel in zip(reversed(encoder_channels[1:]),
                                                                  reversed(encoder_channels[:-1]),
                                                                  decoder_channels):
            self.transposed_convolutions.append(
                nn.Sequential(
                    Upsample(),
                    equalized_layer.EqualizedConv2d(in_channels=current_channel[-1],
                                                    out_channels=decoder_channel[0] - past_channel[-1],
                                                    kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                                    bias=False)
                ))
        # Init final mapping
        self.final_mapping = nn.Sequential(
            FusedLeakyReLU(channel=decoder_channels[-1][-1]),
            equalized_layer.EqualizedConv2d(in_channels=decoder_channels[-1][-1], out_channels=1,
                                            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))

    def forward(self, input: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the 3D U-Net discriminator
        :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, time steps, width, height]
        :return: (Tuple[torch.Tensor, torch.Tensor]) Scalar classification prediction and pixel wise prediction without
        time steps dimension
        """
        # Perform 3D fft on input
        if self.fft:
            if input.shape[1] == 3:
                bf_fft = torch.rfft(input[:, 0], signal_ndim=3, normalized=True, onesided=False).permute(0, 4, 1, 2, 3)
                gfp_fft = torch.rfft(input[:, 1], signal_ndim=3, normalized=True, onesided=False).permute(0, 4, 1, 2, 3)
                rfp_fft = torch.rfft(input[:, 2], signal_ndim=3, normalized=True, onesided=False).permute(0, 4, 1, 2, 3)
                # Make input
                input = torch.cat([input, bf_fft, gfp_fft, rfp_fft], dim=1)
            elif input.shape[1] == 2:
                bf_fft = torch.rfft(input[:, 0], signal_ndim=3, normalized=True, onesided=False).permute(0, 4, 1, 2, 3)
                gfp_fft = torch.rfft(input[:, 1], signal_ndim=3, normalized=True, onesided=False).permute(0, 4, 1, 2, 3)
                # Make input
                input = torch.cat([input, bf_fft, gfp_fft], dim=1)
            else:
                bf_fft = torch.rfft(input[:, 0], signal_ndim=3, normalized=True, onesided=False).permute(0, 4, 1, 2, 3)
                # Make input
                input = torch.cat([input, bf_fft], dim=1)
        # Flatten input
        input = input.flatten(start_dim=1, end_dim=2)
        # Perform decoder mapping and store features
        encoder_features = []
        for index, encoder_block in enumerate(self.encoder_blocks):
            input = encoder_block(input)
            if index != (len(self.encoder_blocks) - 1):
                encoder_features.append(input)
                input = self.downscale_convolutions[index](input)
        # Predict classification
        classification = self.classification_head(input)
        # Decoder forward pass
        for decoder_block, transposed_convolution, encoder_feature \
                in zip(self.decoder_blocks, self.transposed_convolutions, reversed(encoder_features)):
            input = decoder_block(torch.cat([transposed_convolution(input), encoder_feature], dim=1))
        # Perform final mapping
        classification_pixel_wise = self.final_mapping(input).unsqueeze(dim=2)
        return classification, classification_pixel_wise


class ResNetBlock(nn.Module):
    """
    This class implements a simple residual block for the 3D U-Net discriminator including two 3D convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, mini_batch_std_dev: bool = False) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param mini_batch_std_dev: (bool) If true mini batch std dev is utilized
        """
        # Call super constructor
        super(ResNetBlock, self).__init__()
        # Init mini batch std dev if needed
        self.mini_batch_std_dev = MinibatchStdDev() if mini_batch_std_dev else nn.Identity()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            equalized_layer.EqualizedConv2d(in_channels=in_channels + 1 if mini_batch_std_dev else in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            FusedLeakyReLU(out_channels),
            equalized_layer.EqualizedConv2d(in_channels=out_channels, out_channels=out_channels,
                                            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            FusedLeakyReLU(out_channels)
        )
        # Init residual mapping
        self.residual_mapping = equalized_layer.EqualizedConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
            bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block
        :param input: (torch.Tensor) Input tensor 5D
        :return: (torch.Tensor) Output tensor 5d
        """
        # Perform mini batch std dev
        output = self.mini_batch_std_dev(input)
        # Perform main mapping
        output = self.main_mapping(output)
        # Perform residual mapping
        output = (output + self.residual_mapping(input)) / math.sqrt(2)
        return output


class MinibatchStdDev(nn.Module):
    """
    Mini-batch standard deviation module computes the standard deviation of every feature
    vector of a pixel and concatenates the resulting map to the original tensor
    """

    def __init__(self, alpha: float = 1e-8) -> None:
        """
        Constructor method
        :param alpha: (float) Small constant for numeric stability
        """
        # Constructor method
        super(MinibatchStdDev, self).__init__()
        # Save parameters
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (Torch Tensor) Input tensor [batch size, channels,, height, width]
        :return: (Torch Tensor) Output tensor [batch size, channels, height, width]
        """
        # Calc stddev
        output = input - torch.mean(input, dim=0, keepdim=True)
        output = torch.sqrt(torch.mean(output ** 2, dim=0, keepdim=False).clamp(min=self.alpha))
        output = torch.mean(output).view(1, 1, 1)
        output = output.repeat(input.shape[0], 1, input.shape[2], input.shape[3])
        output = torch.cat((input, output), 1)
        return output


class Upsample(nn.Module):
    """
    Upsample module
    """

    def __init__(self, blur_kernel: List[int] = [1, 3, 3, 1], factor: int = 2) -> None:
        """
        Constructor method
        :param blur_kernel: (List[int]) List of weights for the blur kernel to be used
        :param factor: (int) Upscale factor
        """
        # Call super constructor
        super(Upsample, self).__init__()
        # Save parameter
        self.factor = factor
        # Make kernel
        kernel = self.make_kernel(kernel=blur_kernel)
        # Save kernel
        self.register_buffer('kernel', kernel)
        # Calc padding factor
        padding_factor = kernel.shape[0] - factor
        # Calc padding
        self.padding = (((padding_factor + 1) // 2) + factor - 1, padding_factor // 2)

    def make_kernel(self, kernel: List[int]) -> torch.Tensor:
        """
        Method generates a kernel matrix for a given input list of weights
        :param kernel: (List[int]) List of weights for the blur kernel to be used
        :return: (torch.Tensor) Kernel tensor
        """
        # Kernel into list
        kernel = torch.tensor(kernel, dtype=torch.float)
        # Change dim of tensor if list list is 1d
        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]
        # Normalize kernel
        kernel /= kernel.sum()
        return kernel.float()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Upscaled output tensor
        """
        output = upfirdn2d(input=input, kernel=self.kernel, up=self.factor, pad=self.padding)
        return output


class Blur(nn.Module):
    """
    This class implements a blur upsampling
    """

    def __init__(self, kernel: List[int] = [1, 3, 3, 1], sampling_factor: int = 1,
                 sampling_factor_padding: int = 2,
                 kernel_size: int = 3) -> None:
        """
        Constructor method
        :param kernel: (List[int]) List of kernel weights
        :param sampling_factor: (int) Scaling factor
        :param sampling_factor_padding: (int) Scaling factor for padding
        :param kernel_size: (int) Blur kernel size
        """
        # Call super constructor
        super(Blur, self).__init__()
        # Save padding factor
        self.padding = self.calc_padding(kernel, sampling_factor_padding=sampling_factor_padding,
                                         kernel_size=kernel_size)
        # Init kernel
        kernel = self.make_kernel(kernel)
        # Rescale kernel if sampling factor is bigger than one
        if sampling_factor > 1:
            kernel = kernel * (sampling_factor ** 2)
        # Save kernel
        self.register_buffer('kernel', kernel)

    def calc_padding(self, kernel: List[int], sampling_factor_padding: int = 2,
                     kernel_size: int = 3) -> Tuple[int, int]:
        """
        Method estimates the padding factor
        :param kernel: (List[int]) List of kernel weights
        :param sampling_factor: (int) Factor used in scaling
        :param kernel_size: (int) Kernel size of convolution afterwards
        :return: (Tuple[int, int]) Padding in x and y direction
        """
        padding_factor = (len(kernel) - sampling_factor_padding) + (kernel_size - 1)
        padding = ((padding_factor + 1) // 2, padding_factor // 2)
        return padding

    def make_kernel(self, kernel: List[int]) -> torch.Tensor:
        """
        Method generates a kernel matrix for a given input list of weights
        :param kernel: (List[int]) List of weights for the blur kernel to be used
        :return: (torch.Tensor) Kernel tensor
        """
        # Kernel into list
        kernel = torch.tensor(kernel, dtype=torch.float32)
        # Change dim of tensor if list list is 1d
        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]
        # Normalize kernel
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Output tensor
        """
        output = upfirdn2d(input, self.kernel, pad=self.padding)
        return output


class NonLocalBlock(nn.Module):
    """
    This class implements a non-local block used in the original U-Net discriminator.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        # Call super constructor
        super(NonLocalBlock, self).__init__()
        # Init convolutions
        self.theta = equalized_layer.EqualizedConv2d(in_channels=in_channels, out_channels=out_channels // 8,
                                                     kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.phi = equalized_layer.EqualizedConv2d(in_channels=in_channels, out_channels=out_channels // 8,
                                                   kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.g = equalized_layer.EqualizedConv2d(in_channels=in_channels, out_channels=out_channels // 2,
                                                 kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.o = equalized_layer.EqualizedConv2d(in_channels=out_channels // 2, out_channels=out_channels,
                                                 kernel_size=(1, 1), padding=(0, 0), bias=False)
        # Init residual mapping
        self.residual_mapping = equalized_layer.EqualizedConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0),
            bias=False) if in_channels != out_channels else nn.Identity()
        # Init gain parameter
        self.register_parameter(name="gamma", param=nn.Parameter(torch.tensor(0.), requires_grad=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Output tensor
        """
        # Save input shape
        batch_size, _, height, width = input.shape
        # Perform convolutional mappings [batch size, out channels // 8, height, width]
        theta = self.theta(input)
        # [batch size, out channels // 8, height // 2, width // 2]
        phi = F.max_pool2d(self.phi(input), kernel_size=(2, 2), stride=(2, 2))
        # [batch size, out channels // 2, height // 2, width // 2]
        g = F.max_pool2d(self.g(input), kernel_size=(2, 2), stride=(2, 2))
        # Flatten spatial dimensions
        theta = theta.flatten(start_dim=2)  # [batch size, out channels // 8, height * width]
        phi = phi.flatten(start_dim=2)  # [batch size, out channels // 8, height // 2 * width // 2]
        g = g.flatten(start_dim=2)  # [batch size, out channels // 2, height // 2 * width // 2]
        # Perform matrix multiplication and softmax [batch size, out channels // 8, out channels // 2]
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Final matrix multiplication and convolution [batch size, out channels, height, width]
        output = self.o(torch.bmm(g, beta.transpose(1, 2)).view(batch_size, -1, height, width))
        return (self.gamma * output + self.residual_mapping(input)) / math.sqrt(2)


def generate_cut_mix_augmentation_data(image_real: torch.Tensor,
                                       image_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This functions generates the input image and target real fake image for CutMix augmentation.
    :param image_real: (torch.Tensor) Real image
    :param image_fake: (torch.Tensor) Fake image
    :return: (torch.Tensor) Combined input image and corresponding real/fake label
    """
    # Ensure same dim
    image_fake = image_fake[:image_real.shape[0]]
    # Get target map
    target = _generate_binary_cut_mix_map(height=image_real.shape[-2], width=image_fake.shape[-1],
                                          device=image_real.device)
    # Construct input image
    input_image = image_real * target + image_fake * (- target + 1.)
    return input_image, target


def generate_cut_mix_transformation_data(image_real: torch.Tensor, image_fake: torch.Tensor,
                                         prediction_real: torch.Tensor,
                                         prediction_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function generates the input and target for cut mix transformation consistency.
    :param image_real: (torch.Tensor) Real image
    :param image_fake: (torch.Tensor) Fake image
    :param prediction_real: (torch.Tensor) Pixel-wise prediction of discriminator for real image
    :param prediction_fake: (torch.Tensor) Pixel-wise prediction of discriminator for fake image
    :return: (torch.Tensor) Combined input image and corresponding combined soft real/fake label
    """
    # Ensure same dim
    image_fake = image_fake[:image_real.shape[0]]
    prediction_fake = prediction_fake[:image_real.shape[0]]
    # Get binary map
    binary_map = _generate_binary_cut_mix_map(height=image_real.shape[-2], width=image_fake.shape[-1],
                                              device=image_real.device)
    # Construct input image
    input_image = image_real * binary_map + image_fake * (- binary_map + 1.)
    # Construct target
    target = prediction_real * binary_map + prediction_fake * (- binary_map + 1.)
    return input_image, target


def _generate_binary_cut_mix_map(height: int, width: int,
                                 device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    This function generate a random binary map with two areas for cut mix augmentation of consistency regularization.
    :param height: (int) Height of the map
    :param width: (int) Width of the map
    :param device: (Union[str, torch.device]) Device to utilize
    :return: (torch.Tensor) Binary map
    """
    # Make target
    binary_map = torch.zeros(1, 1, 1, height, width, dtype=torch.float, device=device)
    # Generate a random coordinates to perform cut
    cut_coordinates_height = torch.randint(int(0.1 * height), int(0.9 * height), size=(1,))
    cut_coordinates_width = torch.randint(int(0.1 * width), int(0.9 * width), size=(1,))
    # Apply coordinates
    if random.random() > 0.5:
        binary_map[..., cut_coordinates_height:, cut_coordinates_width:] = 1.0
    else:
        binary_map[..., :cut_coordinates_height, :cut_coordinates_width] = 1.0
    # Invert target randomly
    if random.random() > 0.5:
        binary_map = - binary_map + 1.
    return binary_map
