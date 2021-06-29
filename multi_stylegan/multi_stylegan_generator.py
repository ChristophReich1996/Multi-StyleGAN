from typing import Iterable, List, Union, Tuple, Dict, Any, Optional

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import autograd

from .op_static import FusedLeakyReLU, upfirdn2d

from . import equalized_layer


class Generator(nn.Module):
    """
    Implementation of the stylegan2 generator
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Dict with network configurations
        """
        # Call super constructor
        super(Generator, self).__init__()
        # Save parameter
        channels: Tuple[int, ...] = config["channels"]
        channel_factor: Union[int, float] = config["channel_factor"]
        self.out_channels: int = 3
        self.latent_dimensions: int = config["latent_dimensions"]
        depth_style_mapping: int = config["depth_style_mapping"]
        self.starting_resolution: Tuple[int, int] = config["starting_resolution"]
        # Init style mapping
        self.style_mapping = StyleMapping(latent_dimensions=self.latent_dimensions, depth=depth_style_mapping)
        # Init constant input module
        self.constant_input_1 = ConstantInput(channel=int(channels[0] // channel_factor), size=self.starting_resolution)
        self.constant_input_2 = ConstantInput(channel=int(channels[0] // channel_factor), size=self.starting_resolution)
        # Init first styled convolution
        self.starting_convolution_1 = StyledConv2d(in_channels=int(channels[0] // channel_factor),
                                                   out_channels=int(channels[0] // channel_factor), kernel_size=(3, 3),
                                                   style_dimension=self.latent_dimensions, upsampling=False,
                                                   demodulate=True)
        self.starting_convolution_2 = StyledConv2d(in_channels=int(channels[0] // channel_factor),
                                                   out_channels=int(channels[0] // channel_factor), kernel_size=(3, 3),
                                                   style_dimension=self.latent_dimensions, upsampling=False,
                                                   demodulate=True, modulation_mapping=False)
        # Init first output mapping
        self.starting_output_block_1 = OutputBlock(in_channels=int(channels[0] // channel_factor),
                                                   out_channels=self.out_channels,
                                                   style_dimension=self.latent_dimensions, upsampling=False)
        self.starting_output_block_2 = OutputBlock(in_channels=int(channels[0] // channel_factor),
                                                   out_channels=self.out_channels,
                                                   style_dimension=self.latent_dimensions, upsampling=False,
                                                   modulation_mapping=False)
        # Init main styled convolutions and output blocks
        self.main_convolutions_1 = nn.ModuleList()
        self.output_blocks_1 = nn.ModuleList()
        self.main_convolutions_2 = nn.ModuleList()
        self.output_blocks_2 = nn.ModuleList()
        for index in range(len(channels) - 1):
            self.main_convolutions_1.append(
                StyledConv2d(in_channels=int(channels[index] // channel_factor),
                             out_channels=int(channels[index + 1] // channel_factor), kernel_size=(2, 2),
                             style_dimension=self.latent_dimensions, upsampling=True, demodulate=True))
            self.main_convolutions_1.append(
                StyledConv2d(in_channels=int(channels[index + 1] // channel_factor),
                             out_channels=int(channels[index + 1] // channel_factor), kernel_size=(3, 3),
                             style_dimension=self.latent_dimensions, upsampling=False, demodulate=True))
            self.output_blocks_1.append(OutputBlock(in_channels=int(channels[index + 1] // channel_factor),
                                                    out_channels=self.out_channels,
                                                    style_dimension=self.latent_dimensions, upsampling=True))
            self.main_convolutions_2.append(
                StyledConv2d(in_channels=int(channels[index] // channel_factor),
                             out_channels=int(channels[index + 1] // channel_factor), kernel_size=(2, 2),
                             style_dimension=self.latent_dimensions, upsampling=True, demodulate=True,
                             modulation_mapping=False))
            self.main_convolutions_2.append(
                StyledConv2d(in_channels=int(channels[index + 1] // channel_factor),
                             out_channels=int(channels[index + 1] // channel_factor), kernel_size=(3, 3),
                             style_dimension=self.latent_dimensions, upsampling=False, demodulate=True,
                             modulation_mapping=False))
            self.output_blocks_2.append(OutputBlock(in_channels=int(channels[index + 1] // channel_factor),
                                                    out_channels=self.out_channels,
                                                    style_dimension=self.latent_dimensions, upsampling=True,
                                                    modulation_mapping=False))
        # Init noises as a nn.Module to store each tensor
        self.noises = nn.Module()
        self.noises.register_buffer('noise_start',
                                    torch.randn(1, 1, self.starting_resolution[0], self.starting_resolution[1]))
        for index in range(len(channels) - 1):
            self.noises.register_buffer('noise_{}'.format((2 * index)),
                                        torch.randn(1, 1, 2 ** (index + 3), 2 ** (index + 3)))
            self.noises.register_buffer('noise_{}'.format((2 * index) + 1),
                                        torch.randn(1, 1, 2 ** (index + 3), 2 ** (index + 3)))

    def get_parameters(self, lr_main: float = 1e-03, lr_style: float = 1e-05) -> Iterable:
        """
        Method returns all parameters of the model with different learning rates
        :return: (Iterable) Iterable object including the main parameters of the generator network
        """
        return [{'params': self.constant_input_1.parameters(), 'lr': lr_main},
                {'params': self.starting_convolution_1.parameters(), 'lr': lr_main},
                {'params': self.starting_output_block_1.parameters(), 'lr': lr_main},
                {'params': self.main_convolutions_1.parameters(), 'lr': lr_main},
                {'params': self.output_blocks_1.parameters(), 'lr': lr_main},
                {'params': self.constant_input_2.parameters(), 'lr': lr_main},
                {'params': self.starting_convolution_2.parameters(), 'lr': lr_main},
                {'params': self.starting_output_block_2.parameters(), 'lr': lr_main},
                {'params': self.main_convolutions_2.parameters(), 'lr': lr_main},
                {'params': self.output_blocks_2.parameters(), 'lr': lr_main},
                {'params': self.style_mapping.parameters(), 'lr': lr_style}]

    def forward(self,
                input: Union[List[torch.Tensor], torch.Tensor],
                return_main_style_vectors: bool = False,
                noise: Optional[List[torch.Tensor]] = None,
                randomize_noise: bool = True,
                inject_index: Optional[int] = None,
                input_is_latent: bool = False,
                return_path_length_grads: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        :param input: (List[torch.Tensor], torch.Tensor) Input noise tensor
        :param return_main_style_vectors: (bool) If true latent style vectors will be returned
        :param noise: (Optional[List[torch.Tensor]]) List of noise vectors for noisy bias
        :param randomize_noise: (bool) If true random noisy bias will be produced, otherwise fixed noise is used
        :param inject_index: (Optional[int]) Index to inject two style tensors to styled convolutions
        :param input_is_latent: (bool) If input is in latent space set to true
        :return: (torch.Tensor, Tuple[torch.Tensor, torch.Tensor]) Generated image and optimal latent style vectors
        """
        # Make style vectors if not latent input
        if not input_is_latent:
            if isinstance(input, list):
                styles = [self.style_mapping(input[index]) for index in range(len(input))]
            else:
                styles = self.style_mapping(input)
        # Init or generate noise
        if noise is None:
            if randomize_noise:
                noise_start = None
                noise = [None] * len(self.main_convolutions_1)
            else:
                noise_start = getattr(self.noises, 'noise_start')
                noise = [getattr(self.noises, 'noise_{}'.format(index)) for index in
                         range(len(self.main_convolutions_1))]
        else:
            noise_start = noise[0]
            noise = noise[1:]
        # Construct style tensors
        if not input_is_latent:
            if isinstance(styles, list):
                if inject_index is None:
                    inject_index = np.random.randint(1, len(self.main_convolutions_1) + 2 - 1)
                latent_1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent_2 = styles[1].unsqueeze(1).repeat(1, len(self.main_convolutions_1) + 2 - inject_index, 1)
                latent = torch.cat((latent_1, latent_2), dim=1)
            else:
                latent = styles.unsqueeze(1).repeat(1, len(self.main_convolutions_1) + 2, 1)
        else:
            if input.ndim < 3:
                # Add third dim and repeat to match styled convolutions
                latent = input.unsqueeze(1).repeat(1, len(self.main_convolutions_1) + 2, 1)
            elif input.shape[1] != len(self.main_convolutions_1) + 2:
                # Check that dim 1 has the shape of one
                assert input.shape[1] == 0
                # Repeat input latent vector to match styled convolutions
                latent = input.repeat(1, len(self.main_convolutions_1) + 2, 1)
            else:
                # Set input as latent vector
                latent = input
        # Perform starting operations
        output_1 = self.constant_input_1(latent)
        output_2 = self.constant_input_2(latent)
        output_1, style = self.starting_convolution_1(output_1, latent[:, 0], noise=noise_start)
        output_2 = self.starting_convolution_2(output_2, style, noise=noise_start)
        skip_1, style = self.starting_output_block_1(output_1, latent[:, 1])
        skip_2 = self.starting_output_block_2(output_2, style)
        # Perform main path
        for index in range(len(self.main_convolutions_1) // 2):
            output_1, style = self.main_convolutions_1[index * 2](output_1, latent[:, index * 2 + 1],
                                                                  noise=noise[index * 2])
            output_2 = self.main_convolutions_2[index * 2](output_2, style, noise=noise[index * 2])
            output_1, style = self.main_convolutions_1[index * 2 + 1](output_1, latent[:, index * 2 + 2],
                                                                      noise=noise[index * 2 + 1])
            output_2 = self.main_convolutions_2[index * 2 + 1](output_2, style, noise=noise[index * 2 + 1])
            skip_1, style = self.output_blocks_1[index](output_1, latent[:, index * 2 + 3], skip=skip_1)
            skip_2 = self.output_blocks_2[index](output_1, style, skip=skip_2)
        # Get final image
        image = torch.stack([skip_1, skip_2], dim=1)
        # Return path length grads if utilized
        if return_path_length_grads:
            # Make noise tensor
            noise = torch.randn(image.shape, device=image.device, dtype=torch.float32, requires_grad=True) \
                    / math.sqrt(image.shape[2] * image.shape[3] * image.shape[4])
            # Calc gradient
            grad = autograd.grad(outputs=(image * noise).sum(), inputs=latent, create_graph=True,
                                 retain_graph=True, only_inputs=True)[0]
            return grad
        # Return also the latent vector if needed
        if return_main_style_vectors:
            return image, latent
        else:
            return image


class StyleMapping(nn.Module):
    """
    Implementation of the fully connected path to map the latent tensor
    """

    def __init__(self, latent_dimensions: int = 512, depth: int = 8) -> None:
        """
        Constructor method
        :param latent_dimensions: (int) Dimension of latent tensor
        :param depth: (int) Number if linear layers used in mapping path
        """
        # Call super constructor
        super(StyleMapping, self).__init__()
        # Init layers
        layers = [equalized_layer.PixelwiseNormalization()]
        for _ in range(depth):
            layers.extend([equalized_layer.EqualizedLinear(latent_dimensions, latent_dimensions, bias=False),
                           FusedLeakyReLU(latent_dimensions)])
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param noise: (torch.Tensor) Noise input tensor
        :return: (torch.Tensor) Mapped latent vector
        """
        output = self.layers(noise)
        return output


class ConstantInput(nn.Module):
    """
    Implementation of a module with a constant learnable output
    """

    def __init__(self, channel: int, size: Tuple[int, int] = (4, 4)) -> None:
        """
        Constructor method
        :param channel: (int) Number of channels utilized in constant variable
        :param size: (Tuple[int, int]) Width and height of the constant variable
        """
        # Call super constructor
        super(ConstantInput, self).__init__()
        # Init constant variable
        self.input = nn.Parameter(torch.ones(1, channel, size[0], size[1]), requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of latent path only to get batch size dimension
        :return: (torch.Tensor) Constant tensor
        """
        # Get batch size
        batch_size = input.shape[0]
        # Repeat input tensor to match batch size
        output = self.input.repeat_interleave(dim=0, repeats=batch_size)
        return output


class NoiseInjection(nn.Module):
    """
    Model injects noisy bias to input tensor
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(NoiseInjection, self).__init__()
        # Init bias weights
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def forward(self, input: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param noise: (Optional[torch.Tensor]) Noise tensor
        :return: (torch.Tensor) Output tensor
        """
        if noise is None:
            noise = torch.randn(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device,
                                dtype=torch.float32)

        return input + self.weight * noise


class ModulatedConv2d(nn.Module):
    """
    Modulated 2d convolution proposed in:
    https://arxiv.org/abs/1912.04958
    """

    def __init__(self, in_channels: int, out_channels: int, style_dimension: int,
                 kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 demodulate: bool = True, upsampling: bool = True,
                 blur_kernel: List[int] = [1, 3, 3, 1], modulation_mapping: bool = True) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels to be utilized
        :param out_channels: (int) Number of output channels to be utilized
        :param style_dimension: (int) Number of style dimensions
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size of the convolution filter
        :param demodulate: (bool) True if weights should be demodulated
        :param upsampling: (bool) True if output should be downscaled
        :param blur_kernel: (List[int]) List of weights for the blur kernel to be used
        :param modulation_mapping: (bool) If true modulation mapping is utilized
        """
        # Call super constructor
        super(ModulatedConv2d, self).__init__()
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.upsampling = upsampling
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        # Init blur upsampling
        self.blur = Blur(kernel=blur_kernel, sampling_factor=2, sampling_factor_padding=2,
                         kernel_size=kernel_size[0]) if upsampling else None
        # If upsampling is utilized perform no parring in convolution if not perform same padding in convolution
        if upsampling:
            self.padding = (0, 0)
            self.stride = (2, 2)
        else:
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
            self.stride = (1, 1)
        # Init scaling factor
        self.scale = math.sqrt(2) / math.sqrt(in_channels * self.kernel_size[0] * self.kernel_size[1])
        # Init convolution weights
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, self.kernel_size[0], self.kernel_size[1], dtype=torch.float32),
            requires_grad=True)
        # Init linear layer for modulation
        self.modulation_mapping = equalized_layer.EqualizedLinear(in_channels=style_dimension,
                                                                  out_channels=in_channels,
                                                                  bias=True) if modulation_mapping else None
        # Reset bias of linear layer to ones
        if modulation_mapping:
            self.modulation_mapping.bias.data.fill_(1.0)

    def __repr__(self) -> str:
        """
        Method to return info about the object
        :return: (str) String including information about module
        """
        return ('{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), upsampling={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.upsampling))

    def forward(self, input: torch.Tensor, style: torch.Tensor) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param style: (torch.Tensor) Style tensor
        :return: (torch.Tensor) Output tensor and if modulation mapping is utilized also modulated style is returned
        """
        # Save shape of input
        batch_size, features, height, width = input.shape
        # Check input shape
        assert features == self.in_channels, \
            'Expect input feature shape of {} but get {}.'.format(self.in_channels, features)
        # Get modulated style from linear layer
        if self.modulation_mapping is not None:
            modulated_style = self.modulation_mapping(style).view(batch_size, 1, self.in_channels, 1, 1)
        else:
            modulated_style = style
        # Scale weights of convolution
        weight = self.scale * self.weight * modulated_style
        # Demodulate weights if utilized
        if self.demodulate:
            demodulation_factor = torch.rsqrt(torch.sum(weight ** 2, dim=[2, 3, 4]) + 1e-08)
            weight = weight * demodulation_factor.view(batch_size, self.out_channels, 1, 1, 1)
        # Reshape input
        input = input.view(1, batch_size * features, height, width)
        if self.upsampling:
            # Reshape weights to perform transposed convolution
            weight = weight.view(batch_size, self.out_channels, self.in_channels, self.kernel_size[0],
                                 self.kernel_size[1])
            weight = weight.transpose(1, 2).reshape(batch_size * self.in_channels, self.out_channels,
                                                    self.kernel_size[0], self.kernel_size[1])
            # Perform transposed convolution
            output = F.conv_transpose2d(input=input, weight=weight, padding=self.padding, stride=self.stride,
                                        groups=batch_size)
            # Reshape output
            output = output.view(batch_size, self.out_channels, output.shape[2], output.shape[3])
            # Perform blur operation
            output = self.blur(output)
        else:
            # Reshape weights to perform convolution
            weight = weight.view(batch_size * self.out_channels, self.in_channels, self.kernel_size[0],
                                 self.kernel_size[1])
            # Perform convolution
            output = F.conv2d(input=input, weight=weight, padding=self.padding, stride=self.stride, groups=batch_size)
            # Reshape output
            output = output.view(batch_size, self.out_channels, output.shape[2], output.shape[3])
        if self.modulation_mapping is not None:
            return output, modulated_style
        return output


class StyledConv2d(nn.Module):
    """
    Implementation of a styled convolution, including a modulated 2d convolution, a noisy bias injection and a fused
    leaky ReLU activation.
    https://arxiv.org/abs/1912.04958
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 style_dimension: int, demodulate: bool = True, upsampling: bool = False,
                 blur_kernel: List[int] = [1, 3, 3, 1], modulation_mapping: bool = True) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels to be utilized
        :param out_channels: (int) Number of output channels to be utilized
        :param style_dimension: (int) Number of style dimensions
        :param kernel_size: (int, Tuple[int, int]) Kernel size of the convolution filter
        :param demodulate: (bool) True if weights should be demodulated
        :param upsampling: (bool) True if output should be downscaled
        :param blur_kernel: (List[int]) List of weights for the blur kernel to be used
        :param modulation_mapping: (bool) If true modulation mapping is utilized
        """
        # Call super constructor
        super(StyledConv2d, self).__init__()
        # Save parameters
        self.modulation_mapping = modulation_mapping
        # Init modulated convolution
        self.modulated_convolution = ModulatedConv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, style_dimension=style_dimension,
                                                     demodulate=demodulate, upsampling=upsampling,
                                                     blur_kernel=blur_kernel, modulation_mapping=modulation_mapping)
        # Init noisy bias injection
        self.noise_injection = NoiseInjection()
        # Init activation including bias
        self.activation = FusedLeakyReLU(out_channels)

    def forward(self, input: torch.Tensor, style: torch.Tensor,
                noise: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param style: (torch.Tensor) Style tensor
        :param noise: (torch.Tensor) Noise tensor
        :return: (torch.Tensor) Output tensor and if modulation mapping is utilized style vector is returned
        """
        if self.modulation_mapping:
            output, style = self.modulated_convolution(input, style)
        else:
            output = self.modulated_convolution(input, style)
        output = self.noise_injection(output, noise=noise)
        output = self.activation(output)
        if self.modulation_mapping:
            return output, style
        return output


class OutputBlock(nn.Module):
    """
    Implementation of a output block including a modulated (no demodulation) 2d convolution a upsampling operation and
    a bias operation
    """

    def __init__(self, in_channels: int, style_dimension: int, out_channels: int = 1,
                 upsampling: bool = False, blur_kernel: List[int] = [1, 3, 3, 1],
                 modulation_mapping: bool = True) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param style_dimension: (int) Style vector dimension
        :param out_channels: (int) Number of output channels
        :param upsampling: (bool) If true 2x upsampling is utilized
        :param blur_kernel: (List[int]) List of kernel weights
        :param modulation_mapping: (bool) If true modulation mapping is utilized
        """
        # Call super constructor
        super(OutputBlock, self).__init__()
        # Save parameters
        self.modulation_mapping = modulation_mapping
        # Init upsampling operation
        self.upsampling = Upsample(blur_kernel=blur_kernel, factor=2) if upsampling else nn.Identity()
        # Init modulated convolution
        self.modulated_convolution = ModulatedConv2d(in_channels=in_channels, out_channels=out_channels,
                                                     style_dimension=style_dimension, kernel_size=(1, 1),
                                                     upsampling=False, demodulate=False,
                                                     modulation_mapping=modulation_mapping)
        # Init bias parameter
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1, dtype=torch.float32), requires_grad=True)

    def forward(self, input: torch.Tensor, style: torch.Tensor,
                skip: torch.Tensor = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param style: (torch.Tensor) Output tensor
        :param skip: (torch.Tensor) Tensor for skip connection
        :return: (torch.Tensor) Output tensor
        """
        # Perform convolution
        if self.modulation_mapping:
            output, style = self.modulated_convolution(input, style)
        else:
            output = self.modulated_convolution(input, style)
        # Add bias
        output = output + self.bias
        # Map skip tensor
        if skip is not None:
            skip = self.upsampling(skip)
            output = output + skip
        if self.modulation_mapping:
            return output, style
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

    def __init__(self, kernel: List[int], sampling_factor: int = 1,
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
