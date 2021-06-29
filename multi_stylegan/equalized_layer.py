from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EqualizedConv2d(nn.Module):
    """
    Implementation of a 2d equalized Convolution
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True):
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size
        :param stride: (Union[int, Tuple[int, int]]) Stride factor used in the convolution
        :param padding: (Union[int, Tuple[int, int]]) Padding factor used in the convolution
        :param bias: (bool) Use bias
        """
        # Init super constructor
        super(EqualizedConv2d, self).__init__()
        # Save parameters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        # Init weights tensor for convolution
        self.weight = nn.Parameter(
            nn.init.normal_(torch.empty(out_channels, in_channels, *self.kernel_size, dtype=torch.float)),
            requires_grad=True)
        # Init bias weight if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float), requires_grad=True)
        else:
            self.bias = None
        # Init scale factor
        self.scale = torch.tensor(
            np.sqrt(2) / np.sqrt(in_channels * (self.kernel_size[0] * self.kernel_size[1]))).float()
        self.scale_bias = torch.tensor(np.sqrt(2) / np.sqrt(out_channels)).float()

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ('{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), bias={})'.format(
            self.__class__.__name__,
            self.weight.shape[1],
            self.weight.shape[0],
            self.weight.shape[2],
            self.weight.shape[3],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.bias is not None))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (Torch Tensor) Input tensor 4D
        :return: (Torch tensor) Output tensor 4D
        """
        if self.bias is None:
            output = F.conv2d(input=input, weight=self.weight * self.scale, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(input=input, weight=self.weight * self.scale, bias=self.bias * self.scale_bias,
                              stride=self.stride, padding=self.padding)
        return output


class EqualizedTransposedConv2d(nn.Module):
    """
    Implementation of a 2d equalized transposed Convolution
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = 2,
                 stride: Union[int, Tuple[int, int]] = 2, padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size
        :param stride: (Union[int, Tuple[int, int]]) Stride factor used in the convolution
        :param padding: (Union[int, Tuple[int, int]]) Padding factor used in the convolution
        :param bias: (bool) Use bias
        """
        # Init super constructor
        super(EqualizedTransposedConv2d, self).__init__()
        # Save parameters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        # Init weights tensor for convolution
        self.weight = nn.Parameter(
            nn.init.normal_(torch.empty(in_channels, out_channels, *self.kernel_size, dtype=torch.float)),
            requires_grad=True)
        # Init bias weight if needed
        if bias:
            self.bias = nn.Parameter(torch.ones(out_channels, dtype=torch.float), requires_grad=True)
        else:
            self.bias = None
        # Init scale factor
        self.scale = torch.tensor(
            np.sqrt(2) / np.sqrt(in_channels * (self.kernel_size[0] * self.kernel_size[1]))).float()
        self.scale_bias = torch.tensor(np.sqrt(2) / np.sqrt(out_channels)).float()

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ('{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), bias={})'.format(
            self.__class__.__name__,
            self.weight.shape[1],
            self.weight.shape[0],
            self.weight.shape[2],
            self.weight.shape[3],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.bias is not None))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (Torch Tensor) Input tensor 4D
        :return: (Torch tensor) Output tensor 4D
        """
        if self.bias is None:
            output = F.conv_transpose2d(input=input, weight=self.weight * self.scale, stride=self.stride,
                                        padding=self.padding)
        else:
            output = F.conv_transpose2d(input=input, weight=self.weight * self.scale, bias=self.bias * self.scale_bias,
                                        stride=self.stride, padding=self.padding)
        return output


class EqualizedConv1d(nn.Module):
    """
    This class implements an equalized 1d convolution
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param kernel_size: (int) Kernel size
        :param stride: (int) Stride factor used in the convolution
        :param padding: (int) Padding factor used in the convolution
        :param bias: (bool) Use bias
        """
        # Call super constructor
        super(EqualizedConv1d, self).__init__()
        # Save parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Init weight parameter
        self.weight = nn.Parameter(
            nn.init.normal_(torch.empty(out_channels, in_channels, kernel_size, dtype=torch.float)), requires_grad=True)
        # Init bias if utilized
        if bias:
            self.bias = nn.Parameter(torch.ones(out_channels, dtype=torch.float), requires_grad=True)
        else:
            self.bias = None
        # Init scale factor
        self.scale = torch.tensor(
            np.sqrt(2) / np.sqrt(in_channels * (self.kernel_size))).float()
        self.scale_bias = torch.tensor(np.sqrt(2) / np.sqrt(out_channels)).float()

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ('{}({}, {}, kernel_size={}, stride={}, padding={}, bias={})'.format(
            self.__class__.__name__,
            self.weight.shape[1],
            self.weight.shape[0],
            self.weight.shape[2],
            self.stride,
            self.padding,
            self.bias is not None))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor 3D
        :return: (torch.Tensor) Output tensor 3D
        """
        if self.bias is None:
            output = F.conv1d(input=input, weight=self.weight * self.scale, stride=self.stride,
                              padding=self.padding)
        else:
            output = F.conv1d(input=input, weight=self.weight * self.scale, bias=self.bias * self.scale_bias,
                              stride=self.stride, padding=self.padding)
        return output


class EqualizedLinear(nn.Module):
    """
    Implementation of an equalized linear layer
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param use_bias: (bool) True if bias should be used
        """
        # Init super constructor
        super(EqualizedLinear, self).__init__()
        # Init weights tensor for convolution
        self.weight = nn.Parameter(
            nn.init.normal_(torch.empty(out_channels, in_channels, dtype=torch.float)), requires_grad=True)
        # Init bias weight if needed
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).fill_(0), requires_grad=True)
        else:
            self.bias = None
        # Init scale factor
        self.scale = np.sqrt(2) / np.sqrt(in_channels)
        self.scale_bias = np.sqrt(2) / np.sqrt(out_channels)

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ('{}({}, {}, bias={})'.format(self.__class__.__name__, self.weight.shape[1], self.weight.shape[0],
                                             self.bias is not None))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor 2D or 3D
        :return: (torch.Tensor) Output tensor 2D or 3D
        """
        if self.bias is None:
            output = F.linear(input=input, weight=self.weight * self.scale)
        else:
            output = F.linear(input=input, weight=self.weight * self.scale, bias=self.bias * self.scale_bias)
        return output


class PixelwiseNormalization(nn.Module):
    """
    Pixelwise Normalization module
    """

    def __init__(self, alpha: float = 1e-8) -> None:
        """
        Constructor method
        :param alpha: (float) Small constants for numeric stability
        """
        super(PixelwiseNormalization, self).__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (Torch Tensor) Input tensor
        :return: (Torch Tensor) Normalized output tensor with same shape as input
        """
        output = input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + self.alpha)
        return output
