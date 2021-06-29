import torch


def normalize_0_1(tensor: torch.Tensor, max: float = None, min: float = None) -> torch.Tensor:
    """
    Function normalizes a given input tensor channel-wise to a rage between zero and one.
    :param tensor: (torch.Tensor) Input tensor of the shape [channels, height, width]
    :param max: (float) Max value utilized in the normalization
    :param min: (float) Min value of the normalization
    :return: (torch.Tensor) Normalized input tensor of the same shape as the input
    """
    # Save shape
    channels, height, width = tensor.shape
    # Flatten input tensor to the shape [channels, height * width]
    tensor = tensor.flatten(start_dim=1)
    # Get channel wise min and max
    tensor_min = tensor.min(dim=1, keepdim=True)[0].float() if min is None else torch.tensor(min, dtype=torch.float)
    tensor_max = tensor.max(dim=1, keepdim=True)[0].float() if max is None else torch.tensor(max, dtype=torch.float)
    # Normalize tensor
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    # Reshape tensor to original shape
    tensor = tensor.reshape(channels, height, width)
    return tensor
