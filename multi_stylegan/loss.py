from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class WassersteinDiscriminatorLoss(nn.Module):
    """
    This class implements the Wasserstein loss for a discriminator network.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(WassersteinDiscriminatorLoss, self).__init__()

    def forward(self, prediction_real: torch.Tensor,
                prediction_fake: torch.Tensor, weight: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the loss module
        :param prediction_real: (torch.Tensor) Prediction for real samples
        :param prediction_fake: (torch.Tensor) Prediction for fake samples
        :param weight: (torch.Tensor) Weights map to be applied
        :return: (Tuple[torch.Tensor, torch.Tensor]) Scalar loss value
        """
        # Compute loss
        if weight is not None:
            loss_real = - torch.mean(
                prediction_real * weight.view(1, 1, 1, weight.shape[-2], weight.shape[-1]).to(prediction_real.device))
            loss_fake = torch.mean(
                prediction_fake * weight.view(1, 1, 1, weight.shape[-2], weight.shape[-1]).to(prediction_fake.device))
            return loss_real, loss_fake
        else:
            loss_real = - torch.mean(prediction_real)
            loss_fake = torch.mean(prediction_fake)
            return loss_real, loss_fake


class WassersteinDiscriminatorLossCutMix(nn.Module):
    """
    This class implements the Wasserstein loss for a discriminator network when utilizing cut mix augmentation.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(WassersteinDiscriminatorLossCutMix, self).__init__()

    def forward(self, prediction: torch.Tensor,
                label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction: (torch.Tensor)
        :return: (Tuple[torch.Tensor, torch.Tensor]) Loss values for real and fake part
        """
        # Compute loss
        loss_real = - torch.mean(prediction * label)
        loss_fake = torch.mean(prediction * (-label + 1.))
        return loss_real, loss_fake


class WassersteinGeneratorLoss(nn.Module):
    """
    This class implements the Wasserstein loss for a generator network.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(WassersteinGeneratorLoss, self).__init__()

    def forward(self, prediction_fake: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the loss module
        :param prediction_fake: (torch.Tensor) Prediction for fake samples
        :param weight: (torch.Tensor) Weights map to be applied
        :return: (torch.Tensor) Scalar loss value
        """
        # Compute loss
        if weight is not None:
            loss = - torch.mean(
                prediction_fake * weight.view(1, 1, 1, weight.shape[-2], weight.shape[-1]).to(prediction_fake.device))
            return loss
        else:
            loss = - torch.mean(prediction_fake)
            return loss


class NonSaturatingLogisticGeneratorLoss(nn.Module):
    """
    Implementation of the non saturating GAN loss for the generator network.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(NonSaturatingLogisticGeneratorLoss, self).__init__()

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}'.format(self.__class__.__name__)

    def forward(self, prediction_fake: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass to compute the generator loss
        :param prediction_fake: (torch.Tensor) Prediction of the discriminator for fake samples
        :param weight: (torch.Tensor) Weights map to be applied
        :return: (torch.Tensor) Loss value
        """
        # Calc loss
        if weight is not None:
            loss = torch.mean(
                F.softplus(-prediction_fake) * weight.view(1, 1, 1, weight.shape[-2], weight.shape[-1]).to(
                    prediction_fake.device))
            return loss
        else:
            loss = torch.mean(F.softplus(-prediction_fake))
            return loss


class NonSaturatingLogisticDiscriminatorLoss(nn.Module):
    """
    Implementation of the non saturating GAN loss for the discriminator network.
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(NonSaturatingLogisticDiscriminatorLoss, self).__init__()

    def forward(self, prediction_real: torch.Tensor,
                prediction_fake: torch.Tensor, weight: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for real images
        :param prediction_fake: (torch.Tensor) Prediction of the discriminator for fake images
        :param weight: (torch.Tensor) Weights map to be applied
        :return: (Tuple[torch.Tensor, torch.Tensor]) Loss values for real and fake part
        """
        if weight is not None:
            # Calc real loss part
            loss_real = torch.mean(
                F.softplus(-prediction_real) * weight.view(1, 1, 1, weight.shape[-2], weight.shape[-1]).to(
                    prediction_real.device))
            # Calc fake loss part
            loss_fake = torch.mean(
                F.softplus(prediction_fake) * weight.view(1, 1, 1, weight.shape[-2], weight.shape[-1]).to(
                    prediction_fake.device))
            return loss_real, loss_fake
        else:
            # Calc real loss part
            loss_real = torch.mean(F.softplus(-prediction_real))
            # Calc fake loss part
            loss_fake = torch.mean(F.softplus(prediction_fake))
            return loss_real, loss_fake


class NonSaturatingLogisticDiscriminatorLossCutMix(nn.Module):
    """
    Implementation of the non saturating GAN loss for the discriminator network when performing cut mix augmentation.
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(NonSaturatingLogisticDiscriminatorLossCutMix, self).__init__()

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction: (torch.Tensor)
        :return: (Tuple[torch.Tensor, torch.Tensor]) Loss values for real and fake part
        """
        # Calc real loss part
        loss_real = torch.mean(F.softplus(-prediction) * label)
        # Calc fake loss part
        loss_fake = torch.mean(F.softplus(prediction) * (-label + 1.))
        return loss_real, loss_fake


class HingeGeneratorLoss(WassersteinGeneratorLoss):
    """
    This class implements the hinge gan loss for the generator network. Note that the generator hinge loss is equivalent
    to the generator Wasserstein loss!
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(HingeGeneratorLoss, self).__init__()


class HingeDiscriminatorLoss(nn.Module):
    """
    This class implements the hinge gan loss for the discriminator network.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(HingeDiscriminatorLoss, self).__init__()

    def forward(self, prediction_real: torch.Tensor, prediction_fake: torch.Tensor,
                weight: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for real images
        :param prediction_fake: (torch.Tensor) Prediction of the discriminator for fake images
        :param weight: (torch.Tensor) Weights map to be applied
        :return: (Tuple[torch.Tensor, torch.Tensor]) Loss values for real and fake part
        """
        if weight is not None:
            # Calc loss for real prediction
            loss_real = - torch.mean(torch.minimum(torch.tensor(0., dtype=torch.float, device=prediction_real.device),
                                                   prediction_real - 1.) * weight.view(1, 1, 1, weight.shape[-2],
                                                                                       weight.shape[-1]).to(
                prediction_real.device))
            # Calc loss for fake prediction
            loss_fake = - torch.mean(torch.minimum(torch.tensor(0., dtype=torch.float, device=prediction_real.device),
                                                   - prediction_fake - 1.) * weight.view(1, 1, 1, weight.shape[-2],
                                                                                         weight.shape[-1]).to(
                prediction_fake.device))
            return loss_real, loss_fake
        else:
            # Calc loss for real prediction
            loss_real = - torch.mean(torch.minimum(torch.tensor(0., dtype=torch.float, device=prediction_real.device),
                                                   prediction_real - 1.))
            # Calc loss for fake prediction
            loss_fake = - torch.mean(torch.minimum(torch.tensor(0., dtype=torch.float, device=prediction_real.device),
                                                   - prediction_fake - 1.))
            return loss_real, loss_fake


class HingeDiscriminatorLossCutMix(nn.Module):
    """
    This class implements the hinge gan loss for the discriminator network when utilizing cut mix augmentation.
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(HingeDiscriminatorLossCutMix, self).__init__()

    def forward(self, prediction: torch.Tensor,
                label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Loss parts are not summed up to not retain the whole backward graph later.
        :param prediction: (torch.Tensor)
        :return: (Tuple[torch.Tensor, torch.Tensor]) Loss values for real and fake part
        """
        # Calc loss for real prediction
        loss_real = - torch.mean(torch.minimum(torch.tensor(0., dtype=torch.float, device=prediction.device),
                                               prediction - 1.) * label)
        # Calc loss for fake prediction
        loss_fake = - torch.mean(torch.minimum(torch.tensor(0., dtype=torch.float, device=prediction.device),
                                               - prediction - 1.) * (- label + 1.))
        return loss_real, loss_fake


class R1Regularization(nn.Module):
    """
    Implementation of the R1 GAN regularization.
    """

    def __init__(self):
        """
        Constructor method
        """
        # Call super constructor
        super(R1Regularization, self).__init__()

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}'.format(self.__class__.__name__)

    def forward(self, prediction_real: torch.Tensor, image_real: torch.Tensor,
                prediction_real_pixel_wise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param image_real: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
        """
        # Calc gradient
        grad_real, = autograd.grad(
            outputs=(prediction_real.sum(), prediction_real_pixel_wise.sum())
            if prediction_real_pixel_wise is not None else prediction_real.sum(),
            inputs=image_real, create_graph=True)
        # Calc regularization
        regularization_loss = 0.5 * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return regularization_loss


class R2Regularization(nn.Module):
    """
    Implementation of the R2 GAN regularization.
    """

    def __init__(self):
        """
        Constructor method
        """
        # Call super constructor
        super(R2Regularization, self).__init__()

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}'.format(self.__class__.__name__)

    def forward(self, prediction_fake: torch.Tensor, image_fake) -> torch.Tensor:
        """
        Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param image_real: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
        """
        # Calc gradient
        grad_real = autograd.grad(outputs=prediction_fake.sum(), inputs=image_fake, create_graph=True)
        # Calc regularization
        regularization_loss = 0.5 * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return regularization_loss


class PathLengthRegularization(nn.Module):
    """
    Module implements the path length style gan regularization.
    """

    def __init__(self, decay: float = 0.01) -> None:
        """
        Constructor method
        :param decay: (float) Decay of the current mean path length
        :param weight: (float) Weight factor
        """
        # Call super constructor
        super(PathLengthRegularization, self).__init__()
        # Save parameter
        self.decay = decay
        # Init mean path length
        self.mean_path_length = torch.zeros(1, dtype=torch.float, requires_grad=False)

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return '{}'.format(self.__class__.__name__)

    def forward(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        :param grad: (torch.Tensor) Patch length grads
        :return: (Tuple[torch.Tensor, torch.Tensor]) Path length penalty and path lengths
        """
        # Reduce dims
        # Detach mean path length
        self.mean_path_length.detach_()
        # Get new path lengths
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1) + 1e-08).mean()
        # Mean path length to device
        self.mean_path_length = self.mean_path_length.to(grad.device)
        # Calc path length mean
        self.mean_path_length = self.mean_path_length + self.decay * (path_lengths.mean() - self.mean_path_length)
        # Get path length penalty
        path_length_penalty = torch.mean((path_lengths - self.mean_path_length) ** 2)
        return path_length_penalty, path_lengths


class TopK(nn.Module):
    """
    This class implements the top-k method proposed in:
    https://arxiv.org/pdf/2002.06224.pdf
    """

    def __init__(self, starting_iteration: int, final_iteration: int) -> None:
        """
        Constructor method
        :param starting_iteration: (bool) Number of iteration when to start with top-k training
        :param final_iteration: (bool) Number of iteration when to stop top-k training decrease
        """
        # Call super constructor
        super(TopK, self).__init__()
        # Save parameters
        self.starting_iteration = starting_iteration
        self.final_iteration = final_iteration
        self.iterations = 0

    def calc_v(self) -> float:
        """
        Method tracks the iterations and estimates v.
        :return: (float) v factor
        """
        # Update iterations
        self.iterations += 1
        if self.iterations <= self.starting_iteration:
            return 1.
        elif self.iterations >= self.final_iteration:
            return 0.5
        else:
            return 0.5 * (1. - float(self.iterations - self.starting_iteration)
                          / float(self.final_iteration - self.starting_iteration)) + 0.5

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :return: (torch.Tensor) Output tensor filtered by top-k
        """
        # Calc v
        v = self.calc_v()
        # Flatten input
        input = input.view(-1)
        # Apply top k
        output = torch.topk(input, k=max(1, int(input.shape[0] * v)))
        return output
