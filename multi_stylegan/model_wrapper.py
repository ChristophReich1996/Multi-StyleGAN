from typing import Optional, Union, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np
import random
from rtpt.rtpt import RTPT

from . import loss
from . import misc
from .u_net_2d_discriminator import generate_cut_mix_augmentation_data, generate_cut_mix_transformation_data


class ModelWrapper(object):
    """
    This class implements a model wrapper for the GAN and implements training and validation methods.
    """

    def __init__(self,
                 generator: Union[nn.Module, nn.DataParallel],
                 discriminator: Union[nn.Module, nn.DataParallel],
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer,
                 training_dataset: DataLoader,
                 data_logger: misc.Logger,
                 validation_metrics: Tuple[Callable, ...],
                 hyperparameters: Dict[str, Any],
                 trap_weights_map: Optional[torch.Tensor] = None,
                 generator_loss: nn.Module = loss.NonSaturatingLogisticGeneratorLoss(),
                 discriminator_loss: nn.Module = loss.NonSaturatingLogisticDiscriminatorLoss(),
                 discriminator_regularization_loss: nn.Module = loss.R1Regularization(),
                 cut_mix_augmentation_loss: nn.Module = loss.NonSaturatingLogisticDiscriminatorLossCutMix(),
                 cut_mix_regularization_loss: nn.Module = nn.MSELoss(reduction="mean"),
                 path_length_regularization: nn.Module = loss.PathLengthRegularization(),
                 generator_ema: Optional[Union[nn.Module, nn.DataParallel]] = None,
                 device: str = "cuda",
                 discriminator_learning_rate_schedule: Optional[object] = None) -> None:
        """
        Constructor method
        :param generator: (Union[nn.Module, nn.DataParallel]) Generator network
        :param discriminator: (Union[nn.Module, nn.DataParallel]) Generator network
        :param generator_optimizer: (torch.optim.Optimizer) Generator optimizer
        :param discriminator_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param training_dataset: (DataLoader) Training dataset
        :param data_logger: (misc.Logger) Custom data logger
        :param validation_metrics: (Tuple[Callable, ...]) Tuple of validation metrics to be utilized
        :param hyperparameters: (Dict[str, Any]) Hyperparameter dict
        :param trap_weights_map: (Optional[torch.Tensor]) Optional weights map for trap region
        :param generator_loss: (nn.Module) Generator loss function
        :param discriminator_loss: (nn.Module) Discriminator loss function
        :param discriminator_regularization_loss: (nn.Module) Dis. regularization loss
        :param cut_mix_augmentation_loss: (nn.Module) Cut mix augmentation loss
        :param cut_mix_regularization_loss: (nn.Module) Cut mix regularization loss
        :param path_length_regularization: (nn.Module) Path length regularization loss
        :param generator_ema: (Optional[Union[nn.Module, nn.DataParallel]]) Generator for EMA
        :param device: (str) Device to be utilized (only cuda supported!)
        :param discriminator_learning_rate_schedule: (Optional[object]) Optional discriminator lr schedule
        """
        # Save parameters
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.training_dataset = training_dataset
        self.data_logger = data_logger
        self.validation_metrics = validation_metrics
        self.trap_weights_map = trap_weights_map
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.discriminator_regularization_loss = discriminator_regularization_loss
        self.cut_mix_augmentation_loss = cut_mix_augmentation_loss
        self.cut_mix_regularization_loss = cut_mix_regularization_loss
        self.path_length_regularization = path_length_regularization
        self.device = device
        self.hyperparameters = hyperparameters
        self.discriminator_learning_rate_schedule = discriminator_learning_rate_schedule
        # Make copy of generator to perform ema
        if generator_ema is None:
            if isinstance(self.generator, nn.DataParallel):
                self.generator_ema = copy.deepcopy(self.generator.module.cpu()).to(self.device)
                self.generator_ema = nn.DataParallel(self.generator_ema)
            else:
                self.generator_ema = copy.deepcopy(self.generator.cpu()).to(self.device)
        else:
            self.generator_ema = generator_ema
        # Model into eval mode
        self.generator_ema.eval()
        # Get latent dimensions
        if isinstance(self.generator, nn.DataParallel):
            self.latent_dimensions = self.generator.module.latent_dimensions
        else:
            self.latent_dimensions = self.generator.latent_dimensions
        # Init best fid
        self.best_fvd = np.inf
        # Init validation input noise
        self.validation_input_noise = misc.get_noise(batch_size=15,
                                                     latent_dimension=self.latent_dimensions,
                                                     p_mixed_noise=1.0,
                                                     device="cuda")

    def train(self, epochs: int = 20, validate_after_n_epochs: int = 10,
              save_model_after_n_epochs: int = 5, resume_training: bool = False,
              top_k: bool = False) -> None:
        """
        Training method
        :param epochs: (int) Epochs to perform
        :param validate_after_n_epochs: (int) Validate generator after given number of epochs
        :param save_model_after_n_epochs: (int) Save models after a given number of epochs
        :param resume_training: (bool) If true training is resumed and cut mix and wrong order reg/aug used
        :param top_k: (bool) If true top-k is utilized
        """
        # Init top-k
        if top_k:
            top_k = loss.TopK(
                starting_iteration=int(self.hyperparameters["top_k_start"] * epochs * len(self.training_dataset)),
                final_iteration=int(self.hyperparameters["top_k_finish"] * epochs
                                    * len(self.training_dataset)))
            if resume_training:
                top_k.starting_iteration = 0
                top_k.final_iteration = 1
        else:
            top_k = nn.Identity()
        # Save parameters
        self.epochs = epochs
        # Init RTPT
        rtpt = RTPT(name_initials="CR", experiment_name="DeepFovea++", max_iterations=epochs)
        # Start RTPT
        rtpt.start()
        # Models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataset))
        # Main loop
        for self.epoch in range(epochs):
            # Models into training mode
            self.generator.train()
            self.discriminator.train()
            # Update RTPT
            rtpt.step()
            # Perform gan training
            self._gan_training(resume_training=resume_training, top_k=top_k)
            # Make validation plots
            with torch.no_grad():
                # Models into eval mode
                self.generator.eval()
                self.generator_ema.eval()
                # Make predictions
                if isinstance(self.generator_ema, nn.DataParallel):
                    predictions_ema = self.generator_ema.module(input=self.validation_input_noise,
                                                                randomize_noise=False)
                    predictions_ema_rand = self.generator_ema.module(input=self.validation_input_noise,
                                                                     randomize_noise=True)
                else:
                    predictions_ema = self.generator_ema(input=self.validation_input_noise, randomize_noise=False)
                    predictions_ema_rand = self.generator_ema(input=self.validation_input_noise, randomize_noise=True)
                if isinstance(self.generator, nn.DataParallel):
                    predictions = self.generator.module(input=self.validation_input_noise, randomize_noise=False)
                    predictions_rand = self.generator.module(input=self.validation_input_noise, randomize_noise=True)
                else:
                    predictions = self.generator(input=self.validation_input_noise, randomize_noise=False)
                    predictions_rand = self.generator(input=self.validation_input_noise, randomize_noise=True)
                # Save predictions
                self.data_logger.save_prediction(prediction=predictions_ema,
                                                 name="prediction_ema_{}".format(self.epoch + 1))
                self.data_logger.save_prediction(prediction=predictions_ema_rand,
                                                 name="prediction_ema_rand_{}".format(self.epoch + 1))
                self.data_logger.save_prediction(prediction=predictions,
                                                 name="prediction_{}".format(self.epoch + 1))
                self.data_logger.save_prediction(prediction=predictions_rand,
                                                 name="prediction_rand_{}".format(self.epoch + 1))
            # Perform gan validation
            if ((self.epoch + 1) % validate_after_n_epochs == 0):
                self.validation()
            # Save logs
            self.data_logger.save()
            # Save models and optimizers
            if ((self.epoch + 1) % save_model_after_n_epochs) == 0:
                self.data_logger.save_checkpoint(
                    file_name="checkpoint_{}.pt".format(self.epoch + 1),
                    checkpoint_dict={
                        "generator_ema": self.generator_ema.state_dict(),
                        "generator": self.generator.state_dict(),
                        "generator_optimizer": self.generator_optimizer.state_dict(),
                        "discriminator": self.discriminator.state_dict(),
                        "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                        "path_length_regularization": self.path_length_regularization.state_dict(),
                    }
                )
            # Perform learning rate schedule if utilized
            if self.discriminator_learning_rate_schedule is not None:
                self.discriminator_learning_rate_schedule.step()

    def validation(self) -> None:
        """
        GAN validation
        """
        # Clean cuda cache
        torch.cuda.empty_cache()
        # Set progress bar
        try:
            self.progress_bar.set_description("Validation")
        except AttributeError:
            print("Validation")
        # Model into eval mode
        self.generator_ema.eval()
        # Model to device
        self.generator_ema.to(self.device)
        # Perform validation
        for validation_metric in self.validation_metrics:
            # Get scores
            scores = validation_metric(generator=self.generator_ema, dataset=self.training_dataset)
            if isinstance(scores, np.float64) or isinstance(scores, float):
                score_bf = scores
                # Log scores
                self.data_logger.log_metric(metric_name=validation_metric.__class__.__name__ + "_bf", value=score_bf)
            else:
                if len(scores) == 3:
                    score_bf, score_gfp, score_rfp = scores
                    # Log scores
                    self.data_logger.log_metric(metric_name=validation_metric.__class__.__name__ + "_bf",
                                                value=score_bf)
                    self.data_logger.log_metric(metric_name=validation_metric.__class__.__name__ + "_gfp",
                                                value=score_gfp)
                    self.data_logger.log_metric(metric_name=validation_metric.__class__.__name__ + "_rfp",
                                                value=score_rfp)
                else:
                    score_bf, score_gfp = scores
                    # Log scores
                    self.data_logger.log_metric(metric_name=validation_metric.__class__.__name__ + "_bf",
                                                value=score_bf)
                    self.data_logger.log_metric(metric_name=validation_metric.__class__.__name__ + "_gfp",
                                                value=score_gfp)
            # Save best fid score
            if "FVD" in validation_metric.__class__.__name__:
                try:
                    if self.best_fvd > score_bf:
                        self.best_fvd = score_bf
                except RuntimeError:
                    pass

    def _gan_training(self, resume_training: bool = False,
                      top_k: nn.Module = nn.Identity()) -> None:
        """
        Gan training
        :param resume_training: (bool) If true training is resumed and cut mix and wrong order reg/aug used
        :param top_k: (nn.Module) Top-k module
        """
        # Main loop
        for real_images in self.training_dataset:  # type: torch.Tensor
            # Update progress bar
            self.progress_bar.update(n=1)
            # Data to device
            real_images = real_images.to(self.device)
            ############## Discriminator training ##############
            # Reset gradients
            self.discriminator_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            # Utilize no gradients for generator prediction
            with torch.no_grad():
                # Get noise input vector
                noise_input = misc.get_noise(batch_size=real_images.shape[0],
                                             latent_dimension=self.latent_dimensions,
                                             p_mixed_noise=self.hyperparameters["p_mixed_noise"],
                                             device=self.device)
                # Forward pass of generator
                fake_images: torch.Tensor = self.generator(input=noise_input)

            if self.epoch >= self.hyperparameters["wrong_order_start"] * self.epochs or resume_training:
                # Add one real image of wrong permutation to fake images
                fake_images = torch.cat(
                    [fake_images,
                     real_images[:max(1, int(self.hyperparameters["batch_factor_wrong_order"] * real_images.shape[0])),
                     :, misc.random_permutation(real_images.shape[2])]], dim=0)
            # Forward pass discriminator with real images
            real_prediction, real_prediction_pixel_wise = self.discriminator(real_images, is_real=True,
                                                                             is_cut_mix=False)
            # Forward pass discriminator with fake images
            fake_prediction, fake_prediction_pixel_wise = self.discriminator(fake_images, is_real=False,
                                                                             is_cut_mix=False)
            # Calc discriminator loss
            loss_discriminator_real, loss_discriminator_fake = self.discriminator_loss(real_prediction, fake_prediction)
            # Calc discriminator loss
            loss_discriminator_real_pixel_wise, loss_discriminator_fake_pixel_wise = \
                self.discriminator_loss(
                    real_prediction_pixel_wise, fake_prediction_pixel_wise,
                    weight=self.trap_weights_map
                    if self.hyperparameters["trap_weight"] * self.epochs <= self.epoch or resume_training else None)
            # Calc gradients
            (loss_discriminator_real + loss_discriminator_fake +
             loss_discriminator_real_pixel_wise + loss_discriminator_fake_pixel_wise).backward()
            # Clip discriminator gradients
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.)
            # Optimize discriminator
            self.discriminator_optimizer.step()
            # Log losses
            self.data_logger.log_metric(metric_name="loss_discriminator_real", value=loss_discriminator_real.item())
            self.data_logger.log_metric(metric_name="loss_discriminator_fake", value=loss_discriminator_fake.item())
            self.data_logger.log_metric(metric_name="loss_discriminator_real_pixel_wise",
                                        value=loss_discriminator_real_pixel_wise.item())
            self.data_logger.log_metric(metric_name="loss_discriminator_fake_pixel_wise",
                                        value=loss_discriminator_fake_pixel_wise.item())
            # Perform regularization
            if (self.progress_bar.n % self.hyperparameters["lazy_discriminator_regularization"]) == 0:
                # Reset gradients
                self.discriminator_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                # Set requires grad
                real_images.requires_grad = True
                # Make prediction
                real_prediction, real_prediction_pixel_wise = self.discriminator(real_images, is_real=False,
                                                                                 is_cut_mix=True)
                # Calc loss
                loss_discriminator_regularization = self.discriminator_regularization_loss(real_prediction, real_images,
                                                                                           real_prediction_pixel_wise)
                # Calc gradients
                (self.hyperparameters[
                     "w_discriminator_regularization_r1"] * loss_discriminator_regularization).backward()
                # Clip discriminator gradients
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.)
                # Optimize discriminator
                self.discriminator_optimizer.step()
                # Log augmentation loss
                self.data_logger.log_metric(
                    metric_name="loss_discriminator_regularization",
                    value=(loss_discriminator_regularization).item())
            # Perform cut mix regularization/augmentation
            if (random.random() <= ((0.5 / float(self.epochs)) * float(self.epoch))) \
                    or (resume_training and random.random() <= 0.5):
                # Reset gradients
                self.discriminator_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                # Init cut mix augmentation
                cut_mix_augmentation_images, cut_mix_augmentation_label = \
                    generate_cut_mix_augmentation_data(real_images, fake_images)
                # Make prediction
                _, cut_mix_augmentation_prediction = self.discriminator(cut_mix_augmentation_images, is_cut_mix=True)
                # Calc loss
                cut_mix_augmentation_loss_real, cut_mix_augmentation_loss_fake = \
                    self.cut_mix_augmentation_loss(cut_mix_augmentation_prediction, cut_mix_augmentation_label)
                # Calc gradients
                (self.hyperparameters["w_discriminator_regularization"] *
                 (cut_mix_augmentation_loss_real + cut_mix_augmentation_loss_fake)).backward()
                # Clip discriminator gradients
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.)
                # Optimize discriminator
                self.discriminator_optimizer.step()
                # Log augmentation loss
                self.data_logger.log_metric(
                    metric_name="loss_cut_mix_augmentation",
                    value=(cut_mix_augmentation_loss_real + cut_mix_augmentation_loss_fake).item())
                # Reset gradients
                self.discriminator_optimizer.zero_grad()
                # Init cut mix regularization
                cut_mix_regularization_images, cut_mix_regularization_label = \
                    generate_cut_mix_transformation_data(real_images.detach(), fake_images.detach(),
                                                         real_prediction_pixel_wise.detach(),
                                                         fake_prediction_pixel_wise.detach())
                # Make prediction
                _, cut_mix_regularization_prediction = self.discriminator(cut_mix_regularization_images,
                                                                          is_cut_mix=True)
                # Calc loss
                cut_mix_regularization_loss = self.cut_mix_regularization_loss(cut_mix_regularization_prediction,
                                                                               cut_mix_regularization_label)
                # Calc gradients
                (self.hyperparameters["w_discriminator_regularization"] * cut_mix_regularization_loss).backward()
                # Clip discriminator gradients
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.)
                # Optimize discriminator
                self.discriminator_optimizer.step()
                # Log regularization loss
                self.data_logger.log_metric(metric_name="loss_cut_mix_regularization",
                                            value=cut_mix_regularization_loss.item())
            ############## Generator training ##############
            # Reset gradients
            self.discriminator_optimizer.zero_grad()
            self.generator_optimizer.zero_grad()
            # Get noise input vector
            noise_input = misc.get_noise(batch_size=real_images.shape[0],
                                         latent_dimension=self.latent_dimensions,
                                         p_mixed_noise=self.hyperparameters["p_mixed_noise"],
                                         device=self.device)
            # Forward pass of generator
            fake_images = self.generator(input=noise_input)
            # Discriminator prediction
            fake_prediction, fake_prediction_pixel_wise = self.discriminator(fake_images, is_real=False,
                                                                             is_cut_mix=False)
            # Apply top k
            output = top_k(fake_prediction)
            # Case if top k is utilized
            if isinstance(output, tuple):
                # Unpack output
                fake_prediction, indexes = output
                # Apply indexes to pixel-wise prediction
                fake_prediction_pixel_wise = fake_prediction_pixel_wise[indexes]
            # Case if not top k is utilized
            else:
                fake_prediction = output
            # Calc generator loss
            loss_generator = self.generator_loss(fake_prediction)
            # Calc generator loss
            loss_generator_pixel_wise = self.generator_loss(fake_prediction_pixel_wise, weight=self.trap_weights_map
            if self.hyperparameters["trap_weight"] * self.epochs <= self.epoch or resume_training else None)
            # Calc gradients
            (loss_generator + loss_generator_pixel_wise).backward()
            # Clip generator gradients
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.)
            # Optimize generator
            self.generator_optimizer.step()
            # Log generator loss
            self.data_logger.log_metric(metric_name="loss_generator", value=loss_generator.item())
            # Log generator loss
            self.data_logger.log_metric(metric_name="loss_generator_pixel_wise", value=loss_generator_pixel_wise.item())
            # Perform regularization
            if (self.progress_bar.n % self.hyperparameters["lazy_generator_regularization"]) == 0:
                # Reset gradients
                self.discriminator_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                # Get input noise
                noise_input = misc.get_noise(
                    batch_size=max(1, int(self.hyperparameters["batch_size_shrink_path_length_regularization"] *
                                          real_images.shape[0])),
                    latent_dimension=self.latent_dimensions,
                    p_mixed_noise=self.hyperparameters["p_mixed_noise"],
                    device=self.device)
                # Predict images and latents
                path_length_grads = self.generator(input=noise_input, return_path_length_grads=True)
                # Calc regularization loss
                loss_path_length_regularization, path_length = self.path_length_regularization(path_length_grads)
                # Calc gradients
                loss_path_length_regularization_ = (
                        self.hyperparameters["w_generator_regularization"] * loss_path_length_regularization)
                loss_path_length_regularization_.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.)
                # Optimize
                self.generator_optimizer.step()
                # Log regularization loss
                self.data_logger.log_metric(metric_name="path_length", value=path_length.mean().item())
                self.data_logger.log_metric(metric_name="loss_path_length_regularization",
                                            value=loss_path_length_regularization.item())
            # Perform ema
            misc.exponential_moving_average(model_ema=self.generator_ema, model_train=self.generator)
            # Set progress bar description
            self.progress_bar.set_description("Loss D={:.3f}, Loss G={:.3f}, Best FVD={:.3f}".format(
                (loss_discriminator_fake + loss_discriminator_real +
                 loss_discriminator_fake_pixel_wise + loss_discriminator_real_pixel_wise).item(),
                (loss_generator + loss_generator_pixel_wise).item(), self.best_fvd))
