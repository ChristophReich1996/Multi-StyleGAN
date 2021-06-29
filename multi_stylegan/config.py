from typing import Dict, Any

import math

# U-Net 2D discriminator hyperparameters for resolution
u_net_2d_discriminator_config: Dict[str, Any] = {
    # Set encoder channels
    "encoder_channels": ((3, 128), (128, 256), (256, 384), (384, 768), (768, 1024)),
    # Set decoder channels
    "decoder_channels": ((1024, 768), (768, 384), (384, 256), (256, 128)),
    # Utilize fft input
    "fft": False,
}

# StyleGAN 2 2D generator hyperparameters for resolution
twin_style_gan_2_2d_generator_config: Dict[str, Any] = {
    # Channels utilized in each resolution stage
    "channels": (512, 512, 512, 512, 512, 512, 512),
    # Channel factor
    "channel_factor": 1,
    # Number of latent dimensions
    "latent_dimensions": 512,
    # Depth of the style mapping network
    "depth_style_mapping": 8,
    # Starting resolution
    "starting_resolution": (4, 4)
}

# Additional hyperparameters
generation_hyperparameters: Dict[str, Any] = {
    # Probability of mixed noise input
    "p_mixed_noise": 0.9,
    # Lazy generator regularization factor
    "lazy_generator_regularization": 16,
    # Weights factor for generator regularization
    "w_generator_regularization": math.log(2) / ((256 ** 2) * (math.log(256) - math.log(2))),
    # Lazy discriminator regularization factor
    "lazy_discriminator_regularization": 16,
    # Weights factor for discriminator R1 regularization
    "w_discriminator_regularization_r1": 10.0,
    # Weights factor for discriminator regularization
    "w_discriminator_regularization": 4.0,
    # Fraction of batch size of wrongly ordered samples
    "batch_factor_wrong_order": 1. / 4.,
    # Batch size for path length regularization
    "batch_size_shrink_path_length_regularization": 2. / 4.,
    # Beta for optimizers
    "betas": (0.0, 0.999),
    # Factor of total training steps when top-k should be started
    "top_k_start": 1. / 4.,
    # Factor of total training steps when top-k should be finished with v=0.5
    "top_k_finish": 3. / 4.,
    # Factor of total training steps when wrong time order is utilized
    "wrong_order_start": 3. / 4.,
    # Factor of total training epochs when to applied trap region weights map
    "trap_weight": 1. / 4.
}
