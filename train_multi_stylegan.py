from argparse import ArgumentParser

# Manage command line arguments
parser = ArgumentParser()
parser.add_argument("--cuda_devices", default="0, 1, 2, 3", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--batch_size", default=24, type=int,
                    help="Batch size to be utilized while training.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If set data parallel is utilized.")
parser.add_argument("--epochs", default=100, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--lr_generator", default=2e-04, type=float,
                    help="Learning rate of the generator network.")
parser.add_argument("--lr_discriminator", default=6e-04, type=float,
                    help="Learning rate of the discriminator network.")
parser.add_argument("--path_to_data", default="./60x_10BF_200GFP_200RFP20_3Z_10min",
                    type=str, help="Path to dataset.")
parser.add_argument("--load_checkpoint", default="", type=str,
                    help="Path to checkpoint to be loaded.")
parser.add_argument("--resume_training", default=False, action="store_true",
                    help="Binary flag. If set training is resumed and so cut mix aug/reg and wrong order aug is used.")
parser.add_argument("--no_top_k", default=False, action="store_true",
                    help="Binary flag. If set no top-k is utilized.")
parser.add_argument("--no_ada", default=False, action="store_true",
                    help="Binary flag. If set no adaptive discriminator augmentation is utilized.")
# Get arguments
args = parser.parse_args()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import copy

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import multi_stylegan
import dataset

if __name__ == '__main__':
    # Init models
    print("Init models")
    generator = multi_stylegan.MultiStyleGANGenerator(config=multi_stylegan.multi_style_gan_generator_config).cuda()
    discriminator = multi_stylegan.MultiStyleGANDiscriminator(config=multi_stylegan.u_net_2d_discriminator_config,
                                                              no_rfp=True).cuda()
    hyperparameters = multi_stylegan.generation_hyperparameters
    print("Generator parameters:", sum([p.numel() for p in generator.parameters()]))
    print("Discriminator parameters:", sum([p.numel() for p in discriminator.parameters()]))
    # Init optimizers
    print("Init optimizers")
    generator_optimizer = torch.optim.Adam(
        generator.get_parameters(lr_main=args.lr_generator,
                                 lr_style=args.lr_generator / 100), betas=hyperparameters["betas"])
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_discriminator,
                                               betas=hyperparameters["betas"])
    # Init dataset
    print("Init dataset")
    training_dataset = DataLoader(
        dataset.TFLMDatasetGAN(path=args.path_to_data, no_rfp=True),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.batch_size,
        pin_memory=True)
    # Init path length reg. loss
    path_length_regularization = multi_stylegan.PathLengthRegularization()
    # Apply data parallel
    if args.data_parallel:
        print("Init data parallel")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    if not args.no_ada:
        discriminator = multi_stylegan.AdaptiveDiscriminatorAugmentation(discriminator=discriminator)
    # Load checkpoint if utilized
    if args.load_checkpoint != "":
        print("Load checkpoint")
        checkpoint = torch.load(args.load_checkpoint)
        # Get modules and optimizers
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        path_length_regularization.load_state_dict(checkpoint["path_length_regularization"])
        # Init generator ema
        generator_ema = copy.deepcopy(generator.cpu()).cuda()
        generator_ema.load_state_dict(checkpoint["generator_ema"])
        # Apply data parallel to generator ema if utilized
        if args.data_parallel:
            generator_ema = nn.DataParallel(generator_ema)
        # Ensure generator is on gpu
        generator.cuda()
    # Init data logger
    data_logger = multi_stylegan.Logger()
    data_logger.log_hyperparameter(hyperparameter_dict=hyperparameters)
    # Init model wrapper
    print("Init model wrapper")
    model_wrapper = multi_stylegan.ModelWrapper(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        training_dataset=training_dataset,
        data_logger=data_logger,
        validation_metrics=(
            multi_stylegan.FID(device="cuda", batch_size=args.batch_size, data_parallel=args.data_parallel,
                               no_rfp=True, no_gfp=False),
            multi_stylegan.FVD(device="cuda", batch_size=args.batch_size, data_parallel=args.data_parallel,
                               no_rfp=True, no_gfp=False),
            multi_stylegan.IS(device="cuda", batch_size=args.batch_size, data_parallel=args.data_parallel,
                              no_rfp=True, no_gfp=False)),
        hyperparameters=hyperparameters,
        path_length_regularization=path_length_regularization,
        discriminator_learning_rate_schedule=None,
        trap_weights_map=None
    )
    print("Start training")
    model_wrapper.train(epochs=args.epochs, resume_training=args.resume_training, top_k=not args.no_top_k)
