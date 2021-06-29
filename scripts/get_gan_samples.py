# Script to generate sample sequences of the trained generator
from argparse import ArgumentParser

# Manage command line arguments
parser = ArgumentParser()
parser.add_argument("--cuda_devices", default="0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--samples", default=100, type=int,
                    help="Number of samples to be generated.")
parser.add_argument("--load_checkpoint", default="checkpoint_100.pt", type=str,
                    help="Path to checkpoint to be loaded.")

# Get arguments
args = parser.parse_args()

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import os

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

from multi_stylegan import MultiStyleGANGenerator, multi_style_gan_generator_config
from multi_stylegan.misc import get_noise

# torch.random.manual_seed(1904)

if __name__ == '__main__':
    with torch.no_grad():
        # Init and load generator
        generator = nn.DataParallel(MultiStyleGANGenerator(config=multi_style_gan_generator_config).cuda())
        generator.load_state_dict(torch.load(args.load_checkpoint)["generator_ema"])
        # Genrator into eval mode
        generator.eval()
        # Make sequences
        for index in tqdm(range(args.samples)):
            # Get noise input
            noise_input = get_noise(batch_size=1, device="cuda", p_mixed_noise=0.0, latent_dimension=512)
            # Generate initial sequence
            sequence = generator(noise_input)
            # Get bf and gfp sequence
            bf_sequence = sequence[:, 0:1]
            gfp_sequence = sequence[:, 1:2]
            # Make bf and gfp to rgb
            bf_sequence = bf_sequence.repeat_interleave(3, dim=1)
            gfp_sequence = gfp_sequence.repeat_interleave(3, dim=1)
            gfp_sequence[:, 0] = 0.0
            gfp_sequence[:, 2] = 0.0
            # Reshape sequences
            bf_sequence = bf_sequence[0].permute(1, 0, 2, 3)
            gfp_sequence = gfp_sequence[0].permute(1, 0, 2, 3)
            # Save sequences
            torchvision.utils.save_image(tensor=bf_sequence,
                                         fp="sample_bf_{}.png".format(index), nrow=bf_sequence.shape[0],
                                         padding=0)
            torchvision.utils.save_image(tensor=gfp_sequence,
                                         fp="sample_gfp_{}.png".format(index), nrow=gfp_sequence.shape[0],
                                         padding=0)
