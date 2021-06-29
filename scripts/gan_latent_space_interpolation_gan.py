# Script to compute test metrics for GAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import kornia
import torchvision
import numpy as np
import os

from generation import TwinGenerator2D, style_gan_2_2d_generator_config

torch.random.manual_seed(1996)

if __name__ == '__main__':
    with torch.no_grad():
        # Init and load generator
        generator = nn.DataParallel(TwinGenerator2D(config=style_gan_2_2d_generator_config).cuda())
        generator.load_state_dict(
            torch.load(
                "/home/creich/Neural_Simulation_of_TLFM_Experiments/trained_models/25_01_2021__09_20_41_gan/checkpoint_100.pt")[
                "generator_ema"])
        # Generator into eval mode
        generator.eval()
        # Init latent vector
        noise_input = torch.randn(16, 512, dtype=torch.float32, device="cuda")
        noise_input = F.interpolate(noise_input.permute(1, 0).unsqueeze(dim=1), size=(100 * 16),
                                    mode="linear", align_corners=True).squeeze(dim=1).permute(1, 0)
        noise_input = noise_input.reshape(noise_input.shape[0] // 32, 32, 512)
        # Make predictions
        samples = []
        for index in tqdm(range(noise_input.shape[0])):
            samples.append(generator(input=noise_input[index], randomize_noise=False).cpu())
        samples = torch.cat(samples, dim=0)
        # Get BF and GFP
        samples_bf = samples[:, 0].permute(0, 2, 3, 1)
        samples_bf = torch.cat([samples_bf[..., index] for index in range(samples_bf.shape[-1])], dim=-1)
        samples_bf = samples_bf.unsqueeze(dim=1).repeat_interleave(repeats=3, dim=1)
        samples_gfp = samples[:, 1].permute(0, 2, 3, 1)
        samples_gfp = torch.cat([samples_gfp[..., index] for index in range(samples_gfp.shape[-1])], dim=-1)
        samples_gfp = samples_gfp.unsqueeze(dim=1).repeat_interleave(repeats=3, dim=1)
        samples_gfp[:, 0] = 0.0
        samples_gfp[:, 2] = 0.0
        # Construct video
        video = torch.cat([samples_bf, samples_gfp], dim=2)
        # Save frames
        for index in range(video.shape[0]):
            torchvision.utils.save_image(tensor=video[index][None],
                                         fp="video_gan_2/frame_{}.png".format(str(index).zfill(5)))
        # Make video
        os.system(
            "ffmpeg -r 60 -f image2 -s 768x512 -i video_gan_2/frame_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p video_gan_2.mp4")
