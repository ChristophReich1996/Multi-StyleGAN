# Script to compute test metrics for GAN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

from generation import IS, FID, FVD, FIDCellDETR, TwinGenerator2D, twin_style_gan_2_2d_generator_config, Generator2D, \
    style_gan_2_2d_generator_config, Generator3D, style_gan_2_3d_generator_config
from dataset import TFLMDatasetGAN

# torch.random.manual_seed(1904)

if __name__ == '__main__':
    with torch.no_grad():
        # Init and load generator
        """
        generator = nn.DataParallel(TwinGenerator2D(config=twin_style_gan_2_2d_generator_config).cuda())
        generator.load_state_dict(
            torch.load(
                "/home/creich/Neural_Simulation_of_TLFM_Experiments/trained_models/25_01_2021__09_20_41_gan/checkpoint_100.pt")[
                "generator_ema"])
        """
        """
        generator = nn.DataParallel(Generator3D(config=style_gan_2_3d_generator_config).cuda())
        generator.load_state_dict(
            torch.load(
                "/home/creich/Neural_Simulation_of_TLFM_Experiments/experiments/02_03_2021__18_58_37/models/checkpoint_55.pt")[
                "generator_ema"])
        """
        generator = nn.DataParallel(Generator2D(config=style_gan_2_2d_generator_config).cuda())
        generator.load_state_dict(
            torch.load(
                "/home/creich/Neural_Simulation_of_TLFM_Experiments/experiments/03_03_2021__08_57_28/models/checkpoint_40.pt")[
                "generator_ema"])
        generator.eval()
        # Init dataset
        dataset = DataLoader(
            TFLMDatasetGAN(path="/home/creich/60x_10BF_200GFP_200RFP20_3Z_10min_cropped_checked", no_rfp=True),
            shuffle=False, batch_size=32, drop_last=True, pin_memory=True, num_workers=32)
        # Init metrics
        fvd_metric = FVD(device="cuda", batch_size=32, data_parallel=True,
                         no_rfp=True, no_gfp=False, data_samples=32 * len(dataset),
                         network_path="/home/creich/Neural_Simulation_of_TLFM_Experiments/generation/pretrained_i3d/rgb_imagenet.pt")
        fid_metric = FID(device="cuda", batch_size=32, data_parallel=True, data_samples=32 * len(dataset),
                         no_rfp=True, no_gfp=False)
        fid_metric_cd = FIDCellDETR(device="cuda", batch_size=32, data_parallel=True, data_samples=32 * len(dataset),
                                    no_rfp=True, no_gfp=False)
        is_metric = IS(device="cuda", batch_size=32, data_parallel=True, data_samples=32 * len(dataset),
                       no_rfp=True, no_gfp=False)
        # Calc scores
        number_of_validations:int = 1
        for _ in range(number_of_validations):
            print("FID CD =" + str(fid_metric_cd(generator=generator, dataset=dataset)))
        for _ in range(number_of_validations):
            print("FVD =" + str(fvd_metric(generator=generator, dataset=dataset)))
        for _ in range(number_of_validations):
            print("IS =" + str(is_metric(generator=generator)))
        for _ in range(number_of_validations):
            print("FID =" + str(fid_metric(generator=generator, dataset=dataset)))