# Multi-StyleGAN: Towards Image-Based Simulation of Time-Lapse Live-Cell Microscopy
[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2106.08285-B31B1B.svg)](https://arxiv.org/abs/2106.08285)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChristophReich1996/Multi-StyleGAN/blob/master/LICENSE)

**[Christoph Reich*](https://github.com/ChristophReich1996), [Tim Prangemeier*](https://www.bcs.tu-darmstadt.de/bcs_team/prangemeiertim.en.jsp), [Christian Wildner](https://www.bcs.tu-darmstadt.de/bcs_team/wildnerchristian.en.jsp) & [Heinz Koeppl](https://www.bcs.tu-darmstadt.de/bcs_team/koepplheinz.en.jsp)**<br/>
*Christoph Reich and Tim Prangemeier - both authors contributed equally

## [Project Page]() | [Paper](https://arxiv.org/abs/2106.08285) | [Dataset]() | [Video]() | [Slides]() |

<p align="center">
  <img src="/github/latent_space_interpolation.gif"  alt="1" width = 288px height = 192px >
</p>
  
<p align="center">
  This repository includes the <b>official</b> and <b>maintained</b> <a href="https://pytorch.org/">PyTorch</a> implementation of the paper <a href="https://arxiv.org/abs/2106.08285"> Multi-StyleGAN: Towards Image-Based Simulation of Time-Lapse Live-Cell Microscopy</a>
</p>

## Abstract
*Time-lapse fluorescent microscopy (TLFM) combined with
predictive mathematical modelling is a powerful tool to study the inherently dynamic processes of life on the single-cell level. Such experiments
are costly, complex and labour intensive. A complimentary approach
and a step towards completely in silico experiments, is to synthesise
the imagery itself. Here, we propose Multi-StyleGAN as a descriptive
approach to simulate time-lapse fluorescence microscopy imagery of living cells, based on a past experiment. This novel generative adversarial
network synthesises a multi-domain sequence of consecutive timesteps.
We showcase Multi-StyleGAN on imagery of multiple live yeast cells in
microstructured environments and train on a dataset recorded in our laboratory. The simulation captures underlying biophysical factors and time
dependencies, such as cell morphology, growth, physical interactions, as
well as the intensity of a fluorescent reporter protein. An immediate application is to generate additional training and validation data for feature
extraction algorithms or to aid and expedite development of advanced
experimental techniques such as online monitoring or control of cells.*

**If you find this research useful in your work, please cite our paper:**

```bibtex
@inproceedings{Reich2021,
    title={{Multi-StyleGAN: Towards Image-Based Simulation of Time-Lapse Live-Cell Microscopy}},
    author={Reich, Christoph and Prangemeier, Tim and Wildner, Christian and Koeppl, Heinz},
    booktitle={{International Conference on Medical image computing and computer-assisted intervention (in press)}},
    year={2021},
    organization={Springer}
}
```

## Method

<img src="/github/Multi-StyleGAN.png"  alt="1" width = 617px height = 176px ><br/>
**Figure 1.** Architecture of Multi-StyleGAN. The style mapping network <img src="https://render.githubusercontent.com/render/math?math=f"> (in purple)
transforms the input noise vector <img src="https://render.githubusercontent.com/render/math?math=z\sim \mathcal{N}_{512}(0, 1)"> into a latent vector <img src="https://render.githubusercontent.com/render/math?math=w\in\mathcal{W}">, which in
turn is incorporated into each stage of the generator by three dual-style-convolutional
blocks. The generator predicts a sequence of three consecutive images for both
the brightfield and green fluorescent protein channels. The U-Net discriminator distinguishes between real and
a fake input sequences by making both a scalar and a pixel-wise real/fake prediction.
Standard residual discriminator blocks in gray and non-local blocks in blue.

<img src="/github/Dual-styled-convolutional_block.png"  alt="1" width = 451px height = 221px ><br/>
**Figure 2.** Dual-styled-convolutional block of the Multi-StyleGAN. The incoming latent
vector w is transformed into the style vector s by a linear layer. This style vector modulates (mod) the convolutional weights <img src="https://render.githubusercontent.com/render/math?math=\theta_{b}"> and <img src="https://render.githubusercontent.com/render/math?math=\theta_{g}">, which are optionally demodulated
(demod) before convolving the (optionally bilinearly upsampled) incoming features
of the previous block. Learnable biasses (<img src="https://render.githubusercontent.com/render/math?math=b_{b}"> and <img src="https://render.githubusercontent.com/render/math?math=b_{g}">) and channel-wise Gaussian noise
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}">) scaled by a learnable constant (cb and cg), are added to the features. The final
output features are obtained by applying a leaky ReLU activation.

## Results

<img src="/github/prediction_ema_100_bf_0.png"  alt="1" width = 288px height = 96px >   <img src="/github/prediction_ema_100_bf_12.png"  alt="1" width = 288px height = 96px ><br/>
<img src="/github/prediction_ema_100_gfp_0.png"  alt="1" width = 288px height = 96px >  <img src="/github/prediction_ema_100_gfp_12.png"  alt="1" width = 288px height = 96px ><br/>
**Figure 3.** Samples generated by Multi-StyleGAN. Brightfield channel on the top and green fluorescent protein on the bottom.<br/>

**Table 1.** Evaluation metrics for Multi-StyleGAN and baselines.
| Model | FID (BF) <img src="https://render.githubusercontent.com/render/math?math=\downarrow"> | FVD (BF) <img src="https://render.githubusercontent.com/render/math?math=\downarrow"> | FID (BF) <img src="https://render.githubusercontent.com/render/math?math=\downarrow"> | FVD (BF) <img src="https://render.githubusercontent.com/render/math?math=\downarrow"> |
| --- | --- | --- | --- | --- |
| Multi-StyleGAN | **33.3687** | **4.4632** | **207.8409** | **30.1650** |
| StyleGAN 2 3d + ADA + U-Net dis. | 200.5408 | 45.6296 | 224.7860 | 35.2169 |
| StyleGAN 2 + ADA + U-Net dis. | 76.0344 | 14.7509 | 298.7545 | 31.4771 |

## Dependencies

All required Python packages can be installed by:

```shell script
pip install -r requirements.txt
```

To install the necessary custom CUDA extensions adapted from [StyleGAN 2](https://github.com/NVlabs/stylegan2) [1] run:
 
```shell script
cd multi_stylegan/op_static
python setup.py install
```

The code is tested with [PyTorch 1.8.1](https://pytorch.org/get-started/locally/) and CUDA 11.1 on Ubuntu with Python 3.6! 
Using other PyTorch and CUDA version newer than [PyTorch 1.7.0](https://pytorch.org/get-started/previous-versions/) and 
CUDA 10.1 should also be possible. Please note using a different PyTorch version eventually requires a different version
of [Kornia](https://kornia.github.io/) or [Torchvision](https://pytorch.org/vision/stable/index.html).

## Data

**Our proposed time-lapse fluorescent microscopy is available at [this url](https://arxiv.org/pdf/2106.08285.pdf).**

The dataset includes 9696 images structured in sequences of both brightfield and green fluorescent protein (GFP) channels at a resolution of 256 × 256.

## Trained Model

**The checkpoint of our trained Multi-StyleGAN is available at [this url]().**

The checkpoint (PyTorch state dict) includes the EMA generator weights (`"generator_ema"`), the generator weights 
(`"generator"`), the generator optimizer state (`"generator_optimizer"`), the discriminator weights (`"discriminator"`),
the discriminator optimizer state (`"discriminator_optimizer"`), and the path-length regularization states 
(`"path_length_regularization"`)

## Usage

To train Multi-StyleGAN in the proposed setting run the following command:

```shell script
 python -W ingore train_gan.py --cuda_devices "0, 1, 2, 3" --data_parallel --path_to_data "60x_10BF_200GFP_200RFP20_3Z_10min"
```

Dataset path and cuda devices may differ on other systems!
To perform training runs with different settings use the command line arguments of the [`train_gan.py`](train_gan.py) file.
The [`train_gan.py`](train_gan.py) takes the following command line arguments:

|Argument | Default value | Info|
|--- | --- | ---|
| --cuda_devices (str) | `"0, 1, 2, 3"` | String of cuda device indexes to be used. |
| --batch_size (int) | `24` | Batch size to be utilized while training. |
| --data_parallel (binary flag) | False | Binary flag. If set data parallel is utilized. |
| --epochs (int) | 100 | Number of epochs to perform while training. |
| --lr_generator (float) | `2e-04` | Learning rate of the generator network. |
| --lr_discriminator (float) | `6e-04` | Learning rate of the discriminator network. |
| --path_to_data (str) | `"./60x_10BF_200GFP_200RFP20_3Z_10min"` | Path to dataset. |
| --load_checkpoint (str) | `""` | Path to checkpoint to be loaded. If `""` no loading is performed |
| --resume_training (binary flag) | False | Binary flag. If set training is resumed and so cut mix aug/reg and wrong order aug is used. |
| --no_top_k (binary flag) | False | Binary flag. If set no top-k is utilized. |
| --no_ada (binary flag) | False | Binary flag. If set no adaptive discriminator augmentation is utilized. |

To generate samples of the trained Multi-StyleGAN use the [`get_gan_samples.py`](scripts/get_gan_samples.py) script.

```shell script
 python -W ingore scripts/get_gan_samples.py --cuda_devices "0" --load_checkpoint "checkpoint_100.pt"
```

This script takes the following command line arguments:

|Argument | Default value | Info|
|--- | --- | ---|
| --cuda_devices (str) | `"0"` | String of cuda device indexes to be used. |
| --samples (int) | `100` | Number of samples to be generated. |
| --load_checkpoint (str) | `"checkpoint_100.pt"` | Path to checkpoint to be loaded. |

To generate a latent space interpolation use the [`gan_latent_space_interpolation.py`](scripts/gan_latent_space_interpolation.py) script.
Generating the final `.mp4` video `ffmpeg` is required.

```shell script
 python -W ingore scripts/gan_latent_space_interpolation.py --cuda_devices "0" --load_checkpoint "checkpoint_100.pt"
```

This script takes the following command line arguments:

|Argument | Default value | Info|
|--- | --- | ---|
| --cuda_devices (str) | `"0"` | String of cuda device indexes to be used. |
| --load_checkpoint (str) | `"checkpoint_100.pt"` | Path to checkpoint to be loaded. |

## Acknowledgements

We thank [Markus Baier](https://www.bcs.tu-darmstadt.de/bcs_team/index.en.jsp) for aid with the computational setup, 
[Klaus-Dieter Voss](https://www.bcs.tu-darmstadt.de/bcs_team/index.en.jsp) for aid with the microfluidics 
fabrication, and Tim Kircher, [Tizian Dege](https://github.com/TixXx1337), and 
[Florian Schwald](https://github.com/FlorianSchwald59) for aid with the data preparation.

We also thank [piergiaj](https://github.com/piergiaj) for providing a [PyTorch i3d](https://github.com/piergiaj/pytorch-i3d) 
implementation and trained models, which we used to compute the FVD score. The used code is indicated and is available 
under the [original licence](https://github.com/piergiaj/pytorch-i3d/blob/master/LICENSE.txt).

## References

```bibtex
[1] @inproceedings{Karras2020,
    title={Analyzing and improving the image quality of stylegan},
    author={Karras, Tero and Laine, Samuli and Aittala, Miika and Hellsten, Janne and Lehtinen, Jaakko and Aila, Timo},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={8110--8119},
    year={2020}
}
```
