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

The code is tested on [PyTorch 1.7.1](https://pytorch.org/get-started/previous-versions/) and CUDA 11.1! Using other 
PyTorch and CUDA version newer than PyTorch 1.7.0 and CUDA 11.0 should also be possible.

## Usage

## Trained Models

## Data

**Our proposed time-lapse fluorescent microscopy is available at [this url](https://arxiv.org/pdf/2106.08285.pdf).**

The dataset includes 9696 images structured in sequences of both brightfield and green fluorescent protein (GFP) channels at a resolution of 256 × 256.

## Acknowledgements

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
