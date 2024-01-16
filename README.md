# Highlight-Removal
Summary of Publicly Available Highlight Removal Method

**Raise Issue or PR to add more**

## Dataset

[MIT Intrinsic Images](https://www.cs.toronto.edu/~rgrosse/intrinsic/)

[SSHR](https://github.com/fu123456/TSHRNet)

[SHIQ](https://github.com/fu123456/SHIQ)

[PSD](https://github.com/jianweiguo/SpecularityNet-PSD)

[kvasir](https://datasets.simula.no/kvasir/)

[WHU-Specular](https://github.com/fu123456/SHDNet)

[WHU-TRIIW](https://github.com/fu123456/SHDNet)

## Method

### Traditional

| Name                                                         | Year | Publication                                           | Code                                                         |
| ------------------------------------------------------------ | ---- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Tan et al.                                                   | 2005 | TPAMI                                                 | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Yoon et al.                                                  | 2006 | ICIP                                                  | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Shen et al.                                                  | 2008 | Pattern Recognition                                   | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Shen et al.                                                  | 2009 | Applied Optics                                        | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Yang et al.                                                  | 2010 | ECCV                                                  | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Shen et al.                                                  | 2013 | Applied Optics                                        | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Separating reflection components of textured surfaces using a single image | 2015 | TPAMI                                                 | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Akashi et al.                                                | 2016 | CVIU                                                  | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Nurutdinova et al.                                           | 2017 | VISAPP                                                | [C++](https://github.com/AlexandraPapadaki/Specularity-Shadow-and-Occlusion-Removal-for-Planar-Objects-in-Stereo-Case) |
| Real-Time High-Quality Specular Highlight Removal Using Efficient Pixel Clustering | 2018 | SIBGRAPI                                              | [C++](https://github.com/MarcioCerqueira/RealTimeSpecularHighlightRemoval) |
| SHR                                                          | 2019 | CGF                                                   | [MATLAB](https://github.com/fu123456/Specular_highlight_removal_for_real_world_images) |
| Yamamoto et al.                                              | 2019 | ITE Transactions on Media Technology and Applications | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Saha et al.                                                  | 2020 | IET Image Processing                                  | [MATLAB](https://github.com/7ZFG1/Combining-Highlight-Re-moval-and-Lowlight-Image-Enhancement-Technique-for-HDR-like-Image-Generation-) |
| Polar_HR                                                     | 2021 | TIP                                                   | [MATLAB](https://github.com/wsj890411/Polar_HR)              |

### Learning

| Name             | Year | Publication    | Code                                                         |
| ---------------- | ---- | -------------- | ------------------------------------------------------------ |
| SLRR             | 2018 | ECCV           | [Pytorch](https://github.com/dingguanglei/SLRR-SparseAndLowRankReflectionModel) |
| TASHR            | 2021 | PRCV           | [Pytorch](https://github.com/weizequan/TASHR)                |
| SpecularityNet   | 2021 | TMM            | [Pytorch](https://github.com/jianweiguo/SpecularityNet-PSD)  |
| JSHDR            | 2021 | CVPR           | [Pytorch](https://github.com/fu123456/SHIQ)                  |
| Unet-Transformer | 2022 | CVM            | [Pytorch](https://github.com/hzfengfengxia/specularityRemoval) |
| MG-CycleGAN      | 2022 | PRL            | [Pytorch](https://github.com/hootoon/MG-Cycle-GAN)           |
| TSRNet           | 2023 | ICCV           | [Pytorch](https://github.com/fu123456/TSHRNet)               |
| SHMGAN           | 2023 | Neurocomputing | [Tensorflow](https://github.com/Atif-Anwer/SHMGAN)           |
|                  | 2024 | ICASSP         | [Pytorch](https://github.com/LittleFocus2201/ICASSP2024)     |

## Metric

### Full-Reference

[torchmetrics](https://github.com/Lightning-AI/torchmetrics) for cuda calculation

**PSNR**

**SSIM**
