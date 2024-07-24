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

[EndoSRR](https://github.com/Tobyzai/EndoSRR)

## Method

### Traditional

| Name                                                         | Year | Publication                                           | Code                                                         |
| ------------------------------------------------------------ | ---- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Highlight removal by illumination-constrained inpainting     | 2003 | ICCV                                                  | [MATLAB](https://github.com/Kanvases/Highlight-Removal-by-Illumination-Constrained-Inpainting) |
| Tan et al.                                                   | 2005 | TPAMI                                                 | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Yoon et al.                                                  | 2006 | ICIP                                                  | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Shen et al.                                                  | 2008 | Pattern Recognition                                   | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Shen et al.                                                  | 2009 | Applied Optics                                        | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Yang et al.                                                  | 2010 | ECCV                                                  | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Automatic segmentation and inpainting of specular highlights for endoscopic imaging | 2010 | Journal on Image and Video Processing                 | [MATLAB](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image) |
| Detection and correction of specular reflections for automatic surgical tool segmentation in thoracoscopic images | 2011 | Machine Vision and Applications                       | [MATLAB](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image) |
| Automatic detection and inpainting of specular reflections for colposcopic images | 2011 | Open Computer Science                                 | [MATLAB](https://github.com/jiemojiemo/some_specular_detection_and_inpainting_methods_for_endoscope_image) |
| Shen et al.                                                  | 2013 | Applied Optics                                        | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| HDR image generation from LDR image with highlight removal   | 2015 | ICMEW                                                 | [Python](https://github.com/EthanWooo/Huo15)                 |
| Akashi et al.                                                | 2016 | CVIU                                                  | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Nurutdinova et al.                                           | 2017 | VISAPP                                                | [C++](https://github.com/AlexandraPapadaki/Specularity-Shadow-and-Occlusion-Removal-for-Planar-Objects-in-Stereo-Case) |
| Real-Time High-Quality Specular Highlight Removal Using Efficient Pixel Clustering | 2018 | SIBGRAPI                                              | [C++](https://github.com/MarcioCerqueira/RealTimeSpecularHighlightRemoval) |
| SHR                                                          | 2019 | CGF                                                   | [MATLAB](https://github.com/fu123456/Specular_highlight_removal_for_real_world_images) |
| Yamamoto et al.                                              | 2019 | ITE Transactions on Media Technology and Applications | [MATLAB](https://github.com/vitorsr/SIHR)                    |
| Specular Reflections Removal for Endoscopic Image Sequences With Adaptive-Rpca Decomposition | 2020 | TMI                                                   |                                                              |
| Highlight removal for endoscopic images based on accelerated adaptive non-convex RPCA decomposition | 2023 | CMPB                                                  |                                                              |
| Saha et al.                                                  | 2020 | IET Image Processing                                  | [MATLAB](https://github.com/7ZFG1/Combining-Highlight-Re-moval-and-Lowlight-Image-Enhancement-Technique-for-HDR-like-Image-Generation-) |
| Polar_HR                                                     | 2021 | TIP                                                   | [MATLAB](https://github.com/wsj890411/Polar_HR)              |

### Learning

| Name             | Year | Publication                                                  | Code                                                         |
| ---------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SLRR             | 2018 | ECCV                                                         | [Pytorch](https://github.com/dingguanglei/SLRR-SparseAndLowRankReflectionModel) |
| Spec-CGAN        | 2020 | Image and Vision Computing                                   | [Tensorflow](https://github.com/msiraj83/SPEC-CGAN)          |
| TASHR            | 2021 | PRCV                                                         | [Tensorflow](https://github.com/weizequan/TASHR)             |
| SpecularityNet   | 2021 | TMM                                                          | [Pytorch](https://github.com/jianweiguo/SpecularityNet-PSD)  |
| JSHDR            | 2021 | CVPR                                                         | [Pytorch](https://github.com/fu123456/SHIQ)                  |
| Liang et al.     | 2021 | Optics Express                                               | [Pytorch](https://github.com/Deepyanyuan/FaceIntrinsicDecomposition) |
| Unet-Transformer | 2022 | CVM                                                          | [Pytorch](https://github.com/hzfengfengxia/specularityRemoval) |
| MG-CycleGAN      | 2022 | PRL                                                          | [Pytorch](https://github.com/hootoon/MG-Cycle-GAN)           |
| TSRNet           | 2023 | ICCV                                                         | [Pytorch](https://github.com/fu123456/TSHRNet)               |
| SHMGAN           | 2023 | Neurocomputing                                               | [Tensorflow](https://github.com/Atif-Anwer/SHMGAN)           |
| CycleSTTN        | 2023 | MICCAI                                                       | [Pytorch](https://github.com/RemaDaher/CycleSTTN)            |
| Endo-STTN        | 2023 | MIA                                                          | [Pytorch](https://github.com/endomapper/Endo-STTN)           |
|                  | 2024 | ICASSP                                                       | [Pytorch](https://github.com/LittleFocus2201/ICASSP2024)     |
| EndoSRR          | 2024 | International Journal of Computer Assisted Radiology and Surgery | [Pytorch](https://github.com/Tobyzai/EndoSRR)                |
| Film Removal     | 2024 | CVPR                                                         | [Pytorch](https://github.com/jqtangust/filmremoval)          |
| DHAN-SHR         | 2024 | MM                                                           | [Pytorch](https://github.com/CXH-Research/DHAN-SHR)          |

## Metric

### Full-Reference

[torchmetrics](https://github.com/Lightning-AI/torchmetrics) for cuda calculation

**PSNR**

**SSIM**
