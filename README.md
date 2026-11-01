# Holo-encoder
High-speed computer-generated holography (CGH) using autoencoder-based deep neural network


[Jiachen Wu, Kexuan Liu, Xiaomeng Sui, and Liangcai Cao, "High-speed computer-generated holography using an autoencoder-based deep neural network," Optics Letters 46, 2908-2911 (2021).](https://doi.org/10.1364/OL.425485)

Learning-based computer-generated holography (CGH) provides a rapid hologram generation approach for holographic displays. Supervised training requires a large-scale dataset with target images and corresponding holograms. We propose an autoencoder-based neural network (holoencoder) for phase-only hologram generation. Physical diffraction propagation was incorporated into the autoencoder’s decoding part. The holoencoder can automatically learn the latent encodings of phase-only holograms in an unsupervised manner. The proposed holoencoder was able to generate high-fidelity 4K resolution holograms in 0.15 s. The reconstruction results validate the good generalizability of the holoencoder, and the experiments show fewer speckles in the reconstructed image compared with the existing CGH algorithms.

## User guide

[Download the bilingual user guide (Chinese and English, PDF)](https://github.com/THUHoloLab/Holo-encoder/raw/refs/heads/main/Holo-encoder_User_Guide_Bilingual.pdf)

The guide covers MATLAB setup, prediction, untrained network generation, training, and common issues.

DIV2K dataset: [Official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

- Training images: [DIV2K_train_HR.zip](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
- Validation images: [DIV2K_valid_HR.zip](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)

Before running, please download the following network model to prediction or retraining. Matlab version: 2021a.

Related work:

- [K. Liu, J. Wu, Z. He, and L. Cao, "4K-DMDNet: diffraction model-driven network for 4K computer-generated holography," Opto-Electronic Advances 6, 220135 (2023).](https://doi.org/10.29026/oea.2023.220135)
- [Y. Luo, K. Liu, Z. He, and L. Cao, "WGCNet: wavelet global context network for speckle-free 4K holographic display with high-frequency preservation," Photonics Research 14(7), 2925–2935 (2026).](https://doi.org/10.1364/PRJ.588876)

Contact:
luoyj25@mails.tsinghua.edu.cn;
clc@tsinghua.edu.cn
