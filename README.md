# Holo-encoder
High-speed computer-generated holography (CGH) using autoencoder-based deep neural network


[Jiachen Wu, Kexuan Liu, Xiaomeng Sui, and Liangcai Cao, "High-speed computer-generated holography using an autoencoder-based deep neural network," Optics Letters 46, 2908-2911 (2021).](https://doi.org/10.1364/OL.425485)

Learning-based computer-generated holography (CGH) provides a rapid hologram generation approach for holographic displays. Supervised training requires a large-scale dataset with target images and corresponding holograms. We propose an autoencoder-based neural network (holoencoder) for phase-only hologram generation. Physical diffraction propagation was incorporated into the autoencoder’s decoding part. The holoencoder can automatically learn the latent encodings of phase-only holograms in an unsupervised manner. The proposed holoencoder was able to generate high-fidelity 4K resolution holograms in 0.15 s. The reconstruction results validate the good generalizability of the holoencoder, and the experiments show fewer speckles in the reconstructed image compared with the existing CGH algorithms.


Before running, please download the following network model to prediction or retraining.   

Pre-training model download: https://cloud.tsinghua.edu.cn/f/2bb6e38a426c445681da/?dl=1   
Untrained model download: https://cloud.tsinghua.edu.cn/f/429b874fdf714194bb6f/?dl=1

Recent work: [K. Liu, J. Wu, Z. He, and L. Cao, "4K-DMDNet: diffraction model-driven network for 4K computer-generated holography," Opto-Electronic Advances 6, 220135-220135 (2023).](https://www.oejournal.org/article/doi/10.29026/oea.2023.220135).

Contact:
lkx20@mails.tsinghua.edu.cn;
clc@tsinghua.edu.cn
