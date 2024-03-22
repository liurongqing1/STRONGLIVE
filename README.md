# STRONGLIVE

----------

STRONGLIVE, an innovative approach to enhancing mobile 4K live video streaming performance. STRONGLIVE capitalizes on recent advancements in SR to facilitate low-latency offloading and seamless real-time streaming. 
We provide scripts for reproducing all the results from our paper. You can train your model from scratch, or use a pre-trained model to enlarge your images.


## Prerequisites ##

VPF works on Linux(Ubuntu 20.04 and Ubuntu 22.04 only) and Windows
- NVIDIA display driver: 525.xx.xx or above
- CUDA Toolkit 11.2 or above
- FFMPEG（See VideoProcessingFramework https://github.com/NVIDIA/VideoProcessingFramework folder for details）
- Python 3 and above
- Pytorch (Corresponding GPU version)
- Installation of packages required for FMEN and EDSR(https://github.com/NJU-Jet/FMEN, https://github.com/sanghyun-son/EDSR-PyTorch)


## Code ##

Clone this repository into any place you want.
`git clone https://github.com/liurongqing1/STRONGLIVE`
`cd STRONGLIVE`