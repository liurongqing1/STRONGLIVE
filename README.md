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


## Quickstart (Demo2) ##

You can test our framework with your video. Place your video in `folder/EDSR-PyTorch/gpucode/videoname`. We only support H264 video.  

Run the script in folder.   

`cd /EDSR-PyTorch/src`  
` python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 15 --scale 4 --save Jockey_000 --save_results --dir_demo Jockey_000 --data_test Jockey_000 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only`  

`--skip` is the selected SR frequency, `--scale` is the upsampling multiplier, `pre_train` is the pre-training model paths.  

You can find the result video from folder`/EDSR-PyTorch/gpucode/videoname`


## train model ##

Before you run the demo, please uncomment the appropriate line in that you want to execute `/EDSR-PyTorch/src/dome2.sh`   

`cd /EDSR-PyTorch/src`  
` sh demo2.sh` 


## Update log ##

- 2024/3/22 
	
	- A cloud-based process has been incorporated, i.e., low-resolution video is VPF decoded frame-by-frame, then input to the SR model or double-three interpolation algorithm for upsampling, and finally VPF is encoded according to the SR frequency.
	- VideoProcessingFramework（VPF）has been incorporated.
	- Uploaded pre-trained model of fmen 4x\6x\8x.