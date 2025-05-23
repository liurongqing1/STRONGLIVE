#!/bin/bash

mkdir /data/lrq/feature_extract/video_cut
mkdir /data/lrq/feature_extract/GT


dir="/data/lrq/feature_extract/Video"

rm -rf /data/lrq/feature_extract/dir.out 
for element in `ls $dir`
do
   file=${element%.mp4*}
   echo $file 1>> /data/lrq/feature_extract/dir.out 
done


for name in $(cat /data/lrq/feature_extract/dir.out);
do
    mkdir /data/lrq/feature_extract/video_cut/$name
    /home/srteam/ffmpeg/ffmpeg -hide_banner  -err_detect ignore_err -i /data/lrq/feature_extract/Video/${name}.mp4 -r 30 -codec:v libx264  -vsync 1  -codec:a aac  -ac 2  -ar 48k  -f segment   -preset fast  -segment_format mpegts  -segment_time 1 -force_key_frames  "expr: gte(t, n_forced * 1)" /data/lrq/feature_extract/video_cut/$name/${name}_%03d.mp4

    dir="/data/lrq/feature_extract/video_cut/$name"

    rm -rf  /data/lrq/feature_extract/cut_video_${name}.out 
    for element in `ls $dir`
    do
       file=${element%.mp4*}
       echo $file >> /data/lrq/feature_extract/cut_video_${name}.out 
    done

    for video in $(cat /data/lrq/feature_extract/cut_video_${name}.out);
    do
       mkdir /data/lrq/feature_extract/GT/$video
       /home/srteam/ffmpeg/ffmpeg  -i /data/lrq/feature_extract/video_cut/$name/${video}.mp4   /data/lrq/feature_extract/GT/$video/%03d.png
    done

done


dir="/data/lrq/feature_extract/GT/"

rm -rf /data/lrq/feature_extract/GT_dir.out 
for element in `ls $dir`
do
   echo $element 1>> /data/lrq/feature_extract/GT_dir.out 
done

for scale in $(cat /data/lrq/feature_extract/scale.out);
do
   res=$((2160/$scale))
   for name in $(cat /data/lrq/feature_extract/GT_dir.out);
   do
      mkdir /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name
      mkdir /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/HR
      mkdir /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic
      mkdir /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic/X${scale}
      mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name
      mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}
      mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}
      mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/0
      mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/30

      cp  -r  /data/lrq/feature_extract/GT/$name/* /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/HR

      ###downsample
      /home/srteam/ffmpeg/ffmpeg -i /data/lrq/feature_extract/GT/$name/%03d.png -vf scale=3840/$scale:2160/$scale -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic/X${scale}/%03d.png
   done
done
