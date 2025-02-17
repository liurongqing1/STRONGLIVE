#!/bin/bash

for((i=1; i<=101; i=i+10))
do
  python /home/srteam/lrq/EDSR-PyTorch/src/main.py  --model FMEN  --epoch $i  --test_every 25  --batch_size 64 --lr 1e-4  --decay 18-19 --data_range 1-121/1-121 --patch_size 400 --scale 8 --save FMEN_x8  --chop --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_DIV2K.pt #--load FMEN_x4
  python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 8 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_best.pt

  python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_009 --save_results --dir_demo Jockey_009 --data_test Jockey_009 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only 
  echo y|/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265  
  /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
  /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null - >&/home/srteam/lrq/EDSR-PyTorch/retrain_vmafup/vmaf_${i}.txt
done

echo  "vmaf" >> /home/srteam/lrq/EDSR-PyTorch/retrain_vmafup/improve.xls
for((i=1; i<=101; i=i+10))
do
  sed 's/ /\n/g' /home/srteam/lrq/EDSR-PyTorch/retrain_vmafup/vmaf_${i}.txt | grep 'aggregateVMAF='| cut -d '=' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/retrain_vmafup/improve.xls
  echo "finsh get VMAF_${i} for you!!"
done
