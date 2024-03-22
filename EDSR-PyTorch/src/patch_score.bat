#!/bin/bash

#echo "mkdir!"
#read -p "print filename:" name
#read -p "print upscale:" upscale

#mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name
#mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}
#mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}
#mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/0

python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8  --save patch_rd000 --save_results --dir_demo patch_rd000 --data_test patch_rd000 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only

for(( i=1;i<=9; i++ ))
do
  /home/srteam/ffmpeg/ffmpeg   -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/00${i}.png  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/00${i}.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/EDSR-PyTorch/VMAF_${i}.txt" -f null -
  /home/srteam/ffmpeg/ffmpeg   -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/00${i}.png  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/00${i}.png -lavfi psnr=stats_file=psnr_logfile.txt -f null - >&/home/srteam/lrq/EDSR-PyTorch/psnr_${i}.txt
  echo "finsh $i vmaf psnr for you!!"
done

for(( i=10;i<=22; i++ ))
do
  /home/srteam/ffmpeg/ffmpeg   -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/0${i}.png  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/0${i}.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/EDSR-PyTorch/VMAF_${i}.txt" -f null -
  /home/srteam/ffmpeg/ffmpeg   -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/0${i}.png  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/0${i}.png -lavfi psnr=stats_file=psnr_logfile.txt -f null - >&/home/srteam/lrq/EDSR-PyTorch/psnr_${i}.txt
  echo "finsh $i vmaf psnr for you!!"
done

echo  "vmaf" >> /home/srteam/lrq/EDSR-PyTorch/patch_rd000score.xls

for(( i=1;i<=22; i++ ))
do
      sed 's/ /\n/g' /home/srteam/lrq/EDSR-PyTorch/VMAF_${i}.txt | grep 'aggregateVMAF='| cut -d '=' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/patch_rd000score.xls
      echo "finsh get VMAF_${i} for you!!"
done

echo  "psnr" >> /home/srteam/lrq/EDSR-PyTorch/patch_rd000score.xls
for(( i=1;i<=22; i++ ))
do
      sed 's/ /\n/g' /home/srteam/lrq/EDSR-PyTorch/psnr_${i}.txt | grep 'average:'| cut -d ':' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/patch_rd000score.xls
      echo "finsh get psnr_${i} for you!!"
done