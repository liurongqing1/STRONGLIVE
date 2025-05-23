#!/usr/bin/env bash
rm -rf /home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/bin/
read -p "print test_filename:" name
#name=$1

for(( i=400;i<=400; i=i+200 ))
do
   start_time=$(date +%s)
   /home/srteam/anaconda3/envs/VMAF2/bin/python /home/srteam/lrq/EDSR-PyTorch/src/main.py  --model FMEN  --epoch 2  --test_every 25  --batch_size 64 --lr 1e-4  --decay 18-19 --data_range 1-21/1-21 --patch_size ${i} --scale 8 --save FMEN_x8  --chop --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_DIV2K2.pt #--load FMEN_x4
   current_time=$(date +%s)
   time_diff=$((current_time - start_time))
   echo  $time_diff
   #echo  $time_diff >> /home/srteam/lrq/EDSR-PyTorch/${name}_train_time_res.xls

   /home/srteam/anaconda3/envs/VMAF2/bin/python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 8 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_best.pt
   /home/srteam/anaconda3/envs/VMAF2/bin/python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save $name --save_results --dir_demo $name --data_test $name --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
   /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X8/${name}_30_sr.h265  -y

   /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X8/${name}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/EDSR-PyTorch/${name}_${i}_vmaf.txt" -f null -
   /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X8/${name}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}.265 -lavfi psnr=stats_file=/home/srteam/lrq/EDSR-PyTorch/psnr_logfile.txt -f null -  
   /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X8/${name}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}.265  -lavfi ssim=stats_file=/home/srteam/lrq/EDSR-PyTorch/ssim_logfile.txt -f null -

   #sed 's/ /\n/g' /home/srteam/lrq/EDSR-PyTorch/${name}_${i}_vmaf.txt | grep 'aggregateVMAF='| cut -d '=' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/${name}_vmaf_res.xls

   rm -rf /home/srteam/lrq/EDSR-PyTorch/${name}_${i}_vmaf.txt
   rm -rf /home/srteam/lrq/EDSR-PyTorch/psnr_logfile.txt  
   rm -rf /home/srteam/lrq/EDSR-PyTorch/ssim_logfile.txt   
done

