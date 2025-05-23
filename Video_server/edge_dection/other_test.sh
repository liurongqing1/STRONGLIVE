name=$1
name2=$2
scale=$3
i=$4

##/home/srteam/ffmpeg/ffmpeg -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name2/HR/%03d.png -vf fps=30 -c:v libx265 /home/srteam/lrq/EDSR-PyTorch/video/${name2}.265  -y

rm -rf /home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/bin/
#pre->other
python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale $scale --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/model_DIV2K.pt
python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale $scale --save $name2 --save_results --dir_demo $name2 --data_test $name2 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/test.pt --test_only
/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265  -y

/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name2}.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/data/lrq/edge_dection/$name/pre_${name2}_bestvmaf.txt" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name2}.265  -lavfi psnr=stats_file=/data/lrq/psnr_logfile.txt -f null - >&/data/lrq/edge_dection/$name/pre_${name2}_bestpsnr.txt
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name2}.265 -lavfi ssim="stats_file=/data/lrq/ssim.log" -f null - >&/data/lrq/edge_dection/$name/pre_${name2}_bestssim.txt


#online->other
rm -rf /home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/bin/

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py  --model FMEN  --epoch 2  --test_every 25  --batch_size 64 --lr 1e-4  --decay 18-19 --data_range 1-${i}/1-${i} --patch_size 400 --scale $scale --save FMEN_x${scale}  --chop --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/model_DIV2K.pt    
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale $scale --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/model_best.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale $scale --save $name2 --save_results --dir_demo $name2 --data_test $name2 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/0/%03d.png -vf fps=30 -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265  -y
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name2}.265  -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/data/lrq/edge_dection/$name/online_${name2}_bestvmaf.txt" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name2}.265 -lavfi psnr=stats_file=/data/lrq/psnr_logfile.txt -f null - >&/data/lrq/edge_dection/$name/online_${name2}_bestpsnr.txt
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name2/results-${name2}/X${scale}/${name2}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name2}.265  -lavfi ssim="stats_file=/data/lrq/ssim.log" -f null - >&/data/lrq/edge_dection/$name/online_${name2}_bestssim.txt

