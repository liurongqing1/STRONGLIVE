name=$1
scale=$2
i=$3

#other online->other
rm -rf /home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/bin/

/home/srteam/ffmpeg/ffmpeg -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/HR/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265 /home/srteam/lrq/EDSR-PyTorch/video/${name}.265  -y

python /home/srteam/lrq/EDSR-PyTorch/src/main.py  --model FMEN  --epoch 2  --test_every 25  --batch_size 64 --lr 1e-4  --decay 18-19 --data_range 1-${i}/1-${i} --patch_size 400 --scale $scale --save FMEN_x${scale}  --chop --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/model_DIV2K.pt    
python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale $scale --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/model_best.pt
python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale $scale --save $name --save_results --dir_demo $name --data_test $name --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/test.pt --test_only
/home/srteam/ffmpeg/ffmpeg -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/${name}_30_sr.h265  -y

/home/srteam/ffmpeg/ffmpeg  -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/${name}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name}.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/data/lrq/edge_dection/$name/online_${name}_bestvmaf.txt" -f null -
/home/srteam/ffmpeg/ffmpeg  -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/${name}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name}.265 -lavfi psnr=stats_file=/data/lrq/psnr_logfile.txt -f null - >&/data/lrq/edge_dection/$name/online_${name}_bestpsnr.txt
/home/srteam/ffmpeg/ffmpeg   -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/${name}_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/video/${name}.265 -lavfi ssim="stats_file=/data/lrq/ssim.log" -f null - >&/data/lrq/edge_dection/$name/online_${name}_bestssim.txt

