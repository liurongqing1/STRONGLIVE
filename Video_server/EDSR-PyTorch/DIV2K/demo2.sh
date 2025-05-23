#4x
##train'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --decay 20-40 --scale 8 --save FMEN_x8 --chop --patch_size 192
##re_train
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --epoch 100 --lr 5e-4 --decay 20-40 --data_range 1-30/1-30  --scale 6 --save FMEN_x6 --chop  --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/testDIV2K.pt

##reparameterize'''
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 4 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_best.pt

##origin+f-test'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --scale 4 --save Jockey --save_results --data_test Jockey --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_best.pt --test_only

##re+f2-test'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 4 --save Jockey --save_results --dir_demo Jockey --data_test Jockey --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 4 --save City --save_results --dir_demo City --data_test City --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 4 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only

##skip
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 3 --scale 8 --save Cal --dir_demo Cal --data_test Cal --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only


#6x
##train'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN  --epoch 100 --lr 5e-4 --decay 20-40 --data_range 1-30/1-30 --scale 4 --save FMEN_x4 --chop 
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN  --epoch 100 --lr 5e-4 --decay 20-40 --data_range 1-30/1-30 --patch_size 200 --scale 8 --save FMEN_x8 --chop 
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN  --epoch 100 --lr 5e-4 --decay 20-40 --data_range 1-30/1-30 --scale 8 --save FMEN_x8 --chop 
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN  --epoch 100 --lr 5e-4 --decay 20-40 --data_range 1-30/1-30 --scale 12 --save FMEN_x12 --chop 

##re_train
#2160 30

#rm -rf /home/srteam/lrq/EDSR-PyTorch/dataset/DIV2K/bin/
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py  --model FMEN  --epoch 2  --test_every 25  --batch_size 64 --lr 1e-4  --decay 18-19 --data_range 1-121/1-121 --patch_size 400 --scale 4 --save FMEN_x4  --chop --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_DIV2K.pt #--load FMEN_x4 model_DIV2K2.pt
python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 4 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_best.pt

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py  --model FMEN  --epoch 2  --test_every 25  --batch_size 64 --lr 1e-4  --decay 18-19 --data_range 1-121/1-121 --patch_size 396 --scale 6 --save FMEN_x6  --chop --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_DIV2K.pt #--load FMEN_x4 model_DIV2K2.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 6 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_best.pt

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py  --model FMEN  --epoch 2  --test_every 25  --batch_size 64 --lr 1e-4  --decay 18-19 --data_range 1-121/1-121 --patch_size 400 --scale 4 --save FMEN_x4  --chop --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_DIV2K.pt #--load FMEN_x4 model_DIV2K2.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 4 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_best.pt

#ԭ
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 8 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_DIV2K.pt


#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/0/%03d.png -r 30 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Jockey_015/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/0/%03d.png -r 30 -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Jockey_015/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8  --save patch_rd000 --save_results --dir_demo patch_rd000 --data_test patch_rd000 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/%03d.png -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/EDSR-PyTorch/VMAF.txt" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/%03d.png -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -


#edge
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Joceky_edge  --save_results --dir_demo Joceky_edge --data_test Joceky_edge --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/experiment/Joceky_edge/results-Joceky_edge/X8/0/%03d.png  -r 30  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Joceky_edge/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/experiment/Joceky_edge/results-Joceky_edge/X8/0/%03d.png  -r 30  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Joceky_edge/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/experiment/Joceky_edge/results-Joceky_edge/X8/0/%03d.png  -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Joceky_edge/HR/%03d.png -lavfi ssim="stats_file=ssim.log" -f null -


#EDVRBMP
for(( i=0;i<=0; i++ ))
do
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Village_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 6 --save Village_00${i} --save_results --dir_demo Village_00${i} --data_test Village_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Village_00${i}/results-Village_00${i}/X6/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Village_00${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Ready_0${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 6 --save Ready_0${i} --save_results --dir_demo Ready_0${i} --data_test Ready_0${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_0${i}/results-Ready_0${i}/X6/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Ready_0${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Bigmouse_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 6 --save Bigmouse_00${i} --save_results --dir_demo Bigmouse_00${i} --data_test Bigmouse_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bigmouse_00${i}/results-Bigmouse_00${i}/X6/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Bigmouse_00${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Jockey_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 6 --save Jockey_00${i} --save_results --dir_demo Jockey_00${i} --data_test Jockey_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_00${i}/results-Jockey_00${i}/X6/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Jockey_00${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Bee_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 6 --save Bee_00${i} --save_results --dir_demo Bee_00${i} --data_test Bee_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_00${i}/results-Bee_00${i}/X6/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/360P_Bee_00${i}/%08d.bmp
  
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Bee_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 4 --save Bee_00${i} --save_results --dir_demo Bee_00${i} --data_test Bee_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_00${i}/results-Bee_00${i}/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Bee_00${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Jockey_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 4 --save Jockey_00${i} --save_results --dir_demo Jockey_00${i} --data_test Jockey_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_00${i}/results-Jockey_00${i}/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Jockey_00${i}/%08d.bmp 
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Village_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 4 --save Village_00${i} --save_results --dir_demo Village_00${i} --data_test Village_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Village_00${i}/results-Village_00${i}/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Village_00${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Bigmouse_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 4 --save Bigmouse_00${i} --save_results --dir_demo Bigmouse_00${i} --data_test Bigmouse_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bigmouse_00${i}/results-Bigmouse_00${i}/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Bigmouse_00${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Ready_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 4 --save Ready_00${i} --save_results --dir_demo Ready_00${i} --data_test Ready_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_00${i}/results-Ready_00${i}/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Ready_00${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Ready_0${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 4 --save Ready_0${i} --save_results --dir_demo Ready_0${i} --data_test Ready_0${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_0${i}/results-Ready_0${i}/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Ready_0${i}/%08d.bmp
   #mkdir /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_LOL_00${i}
   #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 4 --save LOL_00${i} --save_results --dir_demo LOL_00${i} --data_test LOL_00${i} --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
   #/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/LOL_00${i}/results-LOL_00${i}/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_LOL_00${i}/%08d.bmp

done


#Bosp
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Bosp_001 --save_results --dir_demo Bosp_001 --data_test Bosp_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/results-Bosp_001/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/results-Bosp_001/X8/Bosp_001_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/results-Bosp_001/X8/Bosp_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/Bosp_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/results-Bosp_001/X8/Bosp_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/Bosp_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/results-Bosp_001/X8/Bosp_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bosp_001/Bosp_001.265 -lavfi ssim="stats_file=ssim.log" -f null -


#Man
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Man_001 --save_results --dir_demo Man_001 --data_test Man_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/results-Man_001/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/results-Man_001/X8/Man_001_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/results-Man_001/X8/Man_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/Man_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/results-Man_001/X8/Man_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/Man_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/results-Man_001/X8/Man_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Man_001/Man_001.265 -lavfi ssim="stats_file=ssim.log" -f null -

#nv
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save nv_000 --save_results --dir_demo nv_000 --data_test nv_000 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/results-nv_000/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/results-nv_000/X8/nv_000_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/results-nv_000/X8/nv_000_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/nv_000.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/results-nv_000/X8/nv_000_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/nv_000.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/results-nv_000/X8/nv_000_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/nv_000/nv_000.265 -lavfi ssim="stats_file=ssim.log" -f null -

#Jockey
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_000 --save_results --dir_demo Jockey_000 --data_test Jockey_000 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/results-Jockey_000/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/results-Jockey_000/X8/Jockey_000_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/results-Jockey_000/X8/Jockey_000_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/Jockey_000.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/results-Jockey_000/X8/Jockey_000_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/Jockey_000.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/results-Jockey_000/X8/Jockey_000_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_000/Jockey_000.265 -lavfi ssim="stats_file=ssim.log" -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_001 --save_results --dir_demo Jockey_001 --data_test Jockey_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/Jockey_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/Jockey_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/Jockey_001.265 -lavfi ssim="stats_file=ssim.log" -f null -

#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/0/%03d.png -r 30 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Jockey_001/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_009 --save_results --dir_demo Jockey_009 --data_test Jockey_009 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi ssim="stats_file=ssim.log" -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_015 --save_results --dir_demo Jockey_015 --data_test Jockey_015 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/Jockey_015.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/Jockey_015.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/Jockey_015.265 -lavfi ssim="stats_file=ssim.log" -f null -

#Ready
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8  --save Ready_001 --save_results --dir_demo Ready_001 --data_test Ready_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/Ready_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/Ready_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/Ready_001.265 -lavfi ssim="stats_file=ssim.log" -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Ready_009 --save_results --dir_demo Ready_009 --data_test Ready_009 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/Ready_009.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/VMAF.txt" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/Ready_009.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/Ready_009.265 -lavfi ssim="stats_file=ssim.log" -f null -

#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/0/%03d.png -r 30 -thread_queue_size 2048  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready_009/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/0/%03d.png -r 30 -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready_009/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -

#Bee
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Bee_002 --save_results --dir_demo Bee_002 --data_test Bee_002 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/results-Bee_002/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/results-Bee_002/X8/Bee_002_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/results-Bee_002/X8/Bee_002_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/Bee_002.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/results-Bee_002/X8/Bee_002_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/Bee_002.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/results-Bee_002/X8/Bee_002_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bee_002/Bee_002.265 -lavfi ssim="stats_file=ssim.log" -f null -


#2160 6
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --epoch 2 --lr 1e-4  --data_range 1-30/1-30 --skip 5 --patch_size 2160  --scale 6 --save FMEN_x6 --chop  --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_DIV2K.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 6 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_best.pt

#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/0/%03d.png -r 30 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Hand_001/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/0/%03d.png -r 30 -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Hand_001/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -

#Hand
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 6 --save Hand_001 --save_results --dir_demo Hand_001 --data_test Hand_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_best.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_5_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_5_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/Hand_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_5_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/Hand_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_5_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/Hand_001.265 -lavfi ssim="stats_file=ssim.log" -f null -

#360 30
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --epoch 2 --lr 1e-4 --data_range 1-30/1-30 --patch_size 360 --scale 6 --save FMEN_x6 --chop  --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_DIV2K.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 6 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_best.pt

#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/0/%03d.png -r 30 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Hand_001/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/0/%03d.png -r 30 -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Hand_001/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -

#Hand
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 6 --save Hand_001 --save_results --dir_demo Hand_001 --data_test Hand_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_best.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_360p_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_360p_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/Hand_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_360p_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/Hand_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/results-Hand_001/X6/Hand_001_360p_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Hand_001/Hand_001.265 -lavfi ssim="stats_file=ssim.log" -f null -

#ԭ
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 8 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_DIV2K.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8  --save patch_rd000 --save_results --dir_demo patch_rd000 --data_test patch_rd000 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/%03d.png -r 30 -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/%03d.png -r 30 -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -

#edge
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Joceky_edge  --save_results --dir_demo Joceky_edge --data_test Joceky_edge --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/experiment/Joceky_edge/results-Joceky_edge/X8/0/%03d.png  -r 30  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Joceky_edge/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/experiment/Joceky_edge/results-Joceky_edge/X8/0/%03d.png  -r 30  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Joceky_edge/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/experiment/Joceky_edge/results-Joceky_edge/X8/0/%03d.png  -r 30 -i  /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Joceky_edge/HR/%03d.png -lavfi ssim="stats_file=ssim.log" -f null -

#Jockey
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_001 --save_results --dir_demo Jockey_001 --data_test Jockey_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/Jockey_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/Jockey_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/Jockey_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/Jockey_001.265 -lavfi ssim="stats_file=ssim.log" -f null -

#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/0/%03d.png -r 30 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Jockey_001/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_009 --save_results --dir_demo Jockey_009 --data_test Jockey_009 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/results-Jockey_009/X8/Jockey_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_009/Jockey_009.265 -lavfi ssim="stats_file=ssim.log" -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Jockey_015 --save_results --dir_demo Jockey_015 --data_test Jockey_015 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/Jockey_015.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/Jockey_015.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/results-Jockey_015/X8/Jockey_015_div_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_015/Jockey_015.265 -lavfi ssim="stats_file=ssim.log" -f null -

#Ready
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8  --save Ready_001 --save_results --dir_demo Ready_001 --data_test Ready_001 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/Ready_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/Ready_001.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/results-Ready_001/X8/Ready_001_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_001/Ready_001.265 -lavfi ssim="stats_file=ssim.log" -f null -

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip 0 --scale 8 --save Ready_009 --save_results --dir_demo Ready_009 --data_test Ready_009 --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/0/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/Ready_009.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/VMAFyuan.txt" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/Ready_009.265 -lavfi psnr=stats_file=psnr_logfile.txt -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/Ready_009_30_sr.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/Ready_009.265 -lavfi ssim="stats_file=ssim.log" -f null -

#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/0/%03d.png -r 30 -thread_queue_size 2048  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready_009/HR/%03d.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready_009/results-Ready_009/X8/0/%03d.png -r 30 -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready_009/HR/%03d.png -lavfi psnr=stats_file=psnr_logfile.txt -f null -

##reparameterize'''
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 4 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_best.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 6 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/model_best.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 8 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_best.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 12 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x12/model/model_best.pt

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 4 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 6 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 8 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only

#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 6 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/testDIV2K.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 6 --save Ready --save Bigbosp --save_results --dir_demo Bigbosp --data_test Bigbosp  --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/testDIV2K.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 8 --save Ready --save_results --dir_demo Ready --data_test Ready  --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/testDIV2K.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 12 --save City --save_results --dir_demo City --data_test City --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x12/model/test.pt --test_only


#8x
##train'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN  --decay 20-40  --scale 8 --save FMEN_x8 --chop --patch_size 192

##reparameterize'''
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 8 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_best.pt
##re+f2-test'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 8 --save Jockey --save_results --dir_demo Jockey --data_test Jockey --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 8 --save City --save_results --dir_demo City --data_test City --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 8 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 12 --save Bosp --save_results --dir_demo Bosp --data_test Bosp --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x12/model/testDIV2K.pt --test_only


##origin+f-test'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --scale 8 --save Jockey --save_results --dir_demo Jockey --data_test Jockey --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/model_best.pt --test_only


#12x
##train'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --scale 12 --save FMEN_x12 --chop --patch_size 192
##reparameterize'''
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 12 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x12/model/model_best.pt
##re+f2-test'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 12 --save Jockey --save_results --dir_demo Jockey --data_test Jockey --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x12/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 12 --save City --save_results --dir_demo City --data_test City --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x12/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 12 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x12/model/test.pt --test_only

#14x
##train'''
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN --scale 14 --save FMEN_x14 --chop --patch_size 182
##reparameterize'''
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --scale 14 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x14/model/model_best.pt



#16x


##change model structre
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --down_blocks 2 --model FMEN --scale 4 --save FMEN_x4 --chop --patch_size 192
##reparameterize'''
#python /home/srteam/lrq/EDSR-PyTorch/src/reparameterize.py --down_blocks 2 --scale 4 --pretrained_path /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/model_best.pt
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2  --skip 0 --scale 4 --save Jockey --save_results --dir_demo Jockey --data_test Jockey --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only


#bicubic
#python /home/srteam/lrq/EDSR-PyTorch/src/Bicubic.py --scale 12


#sr pp
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 4 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/testDIV2K.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 6 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/testDIV2K.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 8 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/testDIV2K.pt --test_only
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X4/0/%d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X4/0.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/0/%d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/0.h265  
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X8/0/%d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X8/0.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X4/0.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/Ready_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/0.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/Ready_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X8/0.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/Ready_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -

#bicubic pp
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Jockey_001/LR_bicubic/X8/%03d.png  -vf scale=3840:2160 -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/30/%03d.png
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/30/%03d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/30.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/results-Jockey_001/X8/30.h265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Jockey_001/Jockey_001.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -

#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready/LR_bicubic/X4/%d.png  -vf scale=3840:2160 -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X4/30/%d.png
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/Ready/LR_bicubic/X6/%d.png  -vf scale=3840:2160 -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/30/%d.png
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X4/30/%d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X4/30.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/30/%d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/30.h265  
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/30/%d.png -vf fps=30 -pix_fmt yuv420p -c:v libx265  /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/30.h265
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X4/30.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/Ready_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/30.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/Ready_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X8/30.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/Ready_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -

#bmp pp
#/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Bear_000/results-Bear_000/X4/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/EDVR/540P_Bear_000/%08d.bmp
#/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/0/%d.png  /home/srteam/lrq/EDSR-PyTorch/experiment/Ready/results-Ready/X6/0/%08d.bmp
#/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/experiment/patch_rd000/results-patch_rd000/X8/0/%08d.bmp

#/home/srteam/ffmpeg/ffmpeg  -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/%03d.png  /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/patch_rd000/HR/%08d.bmp
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/patch_rd000_L1_270Pto4k30_gap1_edvr.265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/patch_rd000_hr_270Pto4k30_gap1_edvr.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -
#/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/patch_rd000_270Pto4k30_gap1_edvr.265 -r 30 -i /home/srteam/lrq/EDSR-PyTorch/patch_rd000_hr_270Pto4k30_gap1_edvr.265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl" -f null -


