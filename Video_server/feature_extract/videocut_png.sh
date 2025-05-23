#!/bin/bash

dir="/home/srteam/lrq/feature_extract/GT/"

rm -rf /home/srteam/lrq/feature_extract/GT_dir.out 
for element in `ls $dir`
do
   echo $element 1>> /home/srteam/lrq/feature_extract/GT_dir.out 
done

for scale in $(cat /home/srteam/lrq/EDSR-PyTorch/scale.out);
do
   res=$((2160/$scale))
   for name in $(cat /home/srteam/lrq/feature_extract/GT_dir.out);
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


      cp  -r  /home/srteam/lrq/feature_extract/GT/$name/* /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/HR

      ###downsample
      /home/srteam/ffmpeg/ffmpeg -i /home/srteam/lrq/feature_extract/GT/$name/%03d.png -vf scale=3840/$scale:2160/$scale -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic/X${scale}/%03d.png
      ###upsample bicubic
      #/home/srteam/ffmpeg/ffmpeg -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic/X${scale}/%03d.png -vf scale=3840:2160 -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/30/%03d.png
      ###bicubic video
      #/home/srteam/ffmpeg/ffmpeg -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/30/%03d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265 /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/${name}_${res}Pto4k30.265

      ###SR
      dir2=$dir$name
      cd "$dir2"
      fileNum=`ls -l |grep "^-"|wc -l`
      #python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale $scale --save $name --save_results --dir_demo $name  --data_test $name  --data_range 1-${fileNum} --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${scale}/model/testDIV2K.pt --test_only
      ###bmp
      #/home/srteam/ffmpeg/ffmpeg -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/0/%03d.png  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${scale}/0/%08d.bmp
   done
done
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 4 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x4/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 6 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x6/model/test.pt --test_only
#python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --scale 8 --save Ready --save_results --dir_demo Ready --data_test Ready --data_range 1-60 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x8/model/test.pt --test_only

