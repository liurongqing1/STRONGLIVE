#!/bin/bash

echo "mkdir!"
read -p "print filename:" name
read -p "print upscale:" upscale

mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name
mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}
mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}
mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/{0..15}
mkdir /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/30

echo "upsample!"
echo  $name >> /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
echo  "time(ms)" >> /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
for((i=1; i<=1; i++))
do
  echo "$i"
  startTime_s=$(date +%s%N)
  python /home/srteam/lrq/EDSR-PyTorch/src/main.py --model FMEN2 --skip $i --scale $upscale --save $name --save_results --dir_demo $name --data_test $name --data_range 1-30 --pre_train /home/srteam/lrq/EDSR-PyTorch/experiment/FMEN_x${upscale}/model/testDIV2K.pt --test_only
  endTime_s=$(date +%s%N)
  sum=$(($endTime_s-$startTime_s))
  sumTime1=$(($sum/1000000))
  echo "Total:$sumTime1 ms"
  echo "finish ESPCN upsample and time!"

  if ((i != 0))
  then
    startTime_s=$(date +%s%N)
    j=1
    k=1
    while((j<=30 || k<=30))
    do
      echo "$j $k "
      if ((k != j))
      then
        /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic/X${upscale}/00${k}.png  -vf scale=3840:2160 -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/$i/00${k}.png
        /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic/X${upscale}/0${k}.png  -vf scale=3840:2160 -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/$i/0${k}.png  
      fi
      if((j <= k))
      then
        j=$(($j+$i+1))
        k=$(($k+1))
      else
        k=$(($k+1))
      fi
    done
    endTime_s=$(date +%s%N)
    sum=$(($endTime_s-$startTime_s))
    sumTime2=$(($sum/1000000))
    echo "Total:$sumTime2 ms"
    echo "finish BICUBIC upsample and time!"

    sumTime=$(($sumTime1+$sumTime2))
    echo "Total:$sumTime ms"
    echo  "$sumTime" >>  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
    echo "finish $i time!"
  else
    echo  "$sumTime1" >>  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
    echo "finish espcn all time!"
    continue
  fi
done

startTime_s=$(date +%s%N)
/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -i /home/srteam/lrq/EDSR-PyTorch/dataset/benchmark/$name/LR_bicubic/X${upscale}/%d.png  -vf scale=3840:2160 -sws_flags bicubic /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/30/%d.png
endTime_s=$(date +%s%N)
sum=$(($endTime_s-$startTime_s))
sumTime3=$(($sum/1000000))
echo "Total:$sumTime3 ms"
echo  "$sumTime3" >> /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
echo "finish BICUBIC all upsample and time!"



echo "transfer video!"
for(( i=0;i<=15; i++ ))
do
  /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/$i/%d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265 -x265-params "bframes=0" -keyint_min $($i+1) -g $($i+1) -sc_threshold 0 /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/$i.h265
  echo "finsh $i!!"
done

/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -start_number 001 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/30/%d.png -vf fps=20 -pix_fmt yuv420p -c:v libx265 /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/30.h265
echo "finsh 30!!"




echo "calculating psnr+vmaf!"
for(( i=0;i<=15; i++ ))
do
   /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/$i.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/${name}_265.h265 -lavfi psnr=stats_file=psnr -f null - >&/home/srteam/lrq/test_photo_motivation/$upscale/$name/psnr_${i}.txt
   /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/$i.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/${name}_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/test_photo_motivation/$upscale/$name/VMAF_${i}.txt" -f null -
   echo "finsh $i vmaf psnr for you!!"
done

/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/30.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/${name}_265.h265 -lavfi psnr=stats_file=psnr -f null - >&/home/srteam/lrq/test_photo_motivation/$upscale/$name/psnr_30.txt
/home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/30.h265 -r 20 -i /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/${name}_265.h265 -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/home/srteam/lrq/test_photo_motivation/$upscale/$name/VMAF_30.txt" -f null -
echo "finsh 30 vmaf psnr for you!!"

echo "finish calculating!"


echo "get VMAF+PSNR+SRvideo_size!"

echo  "size" >> /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
for(( i=0;i<=15; i++ ))
do
      ls -lh  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/$i.h265 | awk '{print $5}'  | tee -a  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
      echo "finsh $i after SR video size for you!!"
done
ls -lh  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/results-${name}/X${upscale}/30.h265 | awk '{print $5}'  | tee -a  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
echo "finsh 30 after SR video size for you!!"


echo  "vmaf" >> /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
for(( i=0;i<=15; i++ ))
do
      sed 's/ /\n/g' /home/srteam/lrq/test_photo_motivation/$upscale/$name/VMAF_${i}.txt | grep 'aggregateVMAF='| cut -d '=' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
      echo "finsh get VMAF_${i} for you!!"
done
sed 's/ /\n/g' /home/srteam/lrq/test_photo_motivation/$upscale/$name/VMAF_30.txt | grep 'aggregateVMAF='| cut -d '=' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
echo "finsh get VMAF_30 for you!!"


echo  "psnr" >> /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
for(( i=0;i<=15; i++ ))
do
      sed 's/ /\n/g' /home/srteam/lrq/test_photo_motivation/$upscale/$name/psnr_${i}.txt| grep 'average:'| cut -d ':' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
      echo "finsh get psnr_${i} for you!!"
done
sed 's/ /\n/g' /home/srteam/lrq/test_photo_motivation/$upscale/$name/psnr_30.txt| grep 'average:'| cut -d ':' -f 2-3 | tee -a  /home/srteam/lrq/EDSR-PyTorch/experiment/$name/${name}_${upscale}.xls
echo "finsh get psnr_30 for you!!"

echo "finish Numeric extraction!"














