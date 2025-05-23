#!/bin/bash
#read -p "print filename:" name
name=$1
scale=$2


for(( i=1;i<=3840; i=i+200 ))
do
   for(( j=1;j<=2160; j=j+200 ))
   do
      /home/srteam/ffmpeg/ffmpeg -thread_queue_size 2048 -i /data/lrq/edge_dection/$name/blur/${i}-${j}.png -thread_queue_size 2048 -i /data/lrq/edge_dection/$name/sharp/001.png -lavfi libvmaf="model_path=/home/srteam/ffmpeg/model/vmaf_4k_v0.6.1.pkl:log_path=/data/lrq/edge_dection/$name/vmaf_${i}_${j}.txt" -f null -
      echo "finsh $i-$j vmaf for you!!"
   done
done

echo vmaf >> /data/lrq/edge_dection/$name/vmaf${scale}.xls
for(( i=1;i<=3840; i=i+200 ))
do
   for(( j=1;j<=2160; j=j+200 ))
   do
      sed 's/ /\n/g' /data/lrq/edge_dection/$name/vmaf_${i}_${j}.txt | grep 'aggregateVMAF='| cut -d '=' -f 2-3 | tee -a  /data/lrq/edge_dection/$name/vmaf${scale}.xls
      rm -rf /data/lrq/edge_dection/$name/vmaf_${i}_${j}.txt
      echo "finsh get VMAF_${i}_${j} for you!!"
   done
done
