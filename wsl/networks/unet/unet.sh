# CUDA_VISIBLE_DEVICES=0 python3 run_unet.py --p 0 &
#percents=(0.005 0.01 0.05 0.1 0.25 0.5 0.75 0.9 1)
#percents=(0.01 0.05 0.25 0.75 0.9)
#percents=(0.005 0.1 0.5 1)
percents=(1)
weight=10
#percents=(0.005)
for i in "${percents[@]}"; do
     echo $i $weight
     #echo 'USING BCE'
#     CUDA_VISIBLE_DEVICES=3,5 python3 run_unet.py --p $i  --weight $weight
     #CUDA_VISIBLE_DEVICES=3,4 python3 run_unet.py --p $i  --weight $weight
     CUDA_VISIBLE_DEVICES=2,4 python3 make_model_visuals.py --p $i --w $weight
done
