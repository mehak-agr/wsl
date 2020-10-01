depth=(101 50)

for d in ${depth[@]}
do
echo 'depth' $d
CUDA_VISIBLE_DEVICES=0,1 wsl retinanet --depth $d --lr 0.0001 --ID first
done
