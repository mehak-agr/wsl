# +
lr=(0.001 0.00001 0.0000001)
k=(1 2 4 8 16 32 64)
alpha=(0.0 0.2 0.4 0.6 0.8 1.0)
maps=(1 2 4 8)

for l in ${lr[@]}
do
for r in ${k[@]}
do
for a in ${alpha[@]}
do
for m in ${maps[@]}
do
echo $l $r $a $m
CUDA_VISIBLE_DEVICES=0 wsl medinet --lr $l --network resnet --depth 50 --wildcat --k $r --alpha $a --maps $m --ID first
done
done
done
done
