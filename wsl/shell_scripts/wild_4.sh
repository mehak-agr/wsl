# +
#start=(0 50 100 150 200 250) 
#start=(300 350 400 450 500 550)
#start=(600 650 700 750 800 850) 
#start=(900 950 1000 1050 1100)
start=(1100 1200)
add1=25
add2=50
add3=75

for s in ${start[@]}
do 
echo $s
CUDA_VISIBLE_DEVICES=3 wsl saliency --start $s --name first &
CUDA_VISIBLE_DEVICES=4 wsl saliency --start $(($s+$add1)) --name first &
CUDA_VISIBLE_DEVICES=3 wsl saliency --start $(($s+$add2)) --name first &
CUDA_VISIBLE_DEVICES=4 wsl saliency --start $(($s+$add3)) --name first &
done
wait
