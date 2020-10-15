# +
#start=(0 50 100 150 200 250) 
#start=(300 350 400 450 500 550)
start=(600 650 700 750 800 850) 
#start=(900 950 1000 1050 1100)
add=25

for s in ${start[@]}
do 
echo $s
CUDA_VISIBLE_DEVICES=2 wsl saliency --start $s --name first &
CUDA_VISIBLE_DEVICES=7 wsl saliency --start $(($s+$add)) --name first &
done
wait
