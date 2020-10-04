#start=(0 50 100 150 200 250) 
#start=(300 350 400 450 500 550)
#start=(600 650 700 750 800 850) 
#start=(900 950 1000 1050 1100)
start=(1100 1150 1200 1250 1300)
for s in ${start[@]}
do 
echo $s
CUDA_VISIBLE_DEVICES=6 wsl wild --start $s --name first &
done
wait
