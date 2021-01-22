CUDA_VISIBLE_DEVICES=0 wsl medinet --data covid_mgh --column mRALE --variable_type continous --extension '' &
CUDA_VISIBLE_DEVICES=1 wsl medinet --data cancer_mgh --column cancer --variable_type 'categorical' --classes 3 &
CUDA_VISIBLE_DEVICES=2 wsl medinet --data cancer_dmist2 --column cancer --variable_type 'categorical' --classes 3 &
CUDA_VISIBLE_DEVICES=3 wsl medinet --data cancer_dmist3 --column cancer --variable_type 'categorical' --classes 3 &
CUDA_VISIBLE_DEVICES=4 wsl medinet --data cancer_dmist4 --column cancer --variable_type 'categorical' --classes 3
