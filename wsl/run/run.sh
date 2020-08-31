CUDA_VISIBLE_DEVICES=5 wsl train --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=6 wsl train --network resnet --depth 18

CUDA_VISIBLE_DEVICES=5 wsl train --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=6 wsl train --network resnet --depth 34

CUDA_VISIBLE_DEVICES=5 wsl train --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --network resnet --depth 50

CUDA_VISIBLE_DEVICES=5 wsl train --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=6 wsl train --network vgg --depth 19