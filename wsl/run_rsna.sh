CUDA_VISIBLE_DEVICES=0 wsl train --lr 1e-3 --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=1 wsl train --lr 1e-3 --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-3 --network resnet --depth 18 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-3 --network resnet --depth 34 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-3 --network resnet --depth 50 &
CUDA_VISIBLE_DEVICES=5 wsl train --lr 1e-3 --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --lr 1e-3 --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=7 wsl train --lr 1e-3 --network vgg --depth 19

CUDA_VISIBLE_DEVICES=0 wsl train --lr 1e-4 --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=1 wsl train --lr 1e-4 --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-4 --network resnet --depth 18 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-4 --network resnet --depth 34 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-4 --network resnet --depth 50 &
CUDA_VISIBLE_DEVICES=5 wsl train --lr 1e-4 --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --lr 1e-4 --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=7 wsl train --lr 1e-4 --network vgg --depth 19

CUDA_VISIBLE_DEVICES=0 wsl train --lr 1e-5 --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=1 wsl train --lr 1e-5 --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-5 --network resnet --depth 18 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-5 --network resnet --depth 34 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-5 --network resnet --depth 50 &
CUDA_VISIBLE_DEVICES=5 wsl train --lr 1e-5 --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --lr 1e-5 --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=7 wsl train --lr 1e-5 --network vgg --depth 19

CUDA_VISIBLE_DEVICES=0 wsl train --lr 1e-6 --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=1 wsl train --lr 1e-6 --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-6 --network resnet --depth 18 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-6 --network resnet --depth 34 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-6 --network resnet --depth 50 &
CUDA_VISIBLE_DEVICES=5 wsl train --lr 1e-6 --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --lr 1e-6 --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=7 wsl train --lr 1e-6 --network vgg --depth 19

CUDA_VISIBLE_DEVICES=0 wsl train --lr 1e-7 --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=1 wsl train --lr 1e-7 --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-7 --network resnet --depth 18 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-7 --network resnet --depth 34 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-7 --network resnet --depth 50 &
CUDA_VISIBLE_DEVICES=5 wsl train --lr 1e-7 --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --lr 1e-7 --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=7 wsl train --lr 1e-7 --network vgg --depth 19

CUDA_VISIBLE_DEVICES=0 wsl train --lr 1e-8 --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=1 wsl train --lr 1e-8 --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-8 --network resnet --depth 18 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-8 --network resnet --depth 34 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-8 --network resnet --depth 50 &
CUDA_VISIBLE_DEVICES=5 wsl train --lr 1e-8 --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --lr 1e-8 --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=7 wsl train --lr 1e-8 --network vgg --depth 19

CUDA_VISIBLE_DEVICES=0 wsl train --lr 1e-9 --network densenet --depth 169 &
CUDA_VISIBLE_DEVICES=1 wsl train --lr 1e-9 --network densenet --depth 121 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-9 --network resnet --depth 18 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-9 --network resnet --depth 34 &
CUDA_VISIBLE_DEVICES=4 wsl train --lr 1e-9 --network resnet --depth 50 &
CUDA_VISIBLE_DEVICES=5 wsl train --lr 1e-9 --network resnet --depth 101 &
CUDA_VISIBLE_DEVICES=6 wsl train --lr 1e-9 --network resnet --depth 152 &
CUDA_VISIBLE_DEVICES=7 wsl train --lr 1e-9 --network vgg --depth 19
